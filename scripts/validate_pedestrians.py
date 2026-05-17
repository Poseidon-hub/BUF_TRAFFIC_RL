import json
import shutil
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import make_config
from src.logging_utils import save_json
from src.scenario import discover_scenario
from src.sumo_env import SumoMultiAgentEnv
from src.utils import add_sumo_tools_to_path


PEDESTRIAN_FIELDS = {
    "departed",
    "arrived",
    "running",
    "waiting_count",
    "total_waiting_time",
    "avg_waiting_time",
    "waiting_time_available",
    "waiting_time_note",
}


def check_existing_eval_json() -> None:
    for name in ["eval_rl.json", "eval_fixed.json"]:
        path = ROOT / "logs" / name
        assert path.exists(), f"{name} does not exist; run main.py before pedestrian validation"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "pedestrian_metrics" in data, f"{name} has no pedestrian_metrics block"
        missing = PEDESTRIAN_FIELDS.difference(data["pedestrian_metrics"].keys())
        assert not missing, f"{name} pedestrian_metrics missing fields: {sorted(missing)}"


def build_tmp_pedestrian_scenario() -> tuple[Path, Optional[str]]:
    cfg = make_config(ROOT)
    scenario = discover_scenario(cfg.scenario_dir, end=120, step_length=cfg.step_length)
    tmp = ROOT / "tmp_validation_ped_scenario"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    files = [scenario.net, *scenario.route_files, *scenario.additional_files]
    for source in files:
        if source:
            shutil.copy2(source, tmp / Path(source).name)

    add_sumo_tools_to_path()
    import sumolib

    net = sumolib.net.readNet(str(tmp / Path(scenario.net).name), withInternal=False)
    ped_edge = None
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        for lane in edge.getLanes():
            try:
                if lane.allows("pedestrian"):
                    ped_edge = edge.getID()
                    break
            except Exception:
                pass
        if ped_edge:
            break

    if not ped_edge:
        save_json(
            ROOT / "logs" / "validation_pedestrians_skipped.json",
            {"reason": "No edge allowing pedestrian was found in the current network."},
        )
        return tmp, None

    ped_route = tmp / "validation_pedestrians.rou.xml"
    ped_route.write_text(
        (
            "<routes>\n"
            '  <person id="ped_validation_0" depart="0">\n'
            f'    <walk from="{ped_edge}" to="{ped_edge}"/>\n'
            "  </person>\n"
            "</routes>\n"
        ),
        encoding="utf-8",
    )
    return tmp, ped_edge


def run_fixed_with_tmp_pedestrians(tmp: Path) -> tuple[dict, dict]:
    cfg = make_config(ROOT)
    cfg.scenario_dir_name = tmp.name
    cfg.episode_seconds = 120
    cfg.train_steps = 120
    cfg.eval_episodes = 1
    cfg.debug_scenario = False
    env = None
    try:
        env = SumoMultiAgentEnv(
            cfg.scenario_dir,
            use_gui=False,
            step_length=cfg.step_length,
            episode_seconds=cfg.episode_seconds,
            min_green=cfg.min_green,
            alpha=cfg.alpha,
            beta=cfg.beta,
            obs_cfg=cfg,
            seed=cfg.seed,
            sumo_extra_args=cfg.sumo_extra_args,
        )
        env.reset()
        lane_report = build_vehicle_lane_filter_report(env)
        info = {}
        done = False
        while not done:
            _, _, done, info = env.step({})
        pedestrian = {
            "departed": info.get("pedestrian_departed", 0),
            "arrived": info.get("pedestrian_arrived", 0),
            "running": info.get("pedestrian_running", 0),
            "waiting_count": info.get("pedestrian_waiting_count"),
            "total_waiting_time": info.get("pedestrian_total_waiting_time"),
            "avg_waiting_time": info.get("pedestrian_avg_waiting_time"),
            "waiting_time_available": info.get("pedestrian_waiting_time_available", True),
            "waiting_time_note": info.get("pedestrian_waiting_time_note", ""),
        }
        return pedestrian, lane_report
    finally:
        if env is not None:
            env.close()


def build_vehicle_lane_filter_report(env: SumoMultiAgentEnv) -> dict:
    report = {
        "vehicle_lanes_count": 0,
        "skipped_pedestrian_lanes_count": 0,
        "skipped_internal_lanes_count": 0,
        "skipped_problem_lanes_count": 0,
        "examples": [],
        "accepted_invalid_lanes": [],
    }
    accepted = {lane for lanes in env.incoming_lanes.values() for lane in lanes}
    controlled = []
    for tls_id in env.tls_ids:
        controlled.extend(env.traci.trafficlight.getControlledLanes(tls_id))
    for lane_id in sorted(set(controlled)):
        try:
            edge_id = env.traci.lane.getEdgeID(lane_id)
            allowed = set(env.traci.lane.getAllowed(lane_id))
            is_ped = allowed and allowed.issubset({"pedestrian"})
            is_internal = lane_id.startswith(":") or edge_id.startswith(":")
            is_walk = "walkingarea" in edge_id.lower() or "crossing" in edge_id.lower()
            if lane_id in accepted:
                report["vehicle_lanes_count"] += 1
                if is_ped or is_internal or is_walk:
                    report["accepted_invalid_lanes"].append(lane_id)
                continue
            if is_ped or is_walk:
                report["skipped_pedestrian_lanes_count"] += 1
            elif is_internal:
                report["skipped_internal_lanes_count"] += 1
        except Exception as exc:
            report["skipped_problem_lanes_count"] += 1
            if len(report["examples"]) < 5:
                report["examples"].append({"lane": lane_id, "error": str(exc)})
    return report


def main() -> int:
    try:
        check_existing_eval_json()
        tmp, ped_edge = build_tmp_pedestrian_scenario()
        if ped_edge is None:
            print("PEDESTRIAN VALIDATION SKIPPED: no pedestrian-capable edge")
            return 0

        pedestrian, lane_report = run_fixed_with_tmp_pedestrians(tmp)
        save_json(ROOT / "logs" / "validation_pedestrians.json", pedestrian)
        save_json(ROOT / "logs" / "vehicle_lane_filter_report.json", lane_report)
        assert int(pedestrian["departed"]) > 0, "pedestrian_departed == 0 in temporary pedestrian scenario"
        assert not lane_report["accepted_invalid_lanes"], (
            "pedestrian/internal lanes were accepted as vehicle lanes: "
            + ", ".join(lane_report["accepted_invalid_lanes"])
        )
        shutil.rmtree(tmp, ignore_errors=True)
        print("PEDESTRIAN VALIDATION PASSED")
        return 0
    except Exception as exc:
        print(f"PEDESTRIAN VALIDATION FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
