import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
TIMEOUT_SECONDS = int(os.environ.get("BFU_VALIDATE_TIMEOUT_SECONDS", "300"))
SEEDS = [
    int(item.strip())
    for item in os.environ.get("BFU_VALIDATE_SEEDS", "42").split(",")
    if item.strip()
]

VEHICLE_PRIMARY = [
    "avg_queue",
    "avg_waiting_time",
    "avg_time_loss",
    "normalized_time_loss_per_departed",
    "total_waiting_time",
    "total_time_loss",
    "avg_speed",
    "total_reward",
    "objective_score",
]
PEDESTRIAN_PRIMARY = ["avg_waiting_time", "total_waiting_time", "waiting_count"]


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    report = {"runs": [], "passed": False}
    if not _sumo_available():
        report["reason"] = "SUMO executable or TraCI/sumolib is unavailable"
        _write_report(report)
        print("VALIDATION FAILED: SUMO executable or TraCI/sumolib is unavailable")
        return 1

    for seed in SEEDS:
        result = _run_seed(seed)
        report["runs"].append(result)
        print(f"seed {seed}: {'passed' if result['passed'] else 'failed'}")

    report["passed"] = bool(report["runs"]) and all(item["passed"] for item in report["runs"])
    _write_report(report)
    print("METRICS VALIDATION PASSED" if report["passed"] else "METRICS VALIDATION FAILED")
    return 0 if report["passed"] else 1


def _sumo_available() -> bool:
    if shutil.which("sumo") is None and not os.environ.get("SUMO_HOME"):
        return False
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401
    except Exception:
        return False
    return True


def _run_seed(seed: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "BFU_NON_INTERACTIVE": "1",
            "BFU_RUN_MODE": "validate_fast",
            "BFU_FORCE_RETRAIN": "1",
            "BFU_SEED": str(seed),
            "BFU_DEVICE": "cpu",
            "BFU_PROGRESS_BAR_STYLE": "single_line",
        }
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process.pid)
        stdout, stderr = process.communicate()
        return {
            "seed": seed,
            "passed": False,
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"main.py timed out after {TIMEOUT_SECONDS} seconds",
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

    runtime = time.perf_counter() - started
    (LOGS / f"validation_seed_{seed}_stdout.txt").write_text(stdout, encoding="utf-8")
    (LOGS / f"validation_seed_{seed}_stderr.txt").write_text(stderr, encoding="utf-8")
    if process.returncode != 0:
        return {
            "seed": seed,
            "passed": False,
            "runtime_seconds": runtime,
            "reason": f"main.py exited with code {process.returncode}",
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

    rl = _load_json(LOGS / "eval_rl.json")
    fixed = _load_json(LOGS / "eval_fixed.json")
    comparison = _load_json(LOGS / "comparison_rl_vs_fixed_native.json")
    run_config = _load_json(LOGS / "run_config_resolved.json")
    checked = _check_metrics(rl, fixed, comparison, run_config)
    checked.update({"seed": seed, "runtime_seconds": runtime})
    return checked


def _check_metrics(rl: dict, fixed: dict, comparison: dict, run_config: dict) -> dict:
    failures = []
    setup = run_config.get("setup") or {}
    controlled_tls = run_config.get("controlled_tls") or []
    ped_meta = run_config.get("pedestrian_metadata") or {}

    if setup.get("mode") != "validate_fast":
        failures.append(f"unexpected run mode: {setup.get('mode')}")
    if int(setup.get("train_episodes", 0) or 0) > 1:
        failures.append("validate_fast trained more than one episode")
    if int(setup.get("train_episode_seconds", 0) or 0) > 15:
        failures.append("validate_fast training horizon is too long")
    if int(setup.get("eval_seconds", 0) or 0) > 90:
        failures.append("validate_fast evaluation horizon is too long")
    if len(controlled_tls) > 2:
        failures.append("validate_fast controls too many TLS")

    rl_actions = rl.get("action_stats") or {}
    fixed_actions = fixed.get("baseline_stats") or fixed.get("action_stats") or {}
    if int(rl_actions.get("decision_count", 0) or 0) <= 0:
        failures.append("RL decision_count is zero")
    if int(rl_actions.get("switch_count", 0) or 0) <= 0:
        failures.append("RL switch_count is zero")
    if int(rl_actions.get("phase_set_count", 0) or 0) <= 0:
        failures.append("RL phase_set_count is zero")
    if int(fixed_actions.get("phase_set_count", 0) or 0) != 0:
        failures.append("fixed-time baseline called setPhase")

    vehicle_pairs = _pairs(comparison.get("vehicle") or {}, VEHICLE_PRIMARY)
    if not _nonzero_and_different(vehicle_pairs):
        failures.append("vehicle primary metrics are zero or identical")

    rl_ped = rl.get("pedestrian_metrics") or {}
    fixed_ped = fixed.get("pedestrian_metrics") or {}
    pedestrian_loaded = max(
        _num(rl_ped.get("departed")),
        _num(rl_ped.get("running")),
        _num(fixed_ped.get("departed")),
        _num(fixed_ped.get("running")),
    )
    if int(setup.get("pedestrian_count", 0) or 0) > 0 and pedestrian_loaded <= 0:
        failures.append("pedestrians were requested, but SUMO saw no departed/running pedestrians")
    pedestrian_pairs = {
        key: {"rl": _num(rl_ped.get(key)), "fixed": _num(fixed_ped.get(key))}
        for key in PEDESTRIAN_PRIMARY
    }
    if not _nonzero_and_different(pedestrian_pairs):
        failures.append("pedestrian timing metrics are zero or identical")

    selected_ped_links = sum(int(item.get("pedestrian_signal_links", 0) or 0) for item in controlled_tls)
    generated_ped_links = int(ped_meta.get("pedestrian_signal_link_count", 0) or 0)
    if max(selected_ped_links, generated_ped_links) <= 0:
        failures.append("selected TLS have no pedestrian signal links")

    return {
        "passed": not failures,
        "reason": "; ".join(failures),
        "primary_metrics": {"vehicle": vehicle_pairs, "pedestrian": pedestrian_pairs},
        "rl_action_stats": rl_actions,
        "fixed_action_stats": fixed_actions,
        "pedestrian_status": {
            "requested": setup.get("pedestrian_count"),
            "departed_or_running": pedestrian_loaded,
            "selected_pedestrian_signal_links": selected_ped_links,
            "generated_pedestrian_signal_links": generated_ped_links,
        },
    }


def _pairs(metrics: dict, names: list[str]) -> dict:
    return {
        key: {
            "rl": _num((metrics.get(key) or {}).get("rl")),
            "fixed": _num((metrics.get(key) or {}).get("fixed")),
        }
        for key in names
        if key in metrics
    }


def _nonzero_and_different(pairs: dict[str, dict[str, float]]) -> bool:
    values = [(item["rl"], item["fixed"]) for item in pairs.values()]
    return any(abs(a) > 1e-9 or abs(b) > 1e-9 for a, b in values) and any(
        abs(a - b) > 1e-6 for a, b in values
    )


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_report(report: dict) -> None:
    (LOGS / "validation_metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
