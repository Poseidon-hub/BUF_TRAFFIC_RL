from typing import Optional

import numpy as np

from .baseline import FixedTimeController
from .checkpointing import build_scenario_signature, checkpoint_file_exists
from .config import ACTION_SIZE
from .dqn import DQNShared, model_info_from_config
from .logging_utils import save_json, write_csv_row
from .sumo_env import SumoMultiAgentEnv
from .utils import safe_mkdir, set_seed


def evaluate(config, mode: str = "rl", episodes: int = 1) -> dict:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode not in {"rl", "fixed_native", "real_timing", "fixed_override"}:
        raise ValueError("mode must be 'rl', 'fixed_native', 'real_timing' or 'fixed_override'")

    safe_mkdir(config.logs_dir)
    if normalized_mode == "rl" and not checkpoint_file_exists(config.checkpoint_path):
        raise RuntimeError(
            f"RL evaluation requires a valid checkpoint: {config.checkpoint_path}. "
            "Train the model first or run main.py."
        )

    eval_seeds = list(getattr(config, "eval_seeds", None) or [config.seed])
    episode_rows = []
    output_name = _output_mode_name(mode, normalized_mode)

    for seed in eval_seeds:
        set_seed(seed)
        for episode in range(int(episodes)):
            row = _run_single_episode(config, mode=normalized_mode, seed=seed, episode=episode)
            episode_rows.append(row)
            if getattr(config, "save_eval_logs", True):
                write_csv_row(config.logs_dir / f"eval_{output_name}.csv", _csv_safe_row(row))

    metrics = _aggregate_episode_rows(normalized_mode, episode_rows)
    if getattr(config, "save_eval_logs", True):
        save_json(config.logs_dir / f"eval_{output_name}.json", metrics)
        if normalized_mode == "fixed_native" and output_name != "fixed":
            save_json(config.logs_dir / "eval_fixed.json", metrics)
        if normalized_mode == "fixed_native" and output_name == "fixed":
            save_json(config.logs_dir / "eval_fixed_native.json", metrics)
    return metrics


def _normalize_mode(mode: str) -> str:
    return "fixed_native" if mode == "fixed" else str(mode)


def _output_mode_name(requested_mode: str, normalized_mode: str) -> str:
    return "fixed" if requested_mode == "fixed" else normalized_mode


def _run_single_episode(config, mode: str, seed: int, episode: int) -> dict:
    env: Optional[SumoMultiAgentEnv] = None
    agent: Optional[DQNShared] = None
    controller = FixedTimeController(period=20, min_green=config.min_green)

    try:
        env = SumoMultiAgentEnv(
            scenario_dir=config.scenario_dir,
            use_gui=bool(getattr(config, "use_gui", False)),
            step_length=config.step_length,
            episode_seconds=config.episode_seconds,
            min_green=config.min_green,
            alpha=config.alpha,
            beta=config.beta,
            obs_cfg=config,
            seed=seed,
            sumo_extra_args=getattr(config, "sumo_extra_args", None),
            mode=mode,
        )
        obs = env.reset()
        config.set_observation_phases(env.max_phases_global)

        if mode == "rl":
            agent = DQNShared(obs_dim=len(next(iter(obs.values()))), action_dim=ACTION_SIZE, config=config)
            try:
                agent.load(config.checkpoint_path)
            except Exception as exc:
                raise RuntimeError("Checkpoint is corrupt or incompatible. Re-run main.py to retrain.") from exc
            agent.q_net.eval()
            agent.target_net.eval()
        elif mode == "fixed_override":
            controller.reset()

        done = False
        step = 0
        queue_values = []
        wait_values = []
        speed_values = []
        time_loss_values = []
        total_reward = 0.0
        total_waiting_time = 0.0
        throughput = 0
        departed = 0
        arrived = 0
        info = {}
        q_hold_values = []
        q_switch_values = []
        greedy_hold_count = 0
        greedy_switch_count = 0
        vehicle_reward_components = []
        pedestrian_reward_components = []
        pedestrian_reward_shares = []

        while not done:
            if mode == "rl":
                actions = {}
                for tls_id, tls_obs in obs.items():
                    q_values = agent.q_values(tls_obs)
                    if len(q_values) >= ACTION_SIZE:
                        q_hold_values.append(float(q_values[0]))
                        q_switch_values.append(float(q_values[1]))
                        if int(np.argmax(q_values)) == 1:
                            greedy_switch_count += 1
                        else:
                            greedy_hold_count += 1
                    actions[tls_id] = agent.act(
                        tls_obs,
                        epsilon=float(getattr(config, "eval_epsilon", 0.0)),
                    )
            elif mode in {"fixed_native", "real_timing"}:
                actions = {}
            else:
                actions = controller.act(env)

            obs, rewards, done, info = env.step(actions)
            total_reward += float(sum(rewards.values()))
            queue_values.append(float(info["avg_queue"]))
            wait_values.append(float(info["avg_waiting_time"]))
            speed_values.append(float(info.get("avg_speed", 0.0)))
            time_loss_values.append(float(info.get("avg_time_loss", 0.0)))
            vehicle_reward_components.append(float(info.get("avg_vehicle_reward_component", 0.0)))
            pedestrian_reward_components.append(float(info.get("avg_pedestrian_reward_component", 0.0)))
            pedestrian_reward_shares.append(float(info.get("pedestrian_reward_share", 0.0)))
            total_waiting_time = float(info.get("total_waiting_time", total_waiting_time))
            throughput = int(info["throughput"])
            departed = int(info.get("departed", 0))
            arrived = int(info.get("arrived", throughput))
            step += 1

        action_stats = info.get("action_stats", _empty_action_stats())
        timing_source = info.get("timing_source", getattr(env, "timing_source", {}))
        pedestrian_metrics = _pedestrian_metrics_from_info(info)
        obs_dim = len(next(iter(obs.values()))) if obs else int(getattr(config, "obs_size", 0))
        scenario_signature = build_scenario_signature(
            env.scenario,
            tls_ids=env.tls_ids,
            obs_dim=obs_dim,
            action_dim=int(getattr(config, "action_size", ACTION_SIZE)),
            num_phases_per_tls=env.num_phases_per_tls,
        )
        model_info = (
            model_info_from_config(config, obs_dim=obs_dim, action_dim=ACTION_SIZE)
            if mode == "rl"
            else None
        )
        reward_breakdown = {
            "avg_vehicle_reward_component": (
                float(np.mean(vehicle_reward_components)) if vehicle_reward_components else 0.0
            ),
            "avg_pedestrian_reward_component": (
                float(np.mean(pedestrian_reward_components)) if pedestrian_reward_components else 0.0
            ),
            "pedestrian_reward_share": (
                float(np.mean(pedestrian_reward_shares)) if pedestrian_reward_shares else 0.0
            ),
            "last_step": info.get("reward_breakdown", {}),
        }
        return {
            "mode": mode,
            "seed": seed,
            "scenario_signature": scenario_signature,
            "episode_seconds": int(config.episode_seconds),
            "step_length": float(config.step_length),
            "sumocfg_path": str(env.scenario.sumocfg_for_mode(mode)) if env.scenario else None,
            "net_file": str(env.scenario.net) if env.scenario and env.scenario.net else None,
            "route_files": [str(path) for path in env.scenario.route_files] if env.scenario else [],
            "additional_files": [str(path) for path in env.scenario.additional_files] if env.scenario else [],
            "real_timing_file": str(env.scenario.real_timing_file)
            if env.scenario and env.scenario.real_timing_file
            else None,
            "episode": episode,
            "steps": step,
            "episode_steps": step,
            "avg_queue": float(np.mean(queue_values)) if queue_values else 0.0,
            "avg_waiting_time": float(np.mean(wait_values)) if wait_values else 0.0,
            "total_waiting_time": total_waiting_time,
            "total_reward": total_reward,
            "throughput": throughput,
            "departed": departed,
            "arrived": arrived,
            "avg_speed": float(np.mean(speed_values)) if speed_values else 0.0,
            "avg_time_loss": float(np.mean(time_loss_values)) if time_loss_values else 0.0,
            "phase_set_count": int(action_stats.get("phase_set_count", 0)),
            "action_stats": action_stats,
            "model_info": model_info,
            "reward_breakdown": reward_breakdown,
            "q_value_stats": _q_value_stats(
                q_hold_values,
                q_switch_values,
                greedy_hold_count,
                greedy_switch_count,
            ),
            "baseline_stats": _baseline_stats_for_mode(mode, action_stats, timing_source),
            "timing_source": timing_source,
            "pedestrian_metrics": pedestrian_metrics,
        }
    finally:
        if env is not None:
            env.close()
        try:
            import traci

            if traci.isLoaded():
                traci.close(False)
        except Exception:
            pass


def _baseline_stats_for_mode(mode: str, action_stats: dict, timing_source: dict) -> Optional[dict]:
    if mode == "fixed_native":
        return {
            "mode": "fixed_native",
            "phase_set_count": int(action_stats.get("phase_set_count", 0)),
            "program_set_count": 0,
            "controlled_by": "SUMO native tlLogic from net.xml",
        }
    if mode == "real_timing":
        return {
            "mode": "real_timing",
            "phase_set_count": int(action_stats.get("phase_set_count", 0)),
            "program_set_count": int(timing_source.get("program_set_count", 0) or 0),
            "controlled_by": "SUMO tlLogic from tls.add.xml",
        }
    if mode == "fixed_override":
        return {
            "mode": "fixed_override",
            "phase_set_count": int(action_stats.get("phase_set_count", 0)),
            "program_set_count": 0,
            "controlled_by": "manual FixedTimeController",
        }
    return None


def _aggregate_episode_rows(mode: str, rows: list) -> dict:
    metrics_values = {
        "avg_queue": _mean(rows, "avg_queue"),
        "avg_waiting_time": _mean(rows, "avg_waiting_time"),
        "total_waiting_time": _mean(rows, "total_waiting_time"),
        "total_reward": _mean(rows, "total_reward"),
        "throughput": _mean(rows, "throughput"),
        "departed": _mean(rows, "departed"),
        "arrived": _mean(rows, "arrived"),
        "episode_steps": _mean(rows, "episode_steps"),
        "avg_speed": _mean(rows, "avg_speed"),
        "avg_time_loss": _mean(rows, "avg_time_loss"),
        "phase_set_count": _mean(rows, "phase_set_count"),
    }
    metrics = {
        "mode": mode,
        "episodes": len(rows),
        "seed": rows[0].get("seed") if rows else None,
        "seeds": sorted({row.get("seed") for row in rows}) if rows else [],
        "scenario_signature": rows[0].get("scenario_signature") if rows else None,
        "episode_seconds": rows[0].get("episode_seconds") if rows else None,
        "step_length": rows[0].get("step_length") if rows else None,
        "sumocfg_path": rows[0].get("sumocfg_path") if rows else None,
        "net_file": rows[0].get("net_file") if rows else None,
        "route_files": rows[0].get("route_files", []) if rows else [],
        "additional_files": rows[0].get("additional_files", []) if rows else [],
        "real_timing_file": rows[0].get("real_timing_file") if rows else None,
        "metrics": metrics_values,
        **metrics_values,
        "action_stats": _aggregate_action_stats([row.get("action_stats", {}) for row in rows]),
        "model_info": rows[0].get("model_info") if rows else None,
        "reward_breakdown": _aggregate_reward_breakdown(
            [row.get("reward_breakdown", {}) for row in rows]
        ),
        "q_value_stats": _aggregate_q_value_stats([row.get("q_value_stats", {}) for row in rows]),
        "baseline_stats": _aggregate_baseline_stats([row.get("baseline_stats") for row in rows], mode),
        "timing_source": _aggregate_timing_source([row.get("timing_source", {}) for row in rows]),
        "pedestrian_metrics": _aggregate_pedestrian_metrics(
            [row.get("pedestrian_metrics", {}) for row in rows]
        ),
        "episodes_detail": rows,
    }
    pedestrian = metrics["pedestrian_metrics"]
    metrics.update(
        {
            "pedestrian_departed": pedestrian["departed"],
            "pedestrian_arrived": pedestrian["arrived"],
            "pedestrian_running": pedestrian["running"],
            "pedestrian_waiting_count": pedestrian["waiting_count"],
            "pedestrian_total_waiting_time": pedestrian["total_waiting_time"],
            "pedestrian_avg_waiting_time": pedestrian["avg_waiting_time"],
            "pedestrian_waiting_time_available": pedestrian["waiting_time_available"],
            "pedestrian_waiting_time_note": pedestrian["waiting_time_note"],
        }
    )
    return metrics


def _aggregate_action_stats(stats_list: list) -> dict:
    total = _empty_action_stats()
    for stats in stats_list:
        stats = stats or {}
        total["decision_count"] += int(stats.get("decision_count", 0))
        total["hold_count"] += int(stats.get("hold_count", 0))
        total["switch_count"] += int(stats.get("switch_count", 0))
        total["blocked_by_min_green_count"] += int(stats.get("blocked_by_min_green_count", 0))
        total["phase_set_count"] += int(stats.get("phase_set_count", 0))
        total["hold_phase_set_count"] += int(stats.get("hold_phase_set_count", 0))
        total["switch_phase_set_count"] += int(stats.get("switch_phase_set_count", 0))
        for tls_id, tls_stats in stats.get("per_tls", {}).items():
            current = total["per_tls"].setdefault(
                tls_id,
                {
                    "hold": 0,
                    "switch": 0,
                    "blocked_by_min_green_count": 0,
                    "phase_set_count": 0,
                    "hold_phase_set_count": 0,
                    "switch_phase_set_count": 0,
                },
            )
            current["hold"] += int(tls_stats.get("hold", 0))
            current["switch"] += int(tls_stats.get("switch", 0))
            current["blocked_by_min_green_count"] += int(
                tls_stats.get("blocked_by_min_green_count", 0)
            )
            current["phase_set_count"] += int(tls_stats.get("phase_set_count", 0))
            current["hold_phase_set_count"] += int(tls_stats.get("hold_phase_set_count", 0))
            current["switch_phase_set_count"] += int(tls_stats.get("switch_phase_set_count", 0))
    return total


def _aggregate_reward_breakdown(items: list) -> dict:
    clean = [item for item in items if isinstance(item, dict)]
    if not clean:
        return {
            "avg_vehicle_reward_component": 0.0,
            "avg_pedestrian_reward_component": 0.0,
            "pedestrian_reward_share": 0.0,
        }
    return {
        "avg_vehicle_reward_component": _mean_dict(clean, "avg_vehicle_reward_component", 0.0),
        "avg_pedestrian_reward_component": _mean_dict(clean, "avg_pedestrian_reward_component", 0.0),
        "pedestrian_reward_share": _mean_dict(clean, "pedestrian_reward_share", 0.0),
        "last_step": clean[-1].get("last_step", {}),
    }


def _aggregate_baseline_stats(items: list, mode: str) -> Optional[dict]:
    clean = [item for item in items if isinstance(item, dict)]
    if not clean:
        return None
    return {
        "mode": clean[0].get("mode", mode),
        "phase_set_count": sum(int(item.get("phase_set_count", 0)) for item in clean),
        "program_set_count": sum(int(item.get("program_set_count", 0) or 0) for item in clean),
        "controlled_by": clean[0].get("controlled_by", ""),
    }


def _aggregate_timing_source(items: list) -> dict:
    clean = [item for item in items if isinstance(item, dict)]
    if not clean:
        return {}
    first = dict(clean[0])
    first["program_set_count"] = sum(int(item.get("program_set_count", 0) or 0) for item in clean)
    return first


def _q_value_stats(q_hold_values: list, q_switch_values: list, greedy_hold_count: int, greedy_switch_count: int) -> dict:
    total = max(1, int(greedy_hold_count) + int(greedy_switch_count))
    return {
        "mean_q_hold": float(np.mean(q_hold_values)) if q_hold_values else None,
        "mean_q_switch": float(np.mean(q_switch_values)) if q_switch_values else None,
        "min_q_hold": float(np.min(q_hold_values)) if q_hold_values else None,
        "max_q_hold": float(np.max(q_hold_values)) if q_hold_values else None,
        "min_q_switch": float(np.min(q_switch_values)) if q_switch_values else None,
        "max_q_switch": float(np.max(q_switch_values)) if q_switch_values else None,
        "greedy_hold_ratio": float(greedy_hold_count) / total,
        "greedy_switch_ratio": float(greedy_switch_count) / total,
    }


def _aggregate_q_value_stats(items: list) -> dict:
    keys = [
        "mean_q_hold",
        "mean_q_switch",
        "min_q_hold",
        "max_q_hold",
        "min_q_switch",
        "max_q_switch",
        "greedy_hold_ratio",
        "greedy_switch_ratio",
    ]
    result = {}
    for key in keys:
        values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
        if not values:
            result[key] = None
        elif key.startswith("min_"):
            result[key] = float(min(values))
        elif key.startswith("max_"):
            result[key] = float(max(values))
        else:
            result[key] = float(np.mean(values))
    return result


def _aggregate_pedestrian_metrics(items: list) -> dict:
    waiting_available = all(item.get("waiting_time_available", True) for item in items)
    total_waiting_values = [
        item.get("total_waiting_time")
        for item in items
        if isinstance(item.get("total_waiting_time"), (int, float))
    ]
    avg_waiting_values = [
        item.get("avg_waiting_time")
        for item in items
        if isinstance(item.get("avg_waiting_time"), (int, float))
    ]
    note = next((item.get("waiting_time_note", "") for item in items if item.get("waiting_time_note")), "")
    return {
        "departed": _mean_dict(items, "departed", 0.0),
        "arrived": _mean_dict(items, "arrived", 0.0),
        "running": _mean_dict(items, "running", 0.0),
        "running_max": max([int(item.get("running_max", 0)) for item in items], default=0),
        "waiting_count": _mean_dict(items, "waiting_count", None),
        "waiting_count_sum": _mean_dict(items, "waiting_count_sum", None),
        "total_waiting_time": float(np.mean(total_waiting_values)) if total_waiting_values else None,
        "avg_waiting_time": float(np.mean(avg_waiting_values)) if avg_waiting_values else None,
        "waiting_time_available": waiting_available,
        "waiting_time_note": "" if waiting_available else (
            note or "TraCI person waiting time is not available in this SUMO version."
        ),
    }


def _pedestrian_metrics_from_info(info: dict) -> dict:
    return {
        "departed": int(info.get("pedestrian_departed", 0)),
        "arrived": int(info.get("pedestrian_arrived", 0)),
        "running": int(info.get("pedestrian_running", 0)),
        "running_max": int(info.get("pedestrian_running_max", 0)),
        "waiting_count": info.get("pedestrian_waiting_count"),
        "waiting_count_sum": info.get("pedestrian_waiting_count_sum"),
        "total_waiting_time": info.get("pedestrian_total_waiting_time"),
        "avg_waiting_time": info.get("pedestrian_avg_waiting_time"),
        "waiting_time_available": bool(info.get("pedestrian_waiting_time_available", True)),
        "waiting_time_note": info.get("pedestrian_waiting_time_note", ""),
    }


def _empty_action_stats() -> dict:
    return {
        "decision_count": 0,
        "hold_count": 0,
        "switch_count": 0,
        "blocked_by_min_green_count": 0,
        "phase_set_count": 0,
        "hold_phase_set_count": 0,
        "switch_phase_set_count": 0,
        "per_tls": {},
    }


def _mean(rows: list, key: str, default=0.0):
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return default
    return float(np.mean(values))


def _mean_dict(items: list, key: str, default=0.0):
    values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
    if not values:
        return default
    return float(np.mean(values))


def _csv_safe_row(row: dict) -> dict:
    action_stats = row.get("action_stats", {})
    pedestrian = row.get("pedestrian_metrics", {})
    timing_source = row.get("timing_source", {})
    baseline_stats = row.get("baseline_stats") or {}
    reward_breakdown = row.get("reward_breakdown") or {}
    model_info = row.get("model_info") or {}
    result = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "action_stats",
            "pedestrian_metrics",
            "timing_source",
            "baseline_stats",
            "reward_breakdown",
            "model_info",
        }
    }
    result.update(
        {
            "action_decision_count": action_stats.get("decision_count", 0),
            "action_hold_count": action_stats.get("hold_count", 0),
            "action_switch_count": action_stats.get("switch_count", 0),
            "action_blocked_by_min_green_count": action_stats.get("blocked_by_min_green_count", 0),
            "action_phase_set_count": action_stats.get("phase_set_count", 0),
            "baseline_phase_set_count": baseline_stats.get("phase_set_count", 0),
            "baseline_program_set_count": baseline_stats.get("program_set_count", 0),
            "timing_programs_loaded": timing_source.get("programs_loaded", 0),
            "timing_program_set_count": timing_source.get("program_set_count", 0),
            "algorithm": model_info.get("algorithm"),
            "use_double_dqn": model_info.get("use_double_dqn"),
            "use_dueling_dqn": model_info.get("use_dueling_dqn"),
            "avg_vehicle_reward_component": reward_breakdown.get("avg_vehicle_reward_component", 0.0),
            "avg_pedestrian_reward_component": reward_breakdown.get("avg_pedestrian_reward_component", 0.0),
            "pedestrian_reward_share": reward_breakdown.get("pedestrian_reward_share", 0.0),
            "pedestrian_departed": pedestrian.get("departed", 0),
            "pedestrian_arrived": pedestrian.get("arrived", 0),
            "pedestrian_running": pedestrian.get("running", 0),
            "pedestrian_avg_waiting_time": pedestrian.get("avg_waiting_time"),
        }
    )
    return result
