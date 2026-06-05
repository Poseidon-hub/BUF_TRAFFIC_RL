from typing import Optional

import numpy as np

from .baseline import FixedTimeController
from .checkpointing import build_scenario_signature, checkpoint_file_exists
from .config import ACTION_SIZE
from .dqn import DQNShared, model_info_from_config
from .logging_utils import save_json, write_csv_row
from .objectives import add_normalized_metrics, objective_payload, objective_improvement_pct, weighted_mobility_score
from .perf_utils import ProgressBar, should_log
from .progress_utils import ProgressPrinter
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
    total_eval_episodes = max(1, len(eval_seeds) * int(episodes))
    is_smoke = str(getattr(config, "run_mode", "")).lower() == "smoke"
    progress = ProgressBar(
        total=total_eval_episodes,
        desc=f"Evaluation {output_name}",
        enabled=False and should_log(config, "compact") and not should_log(config, "verbose") and not is_smoke,
    )
    completed = 0

    try:
        for seed in eval_seeds:
            set_seed(seed)
            for episode in range(int(episodes)):
                row = _run_single_episode(config, mode=normalized_mode, seed=seed, episode=episode)
                episode_rows.append(row)
                completed += 1
                progress.update(completed, suffix=f"episode {completed}/{total_eval_episodes}")
                if getattr(config, "save_eval_logs", True):
                    write_csv_row(config.logs_dir / f"eval_{output_name}.csv", _csv_safe_row(row))
    finally:
        progress.close()

    metrics = _aggregate_episode_rows(normalized_mode, episode_rows)
    metrics["run_mode"] = str(getattr(config, "run_mode", ""))
    if getattr(config, "save_eval_logs", True):
        save_json(config.logs_dir / f"eval_{output_name}.json", metrics)
        if normalized_mode == "fixed_native" and output_name != "fixed":
            save_json(config.logs_dir / "eval_fixed.json", metrics)
        if normalized_mode == "fixed_native" and output_name == "fixed":
            save_json(config.logs_dir / "eval_fixed_native.json", metrics)
    return metrics


def evaluate_many_seeds(config, seeds, episode_seconds: int, mode: str = "rl") -> dict:
    safe_mkdir(config.logs_dir)
    original_seeds = list(getattr(config, "eval_seeds", None) or [config.seed])
    original_seconds = int(getattr(config, "episode_seconds", episode_seconds))
    rows = []
    try:
        for seed in list(seeds):
            config.eval_seeds = [int(seed)]
            config.episode_seconds = int(episode_seconds)
            rl = evaluate(config, mode=mode, episodes=1)
            fixed = evaluate(config, mode="fixed_native", episodes=1)
            add_normalized_metrics(rl)
            add_normalized_metrics(fixed)
            save_json(config.logs_dir / f"eval_rl_seed_{int(seed)}.json", rl)
            save_json(config.logs_dir / f"eval_fixed_native_seed_{int(seed)}.json", fixed)
            rl_score = weighted_mobility_score(rl, baseline=fixed)
            fixed_score = weighted_mobility_score(fixed, baseline=fixed)
            improvement = objective_improvement_pct(fixed_score, rl_score)
            comparison = {
                "seed": int(seed),
                "rl_objective_score": rl_score,
                "fixed_objective_score": fixed_score,
                "objective_score_improvement_pct": improvement,
                "queue_improvement_pct": _improvement_pct(fixed.get("avg_queue", 0.0), rl.get("avg_queue", 0.0)),
                "waiting_time_improvement_pct": _improvement_pct(
                    fixed.get("avg_waiting_time", 0.0),
                    rl.get("avg_waiting_time", 0.0),
                ),
                "avg_time_loss_improvement_pct": _improvement_pct(
                    fixed.get("avg_time_loss", 0.0),
                    rl.get("avg_time_loss", 0.0),
                ),
                "throughput_delta": rl.get("throughput", 0.0) - fixed.get("throughput", 0.0),
                "rl_better_by_objective": bool(rl_score < fixed_score),
            }
            save_json(config.logs_dir / f"comparison_seed_{int(seed)}.json", comparison)
            rows.append(comparison)
    finally:
        config.eval_seeds = original_seeds
        config.episode_seconds = original_seconds

    def values(key):
        return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]

    summary = {
        "seeds": [int(seed) for seed in seeds],
        "number_of_seeds_total": len(rows),
        "number_of_seeds_rl_better": sum(1 for row in rows if row.get("rl_better_by_objective")),
        "per_seed": rows,
    }
    for key in (
        "queue_improvement_pct",
        "waiting_time_improvement_pct",
        "avg_time_loss_improvement_pct",
        "throughput_delta",
        "objective_score_improvement_pct",
    ):
        vals = values(key)
        summary[key + "_mean"] = float(np.mean(vals)) if vals else None
        summary[key + "_std"] = float(np.std(vals)) if vals else None
        summary[key + "_min"] = float(np.min(vals)) if vals else None
        summary[key + "_max"] = float(np.max(vals)) if vals else None
    save_json(config.logs_dir / "comparison_summary.json", summary)
    print()
    print("Multi-seed comparison RL vs native fixed-time")
    print("---------------------------------------------")
    print(f"seeds: {summary['seeds']}")
    print(f"avg_queue_improvement_pct_mean: {summary.get('queue_improvement_pct_mean')}")
    print(f"avg_waiting_time_improvement_pct_mean: {summary.get('waiting_time_improvement_pct_mean')}")
    print(f"avg_time_loss_improvement_pct_mean: {summary.get('avg_time_loss_improvement_pct_mean')}")
    print(f"throughput_delta_mean: {summary.get('throughput_delta_mean')}")
    print(f"objective_score_improvement_pct_mean: {summary.get('objective_score_improvement_pct_mean')}")
    print(
        f"rl_better_by_objective: {summary['number_of_seeds_rl_better']}/"
        f"{summary['number_of_seeds_total']}"
    )
    return summary


def _normalize_mode(mode: str) -> str:
    return "fixed_native" if mode == "fixed" else str(mode)


def _improvement_pct(baseline_value, rl_value):
    baseline_value = float(baseline_value or 0.0)
    rl_value = float(rl_value or 0.0)
    if abs(baseline_value) <= 1e-9:
        return None
    return (baseline_value - rl_value) / abs(baseline_value) * 100.0


def _output_mode_name(requested_mode: str, normalized_mode: str) -> str:
    return "fixed" if requested_mode == "fixed" else normalized_mode


def _progress_label_for_mode(mode: str) -> str:
    if mode == "rl":
        return "Evaluation RL"
    if mode == "fixed_native":
        return "Evaluation fixed"
    if mode == "real_timing":
        return "Evaluation real timing"
    return f"Evaluation {mode}"


def _run_single_episode(config, mode: str, seed: int, episode: int) -> dict:
    env: Optional[SumoMultiAgentEnv] = None
    agent: Optional[DQNShared] = None
    progress: Optional[ProgressPrinter] = None
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
        progress = ProgressPrinter(
            total=int(config.episode_seconds),
            label=_progress_label_for_mode(mode),
            style=getattr(config, "progress_bar_style", "single_line"),
            min_interval=0.5,
            min_percent_delta=1.0,
            enabled=should_log(config, "compact") and not should_log(config, "verbose"),
        )
        progress.update(0, suffix=f"t=0/{int(config.episode_seconds)}", force=True)

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
        total_time_loss = 0.0
        avg_distance_values = []
        total_distance = 0.0
        avg_vehicle_waiting_values = []
        total_vehicle_waiting_time = 0.0
        legacy_total_waiting_time = 0.0
        cumulative_waiting_vehicle_seconds = 0.0
        sum_vehicle_waiting_time_samples = 0.0
        final_total_waiting_time = 0.0
        avg_waiting_time_per_vehicle_sample = 0.0
        vehicles_seen = 0
        vehicle_metric_notes = []
        total_reward = 0.0
        total_waiting_time = 0.0
        throughput = 0
        departed = 0
        arrived = 0
        loaded_total = 0
        running = 0
        current_vehicle_count = 0
        running_values = []
        info = {}
        q_hold_values = []
        q_switch_values = []
        greedy_hold_count = 0
        greedy_switch_count = 0
        eval_tie_break_switch_count = 0
        eval_tie_break_switch_by_tls = {}
        vehicle_reward_components = []
        pedestrian_reward_components = []
        pedestrian_reward_shares = []

        while not done:
            if mode == "rl":
                actions = {}
                decision_interval = max(
                    1,
                    int(
                        getattr(
                            config,
                            "control_decision_interval",
                            getattr(config, "decision_interval", 1),
                        )
                    ),
                )
                is_env_decision_step = int(getattr(env, "step_count", 0)) % decision_interval == 0
                for tls_id, tls_obs in obs.items():
                    q_values = agent.q_values(tls_obs)
                    if len(q_values) >= ACTION_SIZE:
                        q_hold_values.append(float(q_values[0]))
                        q_switch_values.append(float(q_values[1]))
                        if int(np.argmax(q_values)) == 1:
                            greedy_switch_count += 1
                        else:
                            greedy_hold_count += 1
                    action = agent.act(
                        tls_obs,
                        epsilon=float(getattr(config, "eval_epsilon", 0.0)),
                    )
                    if is_env_decision_step and _should_eval_switch_tie_break(config, env, tls_id, q_values, action):
                        action = 1
                        eval_tie_break_switch_count += 1
                        eval_tie_break_switch_by_tls[tls_id] = (
                            eval_tie_break_switch_by_tls.get(tls_id, 0) + 1
                        )
                    actions[tls_id] = action
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
            total_time_loss = float(info.get("total_time_loss", total_time_loss))
            avg_distance_values.append(float(info.get("avg_distance", 0.0)))
            total_distance = float(info.get("total_distance", total_distance))
            avg_vehicle_waiting_values.append(float(info.get("avg_vehicle_waiting_time", 0.0)))
            total_vehicle_waiting_time = float(
                info.get("total_vehicle_waiting_time", total_vehicle_waiting_time)
            )
            if info.get("vehicle_metrics_note"):
                vehicle_metric_notes.append(str(info.get("vehicle_metrics_note")))
            vehicle_reward_components.append(float(info.get("avg_vehicle_reward_component", 0.0)))
            pedestrian_reward_components.append(float(info.get("avg_pedestrian_reward_component", 0.0)))
            pedestrian_reward_shares.append(float(info.get("pedestrian_reward_share", 0.0)))
            total_waiting_time = float(info.get("total_waiting_time", total_waiting_time))
            legacy_total_waiting_time = float(
                info.get("legacy_total_waiting_time", total_waiting_time)
            )
            cumulative_waiting_vehicle_seconds = float(
                info.get("cumulative_waiting_vehicle_seconds", cumulative_waiting_vehicle_seconds)
            )
            sum_vehicle_waiting_time_samples = float(
                info.get("sum_vehicle_waiting_time_samples", sum_vehicle_waiting_time_samples)
            )
            final_total_waiting_time = float(
                info.get("final_total_waiting_time", final_total_waiting_time)
            )
            avg_waiting_time_per_vehicle_sample = float(
                info.get(
                    "avg_waiting_time_per_vehicle_sample",
                    avg_waiting_time_per_vehicle_sample,
                )
            )
            vehicles_seen = int(info.get("vehicles_seen", vehicles_seen))
            throughput = int(info["throughput"])
            departed = int(info.get("departed", 0))
            arrived = int(info.get("arrived", throughput))
            loaded_total += int(info.get("loaded", 0))
            running = int(info.get("running", info.get("current_vehicle_count", 0)))
            current_vehicle_count = int(info.get("current_vehicle_count", running))
            running_values.append(current_vehicle_count)
            step += 1
            if progress is not None:
                sim_time = int(min(float(info.get("sim_time", step)), float(config.episode_seconds)))
                progress.update(sim_time, suffix=f"t={sim_time}/{int(config.episode_seconds)}")

        action_stats = info.get("action_stats", _empty_action_stats())
        action_stats = _with_eval_policy_stats(
            action_stats,
            eval_tie_break_switch_count,
            eval_tie_break_switch_by_tls,
        )
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
        evaluation_mode = (
            str(getattr(config, "evaluation_mode", "rl_greedy"))
            if mode == "rl"
            else mode
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
            "evaluation_mode": evaluation_mode,
            "eval_epsilon": float(getattr(config, "eval_epsilon", 0.0)) if mode == "rl" else 0.0,
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
            "legacy_total_waiting_time": legacy_total_waiting_time,
            "cumulative_waiting_vehicle_seconds": cumulative_waiting_vehicle_seconds,
            "sum_vehicle_waiting_time_samples": sum_vehicle_waiting_time_samples,
            "final_total_waiting_time": final_total_waiting_time,
            "avg_waiting_time_per_vehicle_sample": avg_waiting_time_per_vehicle_sample,
            "vehicles_seen": vehicles_seen,
            "total_reward": total_reward,
            "throughput": throughput,
            "departed": departed,
            "arrived": arrived,
            "loaded": loaded_total,
            "running": running,
            "running_max": max(running_values) if running_values else 0,
            "current_vehicle_count": current_vehicle_count,
            "avg_speed": float(np.mean(speed_values)) if speed_values else 0.0,
            "avg_time_loss": float(np.mean(time_loss_values)) if time_loss_values else 0.0,
            "total_time_loss": total_time_loss,
            "avg_distance": float(np.mean(avg_distance_values)) if avg_distance_values else 0.0,
            "total_distance": total_distance,
            "avg_vehicle_waiting_time": (
                float(np.mean(avg_vehicle_waiting_values)) if avg_vehicle_waiting_values else 0.0
            ),
            "total_vehicle_waiting_time": total_vehicle_waiting_time,
            "vehicle_metrics_note": "; ".join(sorted(set(vehicle_metric_notes))),
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
        if progress is not None:
            progress.close(suffix="done")
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


def _should_eval_switch_tie_break(config, env, tls_id: str, q_values, action: int) -> bool:
    if int(action) == 1:
        return False
    margin = float(getattr(config, "eval_switch_tie_break_margin", 0.0) or 0.0)
    if margin <= 0.0 or len(q_values) < ACTION_SIZE:
        return False
    hold_q = float(q_values[0])
    switch_q = float(q_values[1])
    if hold_q - switch_q > margin:
        return False
    if int(getattr(env, "num_phases_per_tls", {}).get(tls_id, 1)) <= 1:
        return False
    time_since_switch = float(
        getattr(env, "time_since_switch_by_tls", {}).get(tls_id, 0.0) or 0.0
    )
    if time_since_switch < float(getattr(config, "min_green", 0) or 0):
        return False
    return True


def _with_eval_policy_stats(action_stats: dict, tie_break_count: int, tie_break_by_tls: dict) -> dict:
    result = dict(action_stats or {})
    per_tls = {
        str(tls_id): dict(values or {})
        for tls_id, values in (result.get("per_tls") or {}).items()
    }
    result["per_tls"] = per_tls
    result["eval_tie_break_switch_count"] = int(tie_break_count or 0)
    for tls_id, count in (tie_break_by_tls or {}).items():
        current = per_tls.setdefault(
            str(tls_id),
            {
                "hold": 0,
                "switch": 0,
                "blocked_by_min_green_count": 0,
                "phase_set_count": 0,
                "hold_phase_set_count": 0,
                "switch_phase_set_count": 0,
            },
        )
        current["eval_tie_break_switch_count"] = int(count or 0)
    return result


def _aggregate_episode_rows(mode: str, rows: list) -> dict:
    metrics_values = {
        "avg_queue": _mean(rows, "avg_queue"),
        "avg_waiting_time": _mean(rows, "avg_waiting_time"),
        "total_waiting_time": _mean(rows, "total_waiting_time"),
        "legacy_total_waiting_time": _mean(rows, "legacy_total_waiting_time"),
        "cumulative_waiting_vehicle_seconds": _mean(rows, "cumulative_waiting_vehicle_seconds"),
        "sum_vehicle_waiting_time_samples": _mean(rows, "sum_vehicle_waiting_time_samples"),
        "final_total_waiting_time": _mean(rows, "final_total_waiting_time"),
        "avg_waiting_time_per_vehicle_sample": _mean(rows, "avg_waiting_time_per_vehicle_sample"),
        "vehicles_seen": _mean(rows, "vehicles_seen"),
        "total_reward": _mean(rows, "total_reward"),
        "throughput": _mean(rows, "throughput"),
        "departed": _mean(rows, "departed"),
        "arrived": _mean(rows, "arrived"),
        "loaded": _mean(rows, "loaded"),
        "running": _mean(rows, "running"),
        "running_max": _mean(rows, "running_max"),
        "current_vehicle_count": _mean(rows, "current_vehicle_count"),
        "episode_steps": _mean(rows, "episode_steps"),
        "avg_speed": _mean(rows, "avg_speed"),
        "avg_time_loss": _mean(rows, "avg_time_loss"),
        "total_time_loss": _mean(rows, "total_time_loss"),
        "avg_distance": _mean(rows, "avg_distance"),
        "total_distance": _mean(rows, "total_distance"),
        "avg_vehicle_waiting_time": _mean(rows, "avg_vehicle_waiting_time"),
        "total_vehicle_waiting_time": _mean(rows, "total_vehicle_waiting_time"),
        "phase_set_count": _mean(rows, "phase_set_count"),
    }
    metrics = {
        "mode": mode,
        "evaluation_mode": rows[0].get("evaluation_mode", mode) if rows else mode,
        "eval_epsilon": _mean(rows, "eval_epsilon"),
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
        "pedestrian_diagnostics": _aggregate_pedestrian_diagnostics(
            [row.get("pedestrian_metrics", {}).get("diagnostics", {}) for row in rows]
        ),
        "episodes_detail": rows,
    }
    add_normalized_metrics(metrics_values)
    add_normalized_metrics(metrics)
    objective = objective_payload(
        metrics,
        weights=getattr(rows[0].get("config", None), "objective_weights", None) if rows else None,
    )
    metrics.update(objective)
    metrics["metrics"].update(
        {
            key: metrics[key]
            for key in (
                "normalized_waiting_time_per_departed",
                "normalized_waiting_time_per_arrived",
                "normalized_time_loss_per_departed",
                "normalized_time_loss_per_arrived",
            )
            if key in metrics
        }
    )
    truncated = bool(
        metrics_values["running"] > max(metrics_values["arrived"], 0.0)
        or (
            metrics_values["departed"] > 0
            and metrics_values["arrived"] < metrics_values["departed"] * 0.5
        )
    )
    metrics["metric_diagnostics"] = {
        "departed": metrics_values["departed"],
        "arrived": metrics_values["arrived"],
        "running": metrics_values["running"],
        "vehicles_seen": metrics_values["vehicles_seen"],
        "legacy_total_waiting_time_definition": (
            "Legacy sample integral: each step sums vehicle.getWaitingTime() "
            "on TLS-controlled incoming lanes; this double-counts already accumulated waiting."
        ),
        "primary_objective_definition": (
            "weighted_mobility_score minimizes avg_queue, avg_waiting_time, "
            "avg_time_loss and normalized time loss, and rewards throughput."
        ),
        "truncated_episode": truncated,
    }
    if truncated:
        print(
            "WARNING: evaluation episode is truncated: many vehicles are still running. "
            "Prefer longer eval_seconds for final comparison."
        )
    metrics["network_metrics"] = {
        "current_vehicle_count": metrics_values["current_vehicle_count"],
        "running": metrics_values["running"],
        "departed": metrics_values["departed"],
        "arrived": metrics_values["arrived"],
        "loaded": metrics_values["loaded"],
        "avg_speed": metrics_values["avg_speed"],
        "avg_time_loss": metrics_values["avg_time_loss"],
        "total_time_loss": metrics_values["total_time_loss"],
        "avg_distance": metrics_values["avg_distance"],
        "total_distance": metrics_values["total_distance"],
        "avg_vehicle_waiting_time": metrics_values["avg_vehicle_waiting_time"],
        "total_vehicle_waiting_time": metrics_values["total_vehicle_waiting_time"],
        "legacy_total_waiting_time": metrics_values["legacy_total_waiting_time"],
        "cumulative_waiting_vehicle_seconds": metrics_values["cumulative_waiting_vehicle_seconds"],
        "normalized_waiting_time_per_departed": metrics.get("normalized_waiting_time_per_departed"),
        "normalized_time_loss_per_departed": metrics.get("normalized_time_loss_per_departed"),
        "note": "; ".join(sorted({row.get("vehicle_metrics_note", "") for row in rows if row.get("vehicle_metrics_note")})),
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
            "pedestrian_seen": pedestrian.get("seen", 0),
            "pedestrian_avg_speed": pedestrian.get("avg_speed"),
            "pedestrian_arrival_rate": pedestrian.get("arrival_rate"),
            "pedestrian_tls_sensitive_seen": pedestrian.get("tls_sensitive_seen", 0),
        }
    )
    if not metrics.get("pedestrian_diagnostics"):
        metrics["pedestrian_diagnostics"] = pedestrian.get("diagnostics", {})
    _add_pedestrian_phase_diagnostic(metrics, mode)
    return metrics


def _add_pedestrian_phase_diagnostic(metrics: dict, mode: str) -> None:
    if mode != "rl":
        return
    diagnostics = metrics.setdefault("pedestrian_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return
    action_stats = metrics.get("action_stats") or {}
    phase_set_count = int(action_stats.get("phase_set_count", 0) or 0)
    tls_sensitive_seen = float(diagnostics.get("tls_sensitive_persons_seen", 0) or 0)
    if phase_set_count > 0 or tls_sensitive_seen <= 0:
        return
    note = (
        "RL issued hold actions only; no TLS phase switches were applied, "
        "so pedestrian metrics may match fixed-time evaluation."
    )
    current = str(diagnostics.get("warning", "") or "")
    diagnostics["warning"] = f"{current}; {note}" if current else note
    pedestrian_metrics = metrics.get("pedestrian_metrics") or {}
    pedestrian_diagnostics = pedestrian_metrics.get("diagnostics")
    if isinstance(pedestrian_diagnostics, dict):
        pedestrian_diagnostics["warning"] = diagnostics["warning"]


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
        total["eval_tie_break_switch_count"] += int(stats.get("eval_tie_break_switch_count", 0))
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
                    "eval_tie_break_switch_count": 0,
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
            current["eval_tie_break_switch_count"] += int(
                tls_stats.get("eval_tie_break_switch_count", 0)
            )
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
        "seen": _mean_dict(items, "seen", 0.0),
        "running_max": max([int(item.get("running_max", 0)) for item in items], default=0),
        "waiting_count": _mean_dict(items, "waiting_count", None),
        "waiting_count_avg": _mean_dict(items, "waiting_count_avg", None),
        "waiting_count_max": max([int(item.get("waiting_count_max", 0) or 0) for item in items], default=0),
        "waiting_count_sum": _mean_dict(items, "waiting_count_sum", None),
        "total_waiting_time": float(np.mean(total_waiting_values)) if total_waiting_values else None,
        "total_waiting_time_samples": _mean_dict(items, "total_waiting_time_samples", None),
        "avg_waiting_time": float(np.mean(avg_waiting_values)) if avg_waiting_values else None,
        "avg_waiting_time_sampled": _mean_dict(items, "avg_waiting_time_sampled", None),
        "avg_speed": _mean_dict(items, "avg_speed", None),
        "arrival_rate": _mean_dict(items, "arrival_rate", None),
        "time_loss_approx": _mean_dict(items, "time_loss_approx", None),
        "tls_sensitive_seen": _mean_dict(items, "tls_sensitive_seen", 0.0),
        "controlled_crossing_waiting_time": _mean_dict(items, "controlled_crossing_waiting_time", 0.0),
        "waiting_time_available": waiting_available,
        "waiting_time_note": "" if waiting_available else (
            note or "TraCI person waiting time is not available in this SUMO version."
        ),
        "diagnostics": _aggregate_pedestrian_diagnostics(
            [item.get("diagnostics", {}) for item in items]
        ),
    }


def _pedestrian_metrics_from_info(info: dict) -> dict:
    return {
        "departed": int(info.get("pedestrian_departed", 0)),
        "arrived": int(info.get("pedestrian_arrived", 0)),
        "running": int(info.get("pedestrian_running", 0)),
        "seen": int(info.get("pedestrian_seen", 0)),
        "running_max": int(info.get("pedestrian_running_max", 0)),
        "waiting_count": info.get("pedestrian_waiting_count"),
        "waiting_count_avg": info.get("pedestrian_waiting_count_avg"),
        "waiting_count_max": info.get("pedestrian_waiting_count_max"),
        "waiting_count_sum": info.get("pedestrian_waiting_count_sum"),
        "total_waiting_time": info.get("pedestrian_total_waiting_time"),
        "total_waiting_time_samples": info.get("pedestrian_total_waiting_time_samples"),
        "avg_waiting_time": info.get("pedestrian_avg_waiting_time"),
        "avg_waiting_time_sampled": info.get("pedestrian_avg_waiting_time_sampled"),
        "avg_speed": info.get("pedestrian_avg_speed"),
        "arrival_rate": info.get("pedestrian_arrival_rate"),
        "time_loss_approx": info.get("pedestrian_time_loss_approx"),
        "tls_sensitive_seen": info.get("pedestrian_tls_sensitive_seen", 0),
        "controlled_crossing_waiting_time": info.get("pedestrian_controlled_crossing_waiting_time"),
        "waiting_time_available": bool(info.get("pedestrian_waiting_time_available", True)),
        "waiting_time_note": info.get("pedestrian_waiting_time_note", ""),
        "diagnostics": info.get("pedestrian_diagnostics", {}),
    }


def _aggregate_pedestrian_diagnostics(items: list) -> dict:
    clean = [item for item in items if isinstance(item, dict)]
    if not clean:
        return {}
    numeric_keys = [
        "persons_loaded",
        "persons_departed",
        "persons_arrived",
        "persons_running",
        "persons_seen",
        "waiting_samples",
        "speed_samples",
        "controlled_crossing_persons_seen",
        "tls_sensitive_persons_seen",
        "controlled_tls_sensitive_persons_seen",
        "controlled_tls_sensitive_routes",
        "controlled_tls_overlap_edges_count",
        "pedestrian_signal_link_count",
        "pedestrian_routes_count",
    ]
    result = {key: _mean_dict(clean, key, 0.0) for key in numeric_keys}
    edges = []
    for item in clean:
        for edge in item.get("pedestrian_edges_sample", []) or []:
            if edge not in edges and len(edges) < 20:
                edges.append(edge)
    controlled_edges = []
    controlled_tls_ids = []
    for item in clean:
        for edge in item.get("controlled_tls_overlap_edges_sample", []) or []:
            if edge not in controlled_edges and len(controlled_edges) < 20:
                controlled_edges.append(edge)
        for tls_id in item.get("controlled_tls_ids", []) or []:
            if tls_id not in controlled_tls_ids and len(controlled_tls_ids) < 20:
                controlled_tls_ids.append(tls_id)
    warnings = sorted({str(item.get("warning", "")) for item in clean if item.get("warning")})
    result["pedestrian_edges_sample"] = edges
    result["controlled_tls_overlap_edges_sample"] = controlled_edges
    result["controlled_tls_ids"] = controlled_tls_ids
    result["warning"] = "; ".join(warnings)
    return result


def _empty_action_stats() -> dict:
    return {
        "decision_count": 0,
        "hold_count": 0,
        "switch_count": 0,
        "blocked_by_min_green_count": 0,
        "phase_set_count": 0,
        "hold_phase_set_count": 0,
        "switch_phase_set_count": 0,
        "eval_tie_break_switch_count": 0,
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
            "action_eval_tie_break_switch_count": action_stats.get("eval_tie_break_switch_count", 0),
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
