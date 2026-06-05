import shutil
import time
from typing import Optional

import numpy as np

from .checkpointing import build_checkpoint_metadata, checkpoint_file_exists, save_checkpoint_metadata
from .config import ACTION_SIZE
from .dqn import DQNShared
from .logging_utils import append_jsonl, save_json, write_csv_row
from .objectives import add_normalized_metrics, weighted_mobility_score
from .perf_utils import should_log
from .progress_utils import ProgressPrinter
from .sumo_env import SumoMultiAgentEnv
from .utils import safe_mkdir, set_seed


TRAIN_LOG_COLUMNS = [
    "algorithm",
    "use_double_dqn",
    "use_dueling_dqn",
    "global_step",
    "episode",
    "step",
    "steps",
    "episode_steps",
    "epsilon",
    "avg_reward",
    "avg_queue",
    "avg_waiting_time",
    "avg_time_loss",
    "throughput",
    "episode_avg_queue",
    "episode_avg_waiting_time",
    "total_reward",
    "objective_score",
    "loss_avg",
    "td_error_abs_avg",
    "td_error_avg",
    "q_value_avg",
    "q_value_max",
    "q_value_min",
    "q_hold_mean",
    "q_switch_mean",
    "greedy_hold_ratio",
    "greedy_switch_ratio",
    "target_q_avg",
    "replay_size",
    "target_updates_count",
    "last_target_update_step",
    "action_hold_count",
    "action_switch_count",
    "action_blocked_by_min_green_count",
    "phase_set_count",
    "avg_vehicle_reward_component",
    "avg_pedestrian_reward_component",
    "pedestrian_reward_share",
    "interaction_steps",
    "optimizer_steps",
    "device",
    "cuda_used",
]


def train(config) -> dict:
    set_seed(config.seed)
    safe_mkdir(config.checkpoints_dir)
    safe_mkdir(config.logs_dir)

    is_smoke = str(getattr(config, "run_mode", "")).lower() == "smoke"
    started = time.perf_counter()
    loop_started = None
    smoke_time_limit = float(getattr(config, "smoke_max_total_runtime_seconds", 15))
    smoke_max_steps = int(getattr(config, "smoke_max_train_steps", getattr(config, "train_steps", 60)))
    if is_smoke:
        config.train_steps = min(int(config.train_steps), smoke_max_steps)

    train_log_path = config.logs_dir / "train_metrics.csv"
    train_jsonl_path = config.logs_dir / "train_metrics.jsonl"
    env: Optional[SumoMultiAgentEnv] = None
    agent: Optional[DQNShared] = None
    obs = None
    total_steps = 0
    episode_idx = 0
    episode_steps = 0
    episode_reward = 0.0
    recent_rewards = []
    recent_losses = []
    recent_td_errors = []
    recent_q_avgs = []
    recent_q_max = []
    recent_q_min = []
    recent_q_hold = []
    recent_q_switch = []
    recent_greedy_hold = 0
    recent_greedy_switch = 0
    recent_target_q = []
    recent_vehicle_components = []
    recent_pedestrian_components = []
    recent_pedestrian_shares = []
    queue_values = []
    wait_values = []
    progress = None
    smoke_next_progress_step = 0
    last_loss = ""
    optimizer_steps = 0
    learning_started = False
    best_validation_score = None

    if getattr(config, "save_train_logs", True):
        _reset_train_logs_if_schema_changed(train_log_path, train_jsonl_path)

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
            seed=config.seed,
            sumo_extra_args=getattr(config, "sumo_extra_args", None),
        )
        obs = env.reset()
        obs_dim = len(next(iter(obs.values())))
        config.set_observation_phases(env.max_phases_global)
        agent = DQNShared(obs_dim=obs_dim, action_dim=ACTION_SIZE, config=config)
        progress = ProgressPrinter(
            total=int(config.train_steps),
            label="Training",
            style=getattr(config, "progress_bar_style", "single_line"),
            min_interval=0.5,
            min_percent_delta=1.0,
            enabled=(
                should_log(config, "compact")
                and not should_log(config, "verbose")
            ),
        )
        expected_episodes = max(1, int(getattr(config, "train_episodes", 1) or 1))
        progress.update(
            0,
            suffix=f"step 0/{int(config.train_steps)} | episode 1/{expected_episodes} | eps={agent.epsilon(0):.2f}",
            force=True,
        )
        smoke_next_progress_step = int(getattr(config, "smoke_progress_update_every_steps", 1) or 1)

        loop_started = time.perf_counter()
        while total_steps < config.train_steps:
            if is_smoke and total_steps > 0 and time.perf_counter() - loop_started > smoke_time_limit:
                if should_log(config, "compact"):
                    print("Smoke train time limit reached.")
                break
            epsilon = agent.epsilon(total_steps)
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
            is_decision_step = total_steps % decision_interval == 0
            actions = {}
            if is_decision_step:
                for tls_id, tls_obs in obs.items():
                    q_values = agent.q_values(tls_obs)
                    if len(q_values) >= ACTION_SIZE:
                        recent_q_hold.append(float(q_values[0]))
                        recent_q_switch.append(float(q_values[1]))
                        if int(np.argmax(q_values)) == 1:
                            recent_greedy_switch += 1
                        else:
                            recent_greedy_hold += 1
                    actions[tls_id] = agent.act(tls_obs, epsilon=epsilon)
            min_switch_prob = float(getattr(config, "min_switch_action_prob_during_train", 0.0))
            if is_decision_step and min_switch_prob > 0.0 and actions:
                for tls_id in list(actions.keys()):
                    if np.random.random() < min_switch_prob:
                        actions[tls_id] = 1
            next_obs, rewards, done, info = env.step(actions)
            global_step = total_steps + 1
            agent.global_step = global_step
            mean_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            episode_reward += float(sum(rewards.values()))
            episode_steps += 1

            for tls_id in env.tls_ids:
                agent.store(
                    obs[tls_id],
                    actions.get(tls_id, 0),
                    rewards.get(tls_id, 0.0),
                    next_obs[tls_id],
                    done,
                )

            update_metrics = None
            replay_len = len(agent.replay)
            can_update = (
                replay_len >= int(config.start_learning_after)
                and replay_len >= int(config.batch_size)
                and global_step % int(config.train_freq) == 0
            )
            if can_update:
                update_metrics = agent.update()
                if update_metrics is not None:
                    optimizer_steps += 1
                    learning_started = True
                    recent_losses.append(update_metrics.get("loss", 0.0))
                    recent_td_errors.append(update_metrics.get("td_error_avg", 0.0))
                    recent_q_avgs.append(update_metrics.get("q_value_avg", 0.0))
                    recent_q_max.append(update_metrics.get("q_value_max", 0.0))
                    recent_q_min.append(update_metrics.get("q_value_min", 0.0))
                    recent_target_q.append(update_metrics.get("target_q_avg", 0.0))
            if global_step > 0 and global_step % config.target_update_freq == 0:
                agent.update_target(global_step=global_step)

            recent_rewards.append(mean_reward)
            queue_values.append(float(info["avg_queue"]))
            wait_values.append(float(info["avg_waiting_time"]))
            recent_vehicle_components.append(float(info.get("avg_vehicle_reward_component", 0.0)))
            recent_pedestrian_components.append(float(info.get("avg_pedestrian_reward_component", 0.0)))
            recent_pedestrian_shares.append(float(info.get("pedestrian_reward_share", 0.0)))
            if total_steps % config.log_freq == 0 or done or total_steps + 1 >= config.train_steps:
                action_stats = info.get("action_stats", {})
                objective_metrics = {
                    "avg_queue": info.get("episode_avg_queue", info.get("avg_queue", 0.0)),
                    "avg_waiting_time": info.get(
                        "episode_avg_waiting_time",
                        info.get("avg_waiting_time", 0.0),
                    ),
                    "avg_time_loss": info.get("avg_time_loss", 0.0),
                    "total_time_loss": info.get("total_time_loss", 0.0),
                    "throughput": info.get("throughput", 0.0),
                    "departed": info.get("departed", 0.0),
                }
                add_normalized_metrics(objective_metrics)
                greedy_total = max(1, recent_greedy_hold + recent_greedy_switch)
                row = {
                    "algorithm": getattr(config, "algorithm", "dqn"),
                    "use_double_dqn": bool(getattr(config, "use_double_dqn", False)),
                    "use_dueling_dqn": bool(getattr(config, "use_dueling_dqn", False)),
                    "global_step": global_step,
                    "episode": episode_idx,
                    "step": total_steps,
                    "steps": total_steps + 1,
                    "episode_steps": episode_steps,
                    "epsilon": epsilon,
                    "avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                    "avg_queue": info["avg_queue"],
                    "avg_waiting_time": info["avg_waiting_time"],
                    "avg_time_loss": info.get("avg_time_loss", 0.0),
                    "throughput": info.get("throughput", 0),
                    "episode_avg_queue": float(np.mean(queue_values)) if queue_values else 0.0,
                    "episode_avg_waiting_time": float(np.mean(wait_values)) if wait_values else 0.0,
                    "total_reward": episode_reward,
                    "objective_score": weighted_mobility_score(
                        objective_metrics,
                        weights=getattr(config, "objective_weights", None),
                        normalization=getattr(config, "objective_normalization", None),
                    ),
                    "loss_avg": float(np.mean(recent_losses)) if recent_losses else "",
                    "td_error_abs_avg": float(np.mean(recent_td_errors)) if recent_td_errors else "",
                    "td_error_avg": float(np.mean(recent_td_errors)) if recent_td_errors else "",
                    "q_value_avg": float(np.mean(recent_q_avgs)) if recent_q_avgs else "",
                    "q_value_max": float(np.max(recent_q_max)) if recent_q_max else "",
                    "q_value_min": float(np.min(recent_q_min)) if recent_q_min else "",
                    "q_hold_mean": float(np.mean(recent_q_hold)) if recent_q_hold else "",
                    "q_switch_mean": float(np.mean(recent_q_switch)) if recent_q_switch else "",
                    "greedy_hold_ratio": float(recent_greedy_hold) / greedy_total,
                    "greedy_switch_ratio": float(recent_greedy_switch) / greedy_total,
                    "target_q_avg": float(np.mean(recent_target_q)) if recent_target_q else "",
                    "replay_size": len(agent.replay) if agent is not None else 0,
                    "target_updates_count": int(getattr(agent, "target_updates_count", 0)),
                    "last_target_update_step": int(getattr(agent, "last_target_update_step", 0)),
                    "action_hold_count": action_stats.get("hold_count", 0),
                    "action_switch_count": action_stats.get("switch_count", 0),
                    "action_blocked_by_min_green_count": action_stats.get(
                        "blocked_by_min_green_count", 0
                    ),
                    "phase_set_count": action_stats.get("phase_set_count", 0),
                    "avg_vehicle_reward_component": (
                        float(np.mean(recent_vehicle_components)) if recent_vehicle_components else 0.0
                    ),
                    "avg_pedestrian_reward_component": (
                        float(np.mean(recent_pedestrian_components)) if recent_pedestrian_components else 0.0
                    ),
                    "pedestrian_reward_share": (
                        float(np.mean(recent_pedestrian_shares)) if recent_pedestrian_shares else 0.0
                    ),
                    "interaction_steps": global_step,
                    "optimizer_steps": optimizer_steps,
                    "device": str(getattr(agent, "device", "")),
                    "cuda_used": str(getattr(agent, "device", "")) == "cuda",
                }
                if getattr(config, "save_train_logs", True):
                    write_csv_row(train_log_path, row)
                    append_jsonl(train_jsonl_path, row)
                    reward_breakdown = dict(info.get("reward_breakdown", {}) or {})
                    reward_row = {
                        "global_step": global_step,
                        "episode": episode_idx,
                        "reward_variant": reward_breakdown.get(
                            "reward_variant",
                            getattr(config, "reward_variant", ""),
                        ),
                        "queue_penalty": reward_breakdown.get("queue_penalty", 0.0),
                        "wait_penalty": reward_breakdown.get("wait_penalty", 0.0),
                        "neighbor_penalty": reward_breakdown.get("neighbor_penalty", 0.0),
                        "pressure_abs_norm": reward_breakdown.get("pressure_abs_norm", 0.0),
                        "pressure_improvement": reward_breakdown.get("pressure_improvement", 0.0),
                        "queue_improvement": reward_breakdown.get("queue_improvement", 0.0),
                        "wait_improvement": reward_breakdown.get("wait_improvement", 0.0),
                        "time_loss_component": reward_breakdown.get("time_loss_component", 0.0),
                        "throughput_bonus": reward_breakdown.get("throughput_bonus", 0.0),
                        "switch_penalty": reward_breakdown.get("switch_penalty", 0.0),
                        "stuck_phase_penalty": reward_breakdown.get("stuck_phase_penalty", 0.0),
                        "vehicle_component": reward_breakdown.get("vehicle_component", reward_breakdown.get("avg_vehicle_reward_component", 0.0)),
                        "pedestrian_wait_component": reward_breakdown.get("pedestrian_wait_component", reward_breakdown.get("avg_pedestrian_reward_component", 0.0)),
                        "pedestrian_share": reward_breakdown.get("pedestrian_share", reward_breakdown.get("pedestrian_reward_share", 0.0)),
                        "total_reward": reward_breakdown.get("total_reward", 0.0),
                    }
                    write_csv_row(config.logs_dir / "train_reward_components.csv", reward_row)
                last_loss = row["loss_avg"]
                if should_log(config, "verbose"):
                    print(
                        "train "
                        f"episode={episode_idx} step={total_steps} epsilon={epsilon:.3f} "
                        f"reward={episode_reward:.3f} loss={row['loss_avg']} "
                        f"q_avg={row['q_value_avg']} target_updates={row['target_updates_count']} "
                        f"ped_reward_share={row['pedestrian_reward_share']}"
                    )
                elif progress is not None:
                    expected_episodes = max(1, int(getattr(config, "train_episodes", 1) or 1))
                    progress.update(
                        global_step,
                        suffix=(
                            f"step {global_step}/{int(config.train_steps)} | "
                            f"episode {episode_idx + 1}/{expected_episodes} | "
                            f"loss={row['loss_avg']} | eps={epsilon:.3f}"
                        ),
                    )
                recent_rewards.clear()
                recent_losses.clear()
                recent_td_errors.clear()
                recent_q_avgs.clear()
                recent_q_max.clear()
                recent_q_min.clear()
                recent_q_hold.clear()
                recent_q_switch.clear()
                recent_greedy_hold = 0
                recent_greedy_switch = 0
                recent_target_q.clear()
                recent_vehicle_components.clear()
                recent_pedestrian_components.clear()
                recent_pedestrian_shares.clear()

            obs = next_obs
            total_steps += 1
            if is_smoke and should_log(config, "compact") and progress is not None:
                update_every = max(1, int(getattr(config, "smoke_progress_update_every_steps", 1) or 1))
                if total_steps >= smoke_next_progress_step or total_steps >= int(config.train_steps):
                    loss_text = last_loss if last_loss not in {"", None} else "n/a"
                    expected_episodes = max(1, int(getattr(config, "train_episodes", 1) or 1))
                    progress.update(
                        total_steps,
                        suffix=(
                            f"step {total_steps}/{int(config.train_steps)} | "
                            f"episode {episode_idx + 1}/{expected_episodes} | "
                            f"optimizer_steps={optimizer_steps} | loss={loss_text} | eps={epsilon:.3f}"
                        ),
                    )
                    smoke_next_progress_step = total_steps + update_every
            if done and total_steps < config.train_steps:
                completed_episode = episode_idx + 1
                validation_every = int(getattr(config, "validation_every_episodes", 0) or 0)
                if (
                    not is_smoke
                    and validation_every > 0
                    and completed_episode % validation_every == 0
                ):
                    candidate_meta = build_checkpoint_metadata(
                        config=config,
                        scenario=env.scenario,
                        tls_ids=env.tls_ids,
                        obs_dim=obs_dim,
                        num_phases_per_tls=env.num_phases_per_tls,
                    )
                    candidate_path = config.checkpoints_dir / "dqn_last.pt"
                    candidate_meta_path = config.checkpoints_dir / "dqn_last_meta.json"
                    best_path = config.checkpoints_dir / "dqn_best.pt"
                    best_meta_path = config.checkpoints_dir / "dqn_best_meta.json"
                    agent.save(candidate_path)
                    agent.save(config.checkpoint_path)
                    save_checkpoint_metadata(candidate_meta_path, candidate_meta)
                    save_checkpoint_metadata(config.checkpoint_meta_path, candidate_meta)
                    best_validation_score = _validate_checkpoint_candidate(
                        config=config,
                        candidate_path=candidate_path,
                        candidate_meta_path=candidate_meta_path,
                        best_path=best_path,
                        best_meta_path=best_meta_path,
                        episode_number=completed_episode,
                        current_best_score=best_validation_score,
                    )
                obs = env.reset()
                episode_idx += 1
                episode_steps = 0
                episode_reward = 0.0
                queue_values.clear()
                wait_values.clear()

        agent.update_target(global_step=total_steps)
        if total_steps <= 0 and not is_smoke:
            raise RuntimeError("Обучение не выполнило ни одного шага.")
        if len(agent.replay) == 0:
            raise RuntimeError("Replay buffer пустой после обучения; checkpoint не сохраняется.")
        if is_smoke and optimizer_steps <= 0 and len(agent.replay) > 0:
            old_batch_size = int(agent.batch_size)
            try:
                agent.batch_size = max(1, min(old_batch_size, len(agent.replay)))
                update_metrics = agent.update()
                if update_metrics is not None:
                    optimizer_steps += 1
                    learning_started = True
                    last_loss = update_metrics.get("loss", "")
            finally:
                agent.batch_size = old_batch_size
        metadata = build_checkpoint_metadata(
            config=config,
            scenario=env.scenario,
            tls_ids=env.tls_ids,
            obs_dim=obs_dim,
            num_phases_per_tls=env.num_phases_per_tls,
        )
        metadata.update(
            {
                "run_mode": str(getattr(config, "run_mode", "")),
                "smoke_training": bool(is_smoke),
                "train_steps_completed": int(total_steps),
                "interaction_steps": int(total_steps),
                "optimizer_steps": int(optimizer_steps),
                "learning_started": bool(learning_started),
                "note": (
                    "Smoke mode is a functional check; metrics are not meaningful."
                    if is_smoke
                    else ""
                ),
                "training_seconds": round(time.perf_counter() - started, 6),
                "device": str(getattr(agent, "device", "")),
                "cuda_device_name": _cuda_device_name(agent),
            }
        )
        last_path = config.checkpoints_dir / "dqn_last.pt"
        best_path = config.checkpoints_dir / "dqn_best.pt"
        last_meta_path = config.checkpoints_dir / "dqn_last_meta.json"
        best_meta_path = config.checkpoints_dir / "dqn_best_meta.json"
        agent.save(last_path)
        agent.save(config.checkpoint_path)
        save_checkpoint_metadata(config.checkpoint_meta_path, metadata)
        save_checkpoint_metadata(last_meta_path, metadata)
        if not checkpoint_file_exists(config.checkpoint_path):
            raise RuntimeError(f"Checkpoint не был создан: {config.checkpoint_path}")
        if not is_smoke and int(getattr(config, "validation_every_episodes", 0) or 0) > 0:
            best_validation_score = _validate_checkpoint_candidate(
                config=config,
                candidate_path=last_path,
                candidate_meta_path=last_meta_path,
                best_path=best_path,
                best_meta_path=best_meta_path,
                episode_number=episode_idx + 1,
                current_best_score=best_validation_score,
            )
        if is_smoke or best_validation_score is None or not best_path.exists():
            shutil.copy2(last_path, best_path)
            shutil.copy2(last_meta_path, best_meta_path)
        shutil.copy2(best_path, config.checkpoint_path)
        shutil.copy2(best_meta_path, config.checkpoint_meta_path)
        metadata["best_checkpoint_path"] = str(best_path)
        metadata["last_checkpoint_path"] = str(last_path)
        metadata["best_validation_score"] = best_validation_score
        save_json(config.checkpoints_dir / "config.json", config.to_json_dict())
        if is_smoke and should_log(config, "compact"):
            if optimizer_steps <= 0:
                print("Smoke training finished without optimizer step; checkpoint saved for pipeline test.")
            if progress is not None:
                progress.close(
                    suffix=(
                        f"step {total_steps}/{int(config.train_steps)} | "
                        f"optimizer_steps={optimizer_steps} | checkpoint saved"
                    )
                )
        elif should_log(config, "compact"):
            if progress is not None:
                progress.close(
                    suffix=(
                        f"step {total_steps}/{int(config.train_steps)} | "
                        f"optimizer_steps={optimizer_steps} | checkpoint saved"
                    )
                )
            print("Training finished.")
        return {
            "checkpoint_path": str(config.checkpoint_path),
            "meta_path": str(config.checkpoint_meta_path),
            "episodes_done": episode_idx + 1,
            "total_steps": total_steps,
            "interaction_steps": total_steps,
            "optimizer_steps": optimizer_steps,
            "final_epsilon": agent.epsilon(total_steps),
            "train_metrics_path": str(train_log_path),
            "training_seconds": round(time.perf_counter() - started, 6),
        }
    finally:
        if progress is not None:
            progress.close()
        if env is not None:
            env.close()


def _reset_train_logs_if_schema_changed(train_log_path, train_jsonl_path) -> None:
    if train_log_path.exists() and train_log_path.stat().st_size > 0:
        first_line = train_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        current_header = [item.strip() for item in first_line.split(",")]
        if current_header != TRAIN_LOG_COLUMNS:
            train_log_path.unlink(missing_ok=True)
            train_jsonl_path.unlink(missing_ok=True)


def _print_smoke_progress(config, text: str, final: bool = False) -> None:
    style = str(getattr(config, "progress_bar_style", "single_line") or "single_line").lower()
    if style == "none":
        return
    if style == "single_line":
        print("\r" + text, end="\n" if final else "", flush=True)
    else:
        print(text, flush=True)


def _cuda_device_name(agent: Optional[DQNShared]) -> str:
    try:
        import torch

        if agent is None or str(getattr(agent, "device", "")) != "cuda" or not torch.cuda.is_available():
            return ""
        idx = getattr(agent.device, "index", None)
        return torch.cuda.get_device_name(0 if idx is None else idx)
    except Exception:
        return ""


def _validate_checkpoint_candidate(
    config,
    candidate_path,
    candidate_meta_path,
    best_path,
    best_meta_path,
    episode_number: int,
    current_best_score: Optional[float],
) -> Optional[float]:
    try:
        from .eval import evaluate
    except Exception as exc:
        print(f"Validation skipped: could not import evaluator: {exc}")
        return current_best_score

    original_episode_seconds = int(getattr(config, "episode_seconds", 0))
    original_eval_seeds = list(getattr(config, "eval_seeds", None) or [getattr(config, "seed", 42)])
    original_save_eval_logs = bool(getattr(config, "save_eval_logs", True))
    try:
        config.episode_seconds = int(getattr(config, "validation_seconds", original_episode_seconds))
        config.eval_seeds = list(getattr(config, "validation_seeds", [getattr(config, "seed", 42)]))
        config.save_eval_logs = False
        rl_metrics = evaluate(config, mode="rl", episodes=1)
        fixed_metrics = evaluate(config, mode="fixed_native", episodes=1)
    except Exception as exc:
        print(f"Validation after episode {episode_number}: skipped ({type(exc).__name__}: {exc})")
        return current_best_score
    finally:
        config.episode_seconds = original_episode_seconds
        config.eval_seeds = original_eval_seeds
        config.save_eval_logs = original_save_eval_logs

    rl_score = weighted_mobility_score(rl_metrics, baseline=fixed_metrics)
    fixed_score = weighted_mobility_score(fixed_metrics, baseline=fixed_metrics)
    improvement = ((fixed_score - rl_score) / max(abs(fixed_score), 1e-6)) * 100.0
    phase_sets = int(rl_metrics.get("action_stats", {}).get("phase_set_count", 0))
    updated = False
    if phase_sets > 0 and (current_best_score is None or rl_score < current_best_score):
        shutil.copy2(candidate_path, best_path)
        shutil.copy2(candidate_meta_path, best_meta_path)
        current_best_score = rl_score
        updated = True
    print()
    print(f"Validation after episode {episode_number}:")
    print(f"  rl_objective_score: {rl_score:.6f}")
    print(f"  fixed_objective_score: {fixed_score:.6f}")
    print(f"  improvement_pct: {improvement:.3f}")
    print(f"  best_checkpoint_updated: {'yes' if updated else 'no'}")
    return current_best_score
