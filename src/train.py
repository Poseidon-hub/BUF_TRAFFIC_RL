from typing import Optional

import numpy as np

from .checkpointing import build_checkpoint_metadata, checkpoint_file_exists, save_checkpoint_metadata
from .config import ACTION_SIZE
from .dqn import DQNShared
from .logging_utils import append_jsonl, save_json, write_csv_row
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
    "episode_avg_queue",
    "episode_avg_waiting_time",
    "total_reward",
    "loss_avg",
    "td_error_avg",
    "q_value_avg",
    "q_value_max",
    "q_value_min",
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
]


def train(config) -> dict:
    set_seed(config.seed)
    safe_mkdir(config.checkpoints_dir)
    safe_mkdir(config.logs_dir)

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
    recent_target_q = []
    recent_vehicle_components = []
    recent_pedestrian_components = []
    recent_pedestrian_shares = []
    queue_values = []
    wait_values = []

    print("Training started...")
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

        while total_steps < config.train_steps:
            epsilon = agent.epsilon(total_steps)
            actions = {tls_id: agent.act(tls_obs, epsilon=epsilon) for tls_id, tls_obs in obs.items()}
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
            if total_steps >= config.start_learning_after and total_steps % config.train_freq == 0:
                update_metrics = agent.update()
                if update_metrics is not None:
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
                    "episode_avg_queue": float(np.mean(queue_values)) if queue_values else 0.0,
                    "episode_avg_waiting_time": float(np.mean(wait_values)) if wait_values else 0.0,
                    "total_reward": episode_reward,
                    "loss_avg": float(np.mean(recent_losses)) if recent_losses else "",
                    "td_error_avg": float(np.mean(recent_td_errors)) if recent_td_errors else "",
                    "q_value_avg": float(np.mean(recent_q_avgs)) if recent_q_avgs else "",
                    "q_value_max": float(np.max(recent_q_max)) if recent_q_max else "",
                    "q_value_min": float(np.min(recent_q_min)) if recent_q_min else "",
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
                }
                if getattr(config, "save_train_logs", True):
                    write_csv_row(train_log_path, row)
                    append_jsonl(train_jsonl_path, row)
                print(
                    "train "
                    f"episode={episode_idx} step={total_steps} epsilon={epsilon:.3f} "
                    f"reward={episode_reward:.3f} loss={row['loss_avg']} "
                    f"q_avg={row['q_value_avg']} target_updates={row['target_updates_count']} "
                    f"ped_reward_share={row['pedestrian_reward_share']}"
                )
                recent_rewards.clear()
                recent_losses.clear()
                recent_td_errors.clear()
                recent_q_avgs.clear()
                recent_q_max.clear()
                recent_q_min.clear()
                recent_target_q.clear()
                recent_vehicle_components.clear()
                recent_pedestrian_components.clear()
                recent_pedestrian_shares.clear()

            obs = next_obs
            total_steps += 1
            if done and total_steps < config.train_steps:
                obs = env.reset()
                episode_idx += 1
                episode_steps = 0
                episode_reward = 0.0
                queue_values.clear()
                wait_values.clear()

        agent.update_target(global_step=total_steps)
        if total_steps <= 0:
            raise RuntimeError("Обучение не выполнило ни одного шага.")
        if len(agent.replay) == 0:
            raise RuntimeError("Replay buffer пустой после обучения; checkpoint не сохраняется.")
        agent.save(config.checkpoint_path)
        if not checkpoint_file_exists(config.checkpoint_path):
            raise RuntimeError(f"Checkpoint не был создан: {config.checkpoint_path}")
        metadata = build_checkpoint_metadata(
            config=config,
            scenario=env.scenario,
            tls_ids=env.tls_ids,
            obs_dim=obs_dim,
            num_phases_per_tls=env.num_phases_per_tls,
        )
        save_checkpoint_metadata(config.checkpoint_meta_path, metadata)
        save_json(config.checkpoints_dir / "config.json", config.to_json_dict())
        print("Training finished.")
        return {
            "checkpoint_path": str(config.checkpoint_path),
            "meta_path": str(config.checkpoint_meta_path),
            "episodes_done": episode_idx + 1,
            "total_steps": total_steps,
            "final_epsilon": agent.epsilon(total_steps),
            "train_metrics_path": str(train_log_path),
        }
    finally:
        if env is not None:
            env.close()


def _reset_train_logs_if_schema_changed(train_log_path, train_jsonl_path) -> None:
    if train_log_path.exists() and train_log_path.stat().st_size > 0:
        first_line = train_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        current_header = [item.strip() for item in first_line.split(",")]
        if current_header != TRAIN_LOG_COLUMNS:
            train_log_path.unlink(missing_ok=True)
            train_jsonl_path.unlink(missing_ok=True)
