import sys
from dataclasses import dataclass
from pathlib import Path


def _print_dependency_help(missing: str) -> None:
    print("Не хватает Python-зависимостей.")
    print(f"Проблема: {missing}")
    print("Установите зависимости для выбранного интерпретатора Python:")
    print("  python -m pip install -r requirements.txt")
    print("В PyCharm проверьте, что Shift+F10 использует тот же интерпретатор, куда установлены numpy и torch.")


@dataclass
class CheckpointStatus:
    status: str
    reason: str


def get_checkpoint_status(cfg, current_metadata: dict) -> CheckpointStatus:
    from src.checkpointing import (
        checkpoint_file_exists,
        load_checkpoint_header,
        load_checkpoint_metadata,
        validate_checkpoint_compatibility,
    )

    if not checkpoint_file_exists(cfg.checkpoint_path):
        return CheckpointStatus("missing", "checkpoint file is missing or empty")

    _, header_error = load_checkpoint_header(cfg.checkpoint_path)
    if header_error:
        return CheckpointStatus("corrupt", header_error)

    _, meta_error = load_checkpoint_metadata(cfg.checkpoint_meta_path)
    if meta_error:
        return CheckpointStatus("incompatible", meta_error)

    compatible, reason = validate_checkpoint_compatibility(cfg, current_metadata)
    if compatible:
        return CheckpointStatus("valid", reason)
    return CheckpointStatus("incompatible", reason)


def _probe_scenario_obs_dim(cfg) -> tuple:
    from src.sumo_env import SumoMultiAgentEnv

    env = None
    try:
        env = SumoMultiAgentEnv(
            scenario_dir=cfg.scenario_dir,
            use_gui=False,
            step_length=cfg.step_length,
            episode_seconds=cfg.episode_seconds,
            min_green=cfg.min_green,
            alpha=cfg.alpha,
            beta=cfg.beta,
            obs_cfg=cfg,
            seed=cfg.seed,
            sumo_extra_args=getattr(cfg, "sumo_extra_args", None),
        )
        obs = env.reset()
        return len(next(iter(obs.values()))), env.max_phases_global
    finally:
        if env is not None:
            env.close()


def _probe_checkpoint_metadata(cfg) -> dict:
    from src.checkpointing import build_checkpoint_metadata
    from src.sumo_env import SumoMultiAgentEnv

    env = None
    try:
        env = SumoMultiAgentEnv(
            scenario_dir=cfg.scenario_dir,
            use_gui=False,
            step_length=cfg.step_length,
            episode_seconds=cfg.episode_seconds,
            min_green=cfg.min_green,
            alpha=cfg.alpha,
            beta=cfg.beta,
            obs_cfg=cfg,
            seed=cfg.seed,
            sumo_extra_args=getattr(cfg, "sumo_extra_args", None),
        )
        obs = env.reset()
        obs_dim = len(next(iter(obs.values())))
        cfg.set_observation_phases(env.max_phases_global)
        return build_checkpoint_metadata(
            config=cfg,
            scenario=env.scenario,
            tls_ids=env.tls_ids,
            obs_dim=obs_dim,
            num_phases_per_tls=env.num_phases_per_tls,
        )
    finally:
        if env is not None:
            env.close()


def _checkpoint_obs_dim(checkpoint_path: Path) -> int:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return int(checkpoint.get("obs_dim", -1))


def _checkpoint_action_dim(checkpoint_path: Path) -> int:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return int(checkpoint.get("action_dim", -1))


def _improvement_pct(fixed_value, rl_value):
    try:
        fixed = float(fixed_value)
        rl = float(rl_value)
    except (TypeError, ValueError):
        return None
    if abs(fixed) < 1e-9:
        return None
    return (fixed - rl) / fixed * 100.0


def _print_action_diagnostics(metrics: dict) -> None:
    action_stats = metrics.get("action_stats", {})
    diagnostics = {
        "decision_count": action_stats.get("decision_count", 0),
        "hold_count": action_stats.get("hold_count", 0),
        "switch_count": action_stats.get("switch_count", 0),
        "blocked_by_min_green_count": action_stats.get("blocked_by_min_green_count", 0),
        "phase_set_count": action_stats.get("phase_set_count", 0),
    }
    from src.logging_utils import pretty_print_metrics

    pretty_print_metrics("RL action diagnostics", diagnostics)
    if int(action_stats.get("phase_set_count", 0)) == 0:
        print(
            "WARNING: RL evaluation did not set any traffic light phase. "
            "Check action logic/min_green/action space."
        )


def _print_q_diagnostics(metrics: dict) -> None:
    q_stats = metrics.get("q_value_stats", {})
    from src.logging_utils import pretty_print_metrics

    pretty_print_metrics(
        "RL Q diagnostics",
        {
            "mean_q_hold": q_stats.get("mean_q_hold"),
            "mean_q_switch": q_stats.get("mean_q_switch"),
            "greedy_hold_ratio": q_stats.get("greedy_hold_ratio"),
            "greedy_switch_ratio": q_stats.get("greedy_switch_ratio"),
        },
    )


def check_rl_fixed_difference(rl_metrics: dict, fixed_metrics: dict, cfg) -> tuple[bool, str]:
    action_stats = rl_metrics.get("action_stats", {})
    if int(action_stats.get("phase_set_count", 0)) < int(getattr(cfg, "min_rl_phase_set_count", 1)):
        return False, (
            "RL evaluation did not control traffic lights: "
            f"phase_set_count={action_stats.get('phase_set_count', 0)}."
        )

    metric_keys = [
        "avg_queue",
        "avg_waiting_time",
        "total_waiting_time",
        "throughput",
        "total_reward",
        "arrived",
    ]
    tolerance = float(getattr(cfg, "metric_diff_tolerance", 1e-6))
    all_identical = all(
        abs(float(rl_metrics.get(key, 0.0)) - float(fixed_metrics.get(key, 0.0))) <= tolerance
        for key in metric_keys
    )
    if all_identical:
        return False, "RL and fixed metrics are identical. This usually means RL did not affect TLS control."
    return True, ""


def check_pedestrian_metrics(rl_metrics: dict, fixed_metrics: dict) -> tuple[bool, str]:
    rl_ped = rl_metrics.get("pedestrian_metrics", {})
    fixed_ped = fixed_metrics.get("pedestrian_metrics", {})
    if float(rl_ped.get("departed", 0) or 0) <= 0 and float(fixed_ped.get("departed", 0) or 0) <= 0:
        return False, "Pedestrian metrics are zero: no pedestrians departed in RL or fixed evaluation."
    return True, ""


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(rl_value, baseline_value):
    rl = _to_float_or_none(rl_value)
    baseline = _to_float_or_none(baseline_value)
    if rl is None or baseline is None:
        return None
    return rl - baseline


def _max_improvement_pct(baseline_value, rl_value):
    baseline = _to_float_or_none(baseline_value)
    rl = _to_float_or_none(rl_value)
    if baseline is None or rl is None or abs(baseline) < 1e-9:
        return None
    return (rl - baseline) / baseline * 100.0


def _min_improvement_pct(baseline_value, rl_value):
    baseline = _to_float_or_none(baseline_value)
    rl = _to_float_or_none(rl_value)
    if baseline is None or rl is None or abs(baseline) < 1e-9:
        return None
    return (baseline - rl) / baseline * 100.0


def _console_value(value):
    return "n/a" if value is None else value


def compare_pedestrian_metrics(rl_ped_metrics: dict, baseline_ped_metrics: dict, baseline_name: str) -> dict:
    rl = {
        "departed": _to_float_or_none((rl_ped_metrics or {}).get("departed")),
        "arrived": _to_float_or_none((rl_ped_metrics or {}).get("arrived")),
        "running": _to_float_or_none((rl_ped_metrics or {}).get("running")),
        "waiting_count": _to_float_or_none((rl_ped_metrics or {}).get("waiting_count")),
        "total_waiting_time": _to_float_or_none((rl_ped_metrics or {}).get("total_waiting_time")),
        "avg_waiting_time": _to_float_or_none((rl_ped_metrics or {}).get("avg_waiting_time")),
    }
    baseline = {
        "name": baseline_name,
        "departed": _to_float_or_none((baseline_ped_metrics or {}).get("departed")),
        "arrived": _to_float_or_none((baseline_ped_metrics or {}).get("arrived")),
        "running": _to_float_or_none((baseline_ped_metrics or {}).get("running")),
        "waiting_count": _to_float_or_none((baseline_ped_metrics or {}).get("waiting_count")),
        "total_waiting_time": _to_float_or_none((baseline_ped_metrics or {}).get("total_waiting_time")),
        "avg_waiting_time": _to_float_or_none((baseline_ped_metrics or {}).get("avg_waiting_time")),
    }
    delta = {
        "arrived_delta": _delta(rl["arrived"], baseline["arrived"]),
        "running_delta": _delta(rl["running"], baseline["running"]),
        "departed_delta": _delta(rl["departed"], baseline["departed"]),
        "avg_waiting_time_delta": _delta(rl["avg_waiting_time"], baseline["avg_waiting_time"]),
        "total_waiting_time_delta": _delta(rl["total_waiting_time"], baseline["total_waiting_time"]),
        "waiting_count_delta": _delta(rl["waiting_count"], baseline["waiting_count"]),
    }
    improvement = {
        "arrived": _max_improvement_pct(baseline["arrived"], rl["arrived"]),
        "avg_waiting_time": _min_improvement_pct(baseline["avg_waiting_time"], rl["avg_waiting_time"]),
        "total_waiting_time": _min_improvement_pct(baseline["total_waiting_time"], rl["total_waiting_time"]),
        "waiting_count": _min_improvement_pct(baseline["waiting_count"], rl["waiting_count"]),
    }
    flat = {
        "baseline_name": baseline_name,
        "pedestrian_rl_departed": rl["departed"],
        "pedestrian_baseline_departed": baseline["departed"],
        "pedestrian_departed_delta": delta["departed_delta"],
        "pedestrian_rl_arrived": rl["arrived"],
        "pedestrian_baseline_arrived": baseline["arrived"],
        "pedestrian_arrived_delta": delta["arrived_delta"],
        "pedestrian_arrived_improvement_pct": improvement["arrived"],
        "pedestrian_rl_running": rl["running"],
        "pedestrian_baseline_running": baseline["running"],
        "pedestrian_running_delta": delta["running_delta"],
        "pedestrian_rl_waiting_count": rl["waiting_count"],
        "pedestrian_baseline_waiting_count": baseline["waiting_count"],
        "pedestrian_waiting_count_delta": delta["waiting_count_delta"],
        "pedestrian_waiting_count_improvement_pct": improvement["waiting_count"],
        "pedestrian_rl_total_waiting_time": rl["total_waiting_time"],
        "pedestrian_baseline_total_waiting_time": baseline["total_waiting_time"],
        "pedestrian_total_waiting_time_delta": delta["total_waiting_time_delta"],
        "pedestrian_total_waiting_time_improvement_pct": improvement["total_waiting_time"],
        "pedestrian_rl_avg_waiting_time": rl["avg_waiting_time"],
        "pedestrian_baseline_avg_waiting_time": baseline["avg_waiting_time"],
        "pedestrian_avg_waiting_time_delta": delta["avg_waiting_time_delta"],
        "pedestrian_avg_waiting_time_improvement_pct": improvement["avg_waiting_time"],
    }
    return {
        "rl": rl,
        "baseline": baseline,
        "delta": delta,
        "improvement_pct": improvement,
        "notes": {
            "running_note": "pedestrian_running is reported as delta only because it is not always directly interpretable as better/worse.",
            "zero_division_note": "If baseline value is zero, improvement_pct is null.",
        },
        "flat": flat,
    }


def build_baseline_comparison(
    rl_metrics: dict,
    baseline_metrics: dict,
    baseline_key: str,
    baseline_name: str = None,
    fairness_check=None,
) -> dict:
    baseline_name = baseline_name or ("real_timing" if baseline_key == "real" else "native_fixed")
    pedestrian_comparison = compare_pedestrian_metrics(
        rl_metrics.get("pedestrian_metrics", {}),
        baseline_metrics.get("pedestrian_metrics", {}),
        baseline_name=baseline_name,
    )
    comparison = {
        "rl_avg_queue": rl_metrics["avg_queue"],
        f"{baseline_key}_avg_queue": baseline_metrics["avg_queue"],
        "rl_avg_waiting_time": rl_metrics["avg_waiting_time"],
        f"{baseline_key}_avg_waiting_time": baseline_metrics["avg_waiting_time"],
        "rl_total_waiting_time": rl_metrics["total_waiting_time"],
        f"{baseline_key}_total_waiting_time": baseline_metrics["total_waiting_time"],
        "rl_total_reward": rl_metrics["total_reward"],
        f"{baseline_key}_total_reward": baseline_metrics["total_reward"],
        "rl_throughput": rl_metrics["throughput"],
        f"{baseline_key}_throughput": baseline_metrics["throughput"],
        "rl_departed": rl_metrics.get("departed", 0),
        f"{baseline_key}_departed": baseline_metrics.get("departed", 0),
        "rl_arrived": rl_metrics.get("arrived", 0),
        f"{baseline_key}_arrived": baseline_metrics.get("arrived", 0),
        "rl_episode_steps": rl_metrics.get("episode_steps", 0),
        f"{baseline_key}_episode_steps": baseline_metrics.get("episode_steps", 0),
        "queue_improvement_pct": _improvement_pct(baseline_metrics["avg_queue"], rl_metrics["avg_queue"]),
        "waiting_time_improvement_pct": _improvement_pct(
            baseline_metrics["avg_waiting_time"], rl_metrics["avg_waiting_time"]
        ),
        "throughput_delta": rl_metrics["throughput"] - baseline_metrics["throughput"],
        "reward_delta": rl_metrics["total_reward"] - baseline_metrics["total_reward"],
        "pedestrian_metrics": {
            "rl": rl_metrics.get("pedestrian_metrics", {}),
            baseline_key: baseline_metrics.get("pedestrian_metrics", {}),
        },
        "pedestrian_comparison": {
            key: value
            for key, value in pedestrian_comparison.items()
            if key != "flat"
        },
        "rl_action_stats": rl_metrics.get("action_stats", {}),
        f"{baseline_key}_baseline_stats": baseline_metrics.get("baseline_stats", {}),
    }
    comparison.update(pedestrian_comparison["flat"])
    if fairness_check is not None:
        comparison["fairness_check"] = fairness_check
    return comparison


def build_fairness_check(rl_metrics: dict, baseline_metrics: dict) -> dict:
    return {
        "same_net_file": rl_metrics.get("net_file") == baseline_metrics.get("net_file"),
        "same_route_files": rl_metrics.get("route_files") == baseline_metrics.get("route_files"),
        "same_seed": rl_metrics.get("seed") == baseline_metrics.get("seed"),
        "same_episode_seconds": rl_metrics.get("episode_seconds") == baseline_metrics.get("episode_seconds"),
        "same_step_length": rl_metrics.get("step_length") == baseline_metrics.get("step_length"),
        "rl_sumocfg": rl_metrics.get("sumocfg_path"),
        "real_timing_sumocfg": baseline_metrics.get("sumocfg_path"),
    }


def print_baseline_comparison(
    pretty_print_metrics,
    title: str,
    improvement_title: str,
    comparison: dict,
    baseline_key: str,
) -> None:
    display = {
        "rl_avg_queue": comparison.get("rl_avg_queue"),
        f"{baseline_key}_avg_queue": comparison.get(f"{baseline_key}_avg_queue"),
        "rl_avg_waiting_time": comparison.get("rl_avg_waiting_time"),
        f"{baseline_key}_avg_waiting_time": comparison.get(f"{baseline_key}_avg_waiting_time"),
        "rl_total_waiting_time": comparison.get("rl_total_waiting_time"),
        f"{baseline_key}_total_waiting_time": comparison.get(f"{baseline_key}_total_waiting_time"),
        "rl_total_reward": comparison.get("rl_total_reward"),
        f"{baseline_key}_total_reward": comparison.get(f"{baseline_key}_total_reward"),
        "rl_throughput": comparison.get("rl_throughput"),
        f"{baseline_key}_throughput": comparison.get(f"{baseline_key}_throughput"),
        "rl_departed": comparison.get("rl_departed"),
        f"{baseline_key}_departed": comparison.get(f"{baseline_key}_departed"),
        "rl_arrived": comparison.get("rl_arrived"),
        f"{baseline_key}_arrived": comparison.get(f"{baseline_key}_arrived"),
        "rl_episode_steps": comparison.get("rl_episode_steps"),
        f"{baseline_key}_episode_steps": comparison.get(f"{baseline_key}_episode_steps"),
    }
    improvements = {
        "queue_improvement_pct": comparison.get("queue_improvement_pct"),
        "waiting_time_improvement_pct": comparison.get("waiting_time_improvement_pct"),
        "throughput_delta": comparison.get("throughput_delta"),
        "reward_delta": comparison.get("reward_delta"),
    }
    pretty_print_metrics(title, display)
    pretty_print_metrics(improvement_title, improvements)


def print_pedestrian_comparison(
    pretty_print_metrics,
    title: str,
    improvement_title: str,
    comparison: dict,
    baseline_console_key: str,
) -> None:
    pedestrian = comparison.get("pedestrian_comparison", {})
    rl = pedestrian.get("rl", {})
    baseline = pedestrian.get("baseline", {})
    delta = pedestrian.get("delta", {})
    improvement = pedestrian.get("improvement_pct", {})
    values = {
        "rl_pedestrian_departed": _console_value(rl.get("departed")),
        f"{baseline_console_key}_pedestrian_departed": _console_value(baseline.get("departed")),
        "rl_pedestrian_arrived": _console_value(rl.get("arrived")),
        f"{baseline_console_key}_pedestrian_arrived": _console_value(baseline.get("arrived")),
        "rl_pedestrian_running": _console_value(rl.get("running")),
        f"{baseline_console_key}_pedestrian_running": _console_value(baseline.get("running")),
        "rl_pedestrian_waiting_count": _console_value(rl.get("waiting_count")),
        f"{baseline_console_key}_pedestrian_waiting_count": _console_value(baseline.get("waiting_count")),
        "rl_pedestrian_total_waiting_time": _console_value(rl.get("total_waiting_time")),
        f"{baseline_console_key}_pedestrian_total_waiting_time": _console_value(
            baseline.get("total_waiting_time")
        ),
        "rl_pedestrian_avg_waiting_time": _console_value(rl.get("avg_waiting_time")),
        f"{baseline_console_key}_pedestrian_avg_waiting_time": _console_value(
            baseline.get("avg_waiting_time")
        ),
    }
    improvements = {
        "pedestrian_arrived_delta": _console_value(delta.get("arrived_delta")),
        "pedestrian_arrived_improvement_pct": _console_value(improvement.get("arrived")),
        "pedestrian_avg_waiting_time_improvement_pct": _console_value(
            improvement.get("avg_waiting_time")
        ),
        "pedestrian_total_waiting_time_improvement_pct": _console_value(
            improvement.get("total_waiting_time")
        ),
        "pedestrian_waiting_count_improvement_pct": _console_value(
            improvement.get("waiting_count")
        ),
        "pedestrian_running_delta": _console_value(delta.get("running_delta")),
    }
    pretty_print_metrics(title, values)
    pretty_print_metrics(improvement_title, improvements)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils import (
        check_python_dependencies,
        check_sumo_installation,
        safe_mkdir,
        sumo_install_instructions,
    )

    sumo_ok, sumo_details = check_sumo_installation()
    if not sumo_ok:
        print(sumo_install_instructions(sumo_details))
        return 0

    deps_ok, deps_details = check_python_dependencies()
    if not deps_ok:
        _print_dependency_help(deps_details)
        return 0

    try:
        from src.checkpointing import checkpoint_file_exists
        from src.config import make_config
        from src.demo_scenario import create_demo_scenario, validate_demo
        from src.eval import evaluate
        from src.logging_utils import pretty_print_metrics, save_json, write_csv_row
        from src.scenario import describe_scenario, discover_scenario, validate_scenario_start
        from src.timing_profiles import load_timing_profile
        from src.train import train

        cfg = make_config(project_root)
        print(f"RL algorithm: {getattr(cfg, 'algorithm', 'dqn')}")
        print(f"Double DQN: {'enabled' if getattr(cfg, 'use_double_dqn', False) else 'disabled'}")
        print(f"Dueling DQN: {'enabled' if getattr(cfg, 'use_dueling_dqn', False) else 'disabled'}")
        print(
            f"Pedestrian reward: "
            f"{'enabled' if getattr(cfg, 'reward_use_pedestrians', False) else 'disabled'}"
        )
        safe_mkdir(cfg.scenario_dir)
        safe_mkdir(cfg.checkpoints_dir)
        safe_mkdir(cfg.logs_dir)

        scenario = discover_scenario(
            cfg.scenario_dir,
            begin=0.0,
            end=float(cfg.episode_seconds),
            step_length=cfg.step_length,
            auto_generate_sumocfg=bool(getattr(cfg, "auto_generate_sumocfg", True)),
            scenario_sumocfg_file=getattr(cfg, "scenario_sumocfg_file", None),
            scenario_net_file=getattr(cfg, "scenario_net_file", None),
            scenario_route_file=getattr(cfg, "scenario_route_file", None),
            auto_generate_pedestrians_if_missing=bool(
                getattr(cfg, "auto_generate_pedestrians_if_missing", False)
            ),
            pedestrian_count=int(getattr(cfg, "pedestrian_demand_count", 30)),
            pedestrian_begin=int(getattr(cfg, "pedestrian_demand_begin", 0)),
            pedestrian_end=int(getattr(cfg, "pedestrian_demand_end", cfg.episode_seconds)),
            pedestrian_prefix=str(getattr(cfg, "pedestrian_demand_prefix", "auto_ped")),
            real_timing_file_name=str(getattr(cfg, "real_timing_file", "tls.add.xml")),
            rl_use_real_timing_program_as_base=bool(
                getattr(cfg, "rl_use_real_timing_program_as_base", True)
            ),
        )
        if scenario.found:
            ok, message, tls_ids = validate_scenario_start(
                scenario,
                use_gui=False,
                seed=cfg.seed,
                step_length=cfg.step_length,
                extra_args=getattr(cfg, "sumo_extra_args", None),
                steps=10,
            )
            if getattr(cfg, "print_scenario_info", True):
                print("Найден внешний SUMO-сценарий:")
                for line in describe_scenario(scenario, tls_ids=tls_ids):
                    print(f"  {line}")
                print("Pedestrian demand:")
                print(f"  existing pedestrians found: {'yes' if scenario.pedestrian_existing_found else 'no'}")
                print(f"  autogenerated pedestrians: {'yes' if scenario.pedestrian_autogenerated else 'no'}")
                print(f"  pedestrian route file: {scenario.pedestrian_route_file or '-'}")
                print(f"  pedestrian count: {scenario.pedestrian_count}")
                timing_path = getattr(scenario, "real_timing_file", None)
                print("Timing profile:")
                if timing_path:
                    try:
                        profile = load_timing_profile(timing_path)
                        with_real = sorted(set(profile.programs.keys()).intersection(tls_ids))
                        without_real = [tls_id for tls_id in tls_ids if tls_id not in profile.programs]
                        print(f"  real timing file: {timing_path}")
                        print(f"  tlLogic programs: {len(profile.programs)}")
                        print(f"  TLS with real timing: {', '.join(with_real) if with_real else '-'}")
                        print(f"  TLS without real timing: {', '.join(without_real) if without_real else '-'}")
                    except Exception as exc:
                        print(f"  real timing file: {timing_path}")
                        print(f"  timing profile parse error: {exc}")
                else:
                    print(
                        f"  Файл реальных таймингов {getattr(cfg, 'real_timing_file', 'tls.add.xml')} не найден. "
                        "Сравнение RL vs real_timing пропущено."
                    )
            if not ok:
                print(f"Короткая проверка SUMO не прошла: {message}")
            elif not tls_ids:
                print("В сценарии не найдено traffic light controllers. RL evaluation невозможен.")
                return 1
        else:
            print("В scenario/ не найден полноценный сценарий net.xml + rou.xml. Создаю demo-сценарий.")
            create_demo_scenario(cfg.scenario_dir)
            scenario = discover_scenario(
                cfg.scenario_dir,
                begin=0.0,
                end=float(cfg.episode_seconds),
                step_length=cfg.step_length,
                auto_generate_sumocfg=bool(getattr(cfg, "auto_generate_sumocfg", True)),
                scenario_sumocfg_file=getattr(cfg, "scenario_sumocfg_file", None),
                scenario_net_file=getattr(cfg, "scenario_net_file", None),
                scenario_route_file=getattr(cfg, "scenario_route_file", None),
                auto_generate_pedestrians_if_missing=bool(
                    getattr(cfg, "auto_generate_pedestrians_if_missing", False)
                ),
                pedestrian_count=int(getattr(cfg, "pedestrian_demand_count", 30)),
                pedestrian_begin=int(getattr(cfg, "pedestrian_demand_begin", 0)),
                pedestrian_end=int(getattr(cfg, "pedestrian_demand_end", cfg.episode_seconds)),
                pedestrian_prefix=str(getattr(cfg, "pedestrian_demand_prefix", "auto_ped")),
                real_timing_file_name=str(getattr(cfg, "real_timing_file", "tls.add.xml")),
                rl_use_real_timing_program_as_base=bool(
                    getattr(cfg, "rl_use_real_timing_program_as_base", True)
                ),
            )

        if not scenario.found and not validate_demo(cfg.scenario_dir):
            print("Сценарий найден, но быстрая проверка SUMO не прошла. Продолжаю: ошибка проявится с подробным сообщением.")

        current_metadata = _probe_checkpoint_metadata(cfg)
        need_train = False
        if getattr(cfg, "force_retrain", False):
            print("FORCE_RETRAIN=True. Запускаю обучение заново.")
            need_train = True
        else:
            checkpoint_status = get_checkpoint_status(cfg, current_metadata)

        if not getattr(cfg, "force_retrain", False) and checkpoint_status.status == "missing":
            print("Checkpoint отсутствует. Запускаю обучение заново.")
            need_train = True
        elif not getattr(cfg, "force_retrain", False) and checkpoint_status.status == "corrupt":
            print(f"Checkpoint повреждён. {checkpoint_status.reason}")
            print("Запускаю обучение заново.")
            need_train = True
        elif not getattr(cfg, "force_retrain", False) and checkpoint_status.status == "incompatible":
            print(
                "Checkpoint несовместим с текущим алгоритмом или сценарием. "
                "Запускаю обучение заново."
            )
            print(f"Причина: {checkpoint_status.reason}")
            algorithm_related = any(
                token in str(checkpoint_status.reason)
                for token in (
                    "algorithm",
                    "checkpoint_version",
                    "use_double_dqn",
                    "use_dueling_dqn",
                    "model_",
                )
            )
            if getattr(cfg, "auto_retrain_on_shape_mismatch", True) or (
                algorithm_related and getattr(cfg, "force_retrain_on_algorithm_change", True)
            ):
                need_train = True
            else:
                print("Удалите checkpoints/dqn.pt или включите AUTO_RETRAIN_ON_SCENARIO_CHANGE.")
                return 1
        elif not getattr(cfg, "force_retrain", False) and checkpoint_status.status == "valid":
            print("Checkpoint найден и совместим. Обучение пропущено.")

        if need_train:
            train(cfg)
            if not checkpoint_file_exists(cfg.checkpoint_path):
                print(f"Checkpoint не был создан: {cfg.checkpoint_path}")
                return 1
            if not cfg.checkpoint_meta_path.exists() or cfg.checkpoint_meta_path.stat().st_size == 0:
                print(f"Checkpoint metadata не был создан: {cfg.checkpoint_meta_path}")
                return 1
            current_metadata = _probe_checkpoint_metadata(cfg)
            new_status = get_checkpoint_status(cfg, current_metadata)
            if new_status.status != "valid":
                print(f"Новый checkpoint несовместим сразу после обучения: {new_status.reason}")
                return 1
            print(f"Checkpoint сохранён: {cfg.checkpoint_path}")
            if getattr(cfg, "quality_gate_enabled", True):
                for round_idx in range(int(getattr(cfg, "quality_gate_max_retrain_rounds", 0)) + 1):
                    gate_metrics = evaluate(cfg, mode="rl", episodes=1)
                    phase_sets = int(gate_metrics.get("action_stats", {}).get("phase_set_count", 0))
                    if phase_sets >= int(getattr(cfg, "min_rl_phase_set_count", 1)):
                        break
                    print("RL policy is degenerate: no phase switches in evaluation.")
                    if round_idx >= int(getattr(cfg, "quality_gate_max_retrain_rounds", 0)):
                        print("WARNING: RL policy still performs no switches after training quality gate.")
                        break
                    cfg.train_steps = int(getattr(cfg, "quality_gate_extra_episodes", 1)) * int(cfg.episode_seconds)
                    train(cfg)
                    print(f"Checkpoint сохранён: {cfg.checkpoint_path}")

        print("Запускаю evaluation: RL-агенты.")
        rl_metrics = evaluate(cfg, mode="rl", episodes=cfg.eval_episodes)
        if getattr(cfg, "print_action_diagnostics", True):
            _print_action_diagnostics(rl_metrics)
            _print_q_diagnostics(rl_metrics)

        real_metrics = None
        if (
            getattr(cfg, "run_real_timing_baseline", True)
            and getattr(cfg, "use_real_timing_baseline", True)
            and getattr(scenario, "real_timing_file", None)
        ):
            print("Запускаю evaluation: real timing baseline.")
            real_metrics = evaluate(cfg, mode="real_timing", episodes=cfg.eval_episodes)
        elif getattr(cfg, "run_real_timing_baseline", True):
            print(
                f"Файл реальных таймингов {getattr(cfg, 'real_timing_file', 'tls.add.xml')} не найден. "
                "Сравнение RL vs real_timing пропущено."
            )

        fixed_metrics = None
        if getattr(cfg, "run_native_fixed_baseline", True) or real_metrics is None:
            print("Запускаю evaluation: native fixed-time baseline.")
            fixed_metrics = evaluate(cfg, mode="fixed_native", episodes=cfg.eval_episodes)

        primary_metrics = real_metrics if real_metrics is not None else fixed_metrics

        if getattr(cfg, "strict_dev_validation", False):
            if getattr(cfg, "require_rl_phase_switches", False) or getattr(
                cfg, "require_rl_fixed_metric_difference", False
            ):
                ok, message = check_rl_fixed_difference(rl_metrics, primary_metrics, cfg)
                if not ok:
                    print(message)
                    return 1
            if getattr(cfg, "require_nonzero_pedestrian_metrics", False):
                ok, message = check_pedestrian_metrics(rl_metrics, primary_metrics)
                if not ok:
                    print(message)
                    return 1

        primary_comparison = None
        if real_metrics is not None:
            fairness = build_fairness_check(rl_metrics, real_metrics)
            if getattr(cfg, "strict_real_timing_validation", True) and not all(
                value for key, value in fairness.items() if key.startswith("same_")
            ):
                print("WARNING: RL and real_timing fairness check failed.")
            real_comparison = build_baseline_comparison(
                rl_metrics,
                real_metrics,
                "real",
                baseline_name="real_timing",
                fairness_check=fairness,
            )
            print_baseline_comparison(
                pretty_print_metrics,
                "Сравнение RL vs real timing",
                "Улучшения относительно real timing",
                real_comparison,
                "real",
            )
            print_pedestrian_comparison(
                pretty_print_metrics,
                "Пешеходные метрики: RL vs real timing",
                "Улучшения пешеходных метрик относительно real timing",
                real_comparison,
                "real",
            )
            save_json(cfg.logs_dir / "comparison_rl_vs_real_timing.json", real_comparison)
            (cfg.logs_dir / "comparison_rl_vs_real_timing.csv").unlink(missing_ok=True)
            write_csv_row(cfg.logs_dir / "comparison_rl_vs_real_timing.csv", real_comparison)
            primary_comparison = real_comparison

        if fixed_metrics is not None:
            fixed_comparison = build_baseline_comparison(
                rl_metrics,
                fixed_metrics,
                "fixed",
                baseline_name="native_fixed",
            )
            print_baseline_comparison(
                pretty_print_metrics,
                "Сравнение RL vs native fixed-time",
                "Улучшения относительно native fixed-time",
                fixed_comparison,
                "fixed",
            )
            print_pedestrian_comparison(
                pretty_print_metrics,
                "Пешеходные метрики: RL vs native fixed-time",
                "Улучшения пешеходных метрик относительно native fixed-time",
                fixed_comparison,
                "fixed",
            )
            save_json(cfg.logs_dir / "comparison_rl_vs_fixed_native.json", fixed_comparison)
            save_json(cfg.logs_dir / "comparison_rl_vs_native_fixed.json", fixed_comparison)
            (cfg.logs_dir / "comparison_rl_vs_fixed_native.csv").unlink(missing_ok=True)
            (cfg.logs_dir / "comparison_rl_vs_native_fixed.csv").unlink(missing_ok=True)
            write_csv_row(cfg.logs_dir / "comparison_rl_vs_fixed_native.csv", fixed_comparison)
            write_csv_row(cfg.logs_dir / "comparison_rl_vs_native_fixed.csv", fixed_comparison)
            if primary_comparison is None:
                primary_comparison = fixed_comparison

        comparison_with_pedestrians = dict(primary_comparison or {})
        comparison_with_pedestrians["pedestrian_metrics"] = {
            "rl": rl_metrics.get("pedestrian_metrics", {}),
            "fixed_native": fixed_metrics.get("pedestrian_metrics", {}) if fixed_metrics else {},
            "real_timing": real_metrics.get("pedestrian_metrics", {}) if real_metrics else {},
        }
        comparison_with_pedestrians["rl_action_stats"] = rl_metrics.get("action_stats", {})
        comparison_with_pedestrians["primary_baseline"] = "real_timing" if real_metrics is not None else "fixed_native"
        save_json(cfg.logs_dir / "comparison.json", comparison_with_pedestrians)
        (cfg.logs_dir / "comparison.csv").unlink(missing_ok=True)
        write_csv_row(cfg.logs_dir / "comparison.csv", comparison_with_pedestrians)
        return 0
    except KeyboardInterrupt:
        print("\nЗапуск остановлен пользователем.")
        return 130
    except Exception as exc:
        print("\nПроект не смог завершить запуск.")
        print(f"Причина: {type(exc).__name__}: {exc}")
        print("Проверьте scenario/net.xml и scenario/rou.xml, SUMO_HOME, зависимости из requirements.txt.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
