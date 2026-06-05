import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import make_config
from src.diploma_runner import apply_setup, default_run, prepare_scenario
from src.logging_utils import save_json
from src.sumo_env import SumoMultiAgentEnv
from src.utils import safe_mkdir


def main() -> int:
    cfg = make_config(ROOT)
    setup = default_run("validate_fast")
    apply_setup(cfg, setup)
    cfg.episode_seconds = 20
    cfg.train_episodes = 1
    cfg.train_steps = 20
    safe_mkdir(cfg.logs_dir)

    env = None
    result = {
        "tls_ids": [],
        "pedestrian_signal_links": 0,
        "decision_count": 0,
        "attempted_switch_count": 0,
        "phase_set_count": 0,
        "phase_changed_count": 0,
        "per_tls": {},
    }
    try:
        _, selected_tls, _ = prepare_scenario(cfg, setup)
        result["pedestrian_signal_links"] = sum(
            int(item.get("pedestrian_signal_links", 0) or 0) for item in selected_tls
        )
        env = SumoMultiAgentEnv(
            cfg.scenario_dir,
            use_gui=False,
            step_length=cfg.step_length,
            episode_seconds=cfg.episode_seconds,
            min_green=0,
            alpha=cfg.alpha,
            beta=cfg.beta,
            obs_cfg=cfg,
            seed=cfg.seed,
            sumo_extra_args=cfg.sumo_extra_args,
        )
        env.reset()
        result["tls_ids"] = list(env.tls_ids)
        initial_phases = {tls_id: env._get_phase(tls_id) for tls_id in env.tls_ids}

        info = {}
        for _ in range(6):
            _, _, done, info = env.step({tls_id: 1 for tls_id in env.tls_ids})
            if done:
                break

        stats = info.get("action_stats", env.get_action_stats()) if info else env.get_action_stats()
        result["decision_count"] = int(stats.get("decision_count", 0) or 0)
        result["attempted_switch_count"] = int(stats.get("switch_count", 0) or 0)
        result["phase_set_count"] = int(stats.get("phase_set_count", 0) or 0)
        result["per_tls"] = stats.get("per_tls", {})

        final_phases = {tls_id: env._get_phase(tls_id) for tls_id in env.tls_ids}
        result["phase_changed_count"] = sum(
            1 for tls_id in env.tls_ids if initial_phases.get(tls_id) != final_phases.get(tls_id)
        )

        if result["pedestrian_signal_links"] <= 0:
            raise AssertionError("selected TLS have no pedestrian signal links")
        if result["decision_count"] <= 0:
            raise AssertionError("decision_count == 0")
        if result["attempted_switch_count"] <= 0:
            raise AssertionError("switch_count == 0")
        if result["phase_set_count"] <= 0:
            raise AssertionError("phase_set_count == 0")
        if result["phase_changed_count"] <= 0:
            raise AssertionError("no TLS phase changed")

        save_json(ROOT / "logs" / "action_probe.json", result)
        print("ACTION PROBE PASSED")
        return 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        save_json(ROOT / "logs" / "action_probe.json", result)
        print(result["error"])
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
