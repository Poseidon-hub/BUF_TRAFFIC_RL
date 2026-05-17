import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import make_config
from src.logging_utils import save_json
from src.sumo_env import SumoMultiAgentEnv


def main() -> int:
    cfg = make_config(ROOT)
    cfg.debug_scenario = False
    cfg.episode_seconds = min(int(cfg.episode_seconds), 60)

    env = None
    result = {
        "tls_ids": [],
        "decision_count": 0,
        "attempted_switch_count": 0,
        "phase_set_count": 0,
        "phase_changed_count": 0,
        "per_tls": {},
    }
    try:
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
        for tls_id in env.tls_ids:
            result["per_tls"][tls_id] = {"attempted_switch": 0, "phase_set_count": 0}

        for _ in range(5):
            actions = {tls_id: 1 for tls_id in env.tls_ids}
            _, _, done, info = env.step(actions)
            if done:
                break

        stats = info.get("action_stats", env.get_action_stats())
        result["decision_count"] = int(stats.get("decision_count", 0))
        result["attempted_switch_count"] = int(stats.get("switch_count", 0))
        result["phase_set_count"] = int(stats.get("phase_set_count", 0))
        for tls_id, tls_stats in stats.get("per_tls", {}).items():
            result["per_tls"].setdefault(tls_id, {})
            result["per_tls"][tls_id]["attempted_switch"] = int(tls_stats.get("switch", 0))
            result["per_tls"][tls_id]["phase_set_count"] = int(tls_stats.get("phase_set_count", 0))

        final_phases = {tls_id: env._get_phase(tls_id) for tls_id in env.tls_ids}
        result["phase_changed_count"] = sum(
            1 for tls_id in env.tls_ids if initial_phases.get(tls_id) != final_phases.get(tls_id)
        )
        save_json(ROOT / "logs" / "action_probe.json", result)

        if result["decision_count"] <= 0:
            raise AssertionError("action_probe: decision_count == 0")
        if result["attempted_switch_count"] <= 0:
            raise AssertionError("action_probe: attempted_switch_count == 0")
        if result["phase_set_count"] <= 0:
            raise AssertionError("action_probe: phase_set_count == 0")
        if result["phase_changed_count"] <= 0:
            raise AssertionError("action_probe: no TLS phase changed")
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
