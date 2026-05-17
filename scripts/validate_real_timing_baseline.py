import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import make_config
from src.eval import evaluate
from src.scenario import discover_scenario
from src.sumo_env import SumoMultiAgentEnv
from src.timing_profiles import load_timing_profile


def fast_config():
    cfg = make_config(ROOT)
    cfg.debug_scenario = False
    cfg.episode_seconds = 120
    cfg.train_steps = 120
    cfg.eval_episodes = 1
    cfg.eval_seeds = [42]
    return cfg


def discover(cfg):
    return discover_scenario(
        cfg.scenario_dir,
        begin=0.0,
        end=float(cfg.episode_seconds),
        step_length=cfg.step_length,
        auto_generate_sumocfg=True,
        scenario_sumocfg_file=getattr(cfg, "scenario_sumocfg_file", None),
        scenario_net_file=getattr(cfg, "scenario_net_file", None),
        scenario_route_file=getattr(cfg, "scenario_route_file", None),
        auto_generate_pedestrians_if_missing=bool(cfg.auto_generate_pedestrians_if_missing),
        pedestrian_count=int(cfg.pedestrian_demand_count),
        pedestrian_begin=int(cfg.pedestrian_demand_begin),
        pedestrian_end=int(cfg.pedestrian_demand_end),
        pedestrian_prefix=str(cfg.pedestrian_demand_prefix),
        real_timing_file_name=str(cfg.real_timing_file),
        rl_use_real_timing_program_as_base=bool(cfg.rl_use_real_timing_program_as_base),
    )


def validate_env_real_timing(cfg, scenario, profile):
    env = None
    try:
        env = SumoMultiAgentEnv(
            scenario_dir=cfg.scenario_dir,
            use_gui=False,
            step_length=cfg.step_length,
            episode_seconds=30,
            min_green=cfg.min_green,
            alpha=cfg.alpha,
            beta=cfg.beta,
            obs_cfg=cfg,
            seed=cfg.seed,
            sumo_extra_args=cfg.sumo_extra_args,
            mode="real_timing",
        )
        obs = env.reset()
        assert obs, "real_timing env returned empty observations"
        source = env.timing_source
        assert source.get("used_in_this_mode") is True, "real_timing did not use timing profile"
        assert int(source.get("programs_loaded", 0)) == len(profile.programs), "program count mismatch"
        for tls_id, program in profile.programs.items():
            if tls_id not in env.tls_ids:
                if getattr(cfg, "strict_real_timing_validation", True):
                    raise AssertionError(f"tls.add.xml references unknown TLS: {tls_id}")
                continue
            active = source.get("active_programs", {}).get(tls_id)
            assert active == program.programID, f"{tls_id} active program {active} != {program.programID}"
            phase_check = source.get("phase_count_check", {}).get(tls_id, {})
            assert phase_check.get("matches") is True, f"{tls_id} phase count mismatch: {phase_check}"
            duration_check = source.get("phase_duration_check", {}).get(tls_id, {})
            assert duration_check.get("matches") is True, f"{tls_id} phase durations mismatch: {duration_check}"

        info = {}
        for _ in range(10):
            obs, rewards, done, info = env.step({})
            if done:
                break
        assert int(info.get("phase_set_count", 0)) == 0, "real_timing baseline called setPhase"
    finally:
        if env is not None:
            env.close()


def run_main_for_comparison():
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "BFU_FAST_TEST": "1",
            "BFU_EPISODE_SECONDS": "120",
            "BFU_TRAIN_SECONDS": "120",
            "BFU_EVAL_EPISODES": "1",
            "BFU_EVAL_SEEDS": "42",
        }
    )
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=240,
    )
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "validation_real_timing_main_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (ROOT / "logs" / "validation_real_timing_main_stderr.txt").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, f"main.py failed during real_timing validation: {result.stdout}\n{result.stderr}"


def validate_outputs():
    eval_path = ROOT / "logs" / "eval_real_timing.json"
    comparison_path = ROOT / "logs" / "comparison_rl_vs_real_timing.json"
    assert eval_path.exists(), "logs/eval_real_timing.json was not created"
    assert comparison_path.exists(), "logs/comparison_rl_vs_real_timing.json was not created"
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    stats = data.get("baseline_stats") or {}
    assert int(stats.get("phase_set_count", -1)) == 0, "real_timing baseline phase_set_count != 0"
    assert "timing_source" in data, "eval_real_timing.json missing timing_source"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    fairness = comparison.get("fairness_check") or {}
    for key in ["same_net_file", "same_route_files", "same_seed", "same_episode_seconds", "same_step_length"]:
        assert fairness.get(key) is True, f"fairness_check failed: {key}={fairness.get(key)}"
    pedestrian = comparison.get("pedestrian_comparison")
    assert isinstance(pedestrian, dict), "comparison_rl_vs_real_timing.json missing pedestrian_comparison"
    for key in ["rl", "baseline", "delta", "improvement_pct"]:
        assert key in pedestrian, f"pedestrian_comparison missing {key}"
    assert pedestrian["baseline"].get("name") == "real_timing", "real timing pedestrian baseline name mismatch"

    native_cfg = (ROOT / "scenario" / "autogenerated_native.sumocfg").read_text(encoding="utf-8")
    assert "tls.add.xml" not in native_cfg, "native fixed sumocfg unexpectedly includes tls.add.xml"


def main() -> int:
    try:
        cfg = fast_config()
        scenario = discover(cfg)
        timing_path = scenario.real_timing_file
        assert timing_path and Path(timing_path).exists(), "scenario/tls.add.xml was not found"
        profile = load_timing_profile(timing_path)
        assert profile.programs, "tls.add.xml contains no tlLogic programs"
        validate_env_real_timing(cfg, scenario, profile)
        evaluate(cfg, mode="real_timing", episodes=1)
        run_main_for_comparison()
        validate_outputs()
        print("REAL TIMING BASELINE VALIDATION PASSED")
        return 0
    except Exception as exc:
        print(f"REAL TIMING BASELINE VALIDATION FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
