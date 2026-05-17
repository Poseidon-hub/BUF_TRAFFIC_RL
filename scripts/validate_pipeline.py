import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_META_FIELDS = {
    "scenario_signature",
    "scenario_dir",
    "sumocfg_path",
    "net_file",
    "route_files",
    "tls_ids",
    "obs_dim",
    "action_dim",
    "num_phases_per_tls",
    "training_params",
    "created_at",
}

REQUIRED_TRAIN_COLUMNS = {
    "episode",
    "steps",
    "epsilon",
    "total_reward",
    "avg_queue",
    "avg_waiting_time",
    "loss_avg",
    "replay_size",
    "action_hold_count",
    "action_switch_count",
    "phase_set_count",
}

REQUIRED_ACTION_FIELDS = {
    "decision_count",
    "hold_count",
    "switch_count",
    "blocked_by_min_green_count",
    "phase_set_count",
    "per_tls",
}


def validation_env() -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "BFU_FAST_TEST": "1",
            "BFU_FORCE_RETRAIN": "0",
            "BFU_TRAIN_EPISODES": "1",
            "BFU_EPISODE_SECONDS": "120",
            "BFU_TRAIN_SECONDS": "120",
            "BFU_EVAL_EPISODES": "1",
            "BFU_EVAL_SEEDS": "42",
        }
    )
    return env


def run_main(label: str) -> subprocess.CompletedProcess:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT,
        env=validation_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=240,
    )
    (logs / f"{label}_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs / f"{label}_stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AssertionError(f"main.py failed for {label} with exit code {result.returncode}")
    if "Traceback" in result.stdout or "Traceback" in result.stderr:
        raise AssertionError(f"main.py emitted traceback for {label}")
    return result


def validate_after_checkpoint_delete() -> None:
    checkpoint = ROOT / "checkpoints" / "dqn.pt"
    meta = ROOT / "checkpoints" / "dqn_meta.json"
    checkpoint.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    assert not checkpoint.exists(), "dqn.pt still exists after unlink"
    assert not meta.exists(), "dqn_meta.json still exists after unlink"

    for stale in ["train_metrics.csv", "train_metrics.jsonl"]:
        (ROOT / "logs" / stale).unlink(missing_ok=True)

    before_run = time.time()
    result = run_main("validation_main_after_checkpoint_delete")
    stdout = result.stdout

    required = [
        "Checkpoint отсутствует. Запускаю обучение заново.",
        "Training started",
        "Training finished",
        "Checkpoint сохранён",
    ]
    for text in required:
        assert text in stdout, f"stdout missing required text: {text}"
    assert "Checkpoint найден" not in stdout, "main.py claimed checkpoint was found after it was deleted"

    assert checkpoint.exists(), "dqn.pt was not created"
    assert checkpoint.stat().st_size > 0, "dqn.pt is empty"
    assert checkpoint.stat().st_mtime >= before_run, "dqn.pt mtime is older than validation run"
    assert meta.exists(), "dqn_meta.json was not created"
    assert meta.stat().st_size > 0, "dqn_meta.json is empty"

    metadata = json.loads(meta.read_text(encoding="utf-8"))
    missing = REQUIRED_META_FIELDS.difference(metadata.keys())
    assert not missing, f"dqn_meta.json missing fields: {sorted(missing)}"

    validate_train_metrics(stdout)
    validate_eval_outputs()


def validate_train_metrics(stdout: str) -> None:
    train_csv = ROOT / "logs" / "train_metrics.csv"
    assert train_csv.exists(), "logs/train_metrics.csv does not exist"
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames, "train_metrics.csv has no header"
        missing = REQUIRED_TRAIN_COLUMNS.difference(reader.fieldnames)
        assert not missing, f"train_metrics.csv missing columns: {sorted(missing)}"
        rows = list(reader)
    assert rows, "train_metrics.csv has no data rows"
    assert (
        "train episode" in stdout or "Training episode" in stdout or "episode=" in stdout
    ), "stdout contains no visible training progress line"


def validate_eval_outputs() -> None:
    rl_path = ROOT / "logs" / "eval_rl.json"
    fixed_path = ROOT / "logs" / "eval_fixed.json"
    real_path = ROOT / "logs" / "eval_real_timing.json"
    assert rl_path.exists(), "eval_rl.json does not exist"
    assert fixed_path.exists(), "eval_fixed.json does not exist"
    rl = json.loads(rl_path.read_text(encoding="utf-8"))
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    real = json.loads(real_path.read_text(encoding="utf-8")) if real_path.exists() else None

    action_stats = rl.get("action_stats")
    assert isinstance(action_stats, dict), "eval_rl.json has no action_stats object"
    missing = REQUIRED_ACTION_FIELDS.difference(action_stats.keys())
    assert not missing, f"action_stats missing fields: {sorted(missing)}"
    assert int(action_stats["decision_count"]) > 0, "action_stats.decision_count == 0"
    assert int(action_stats["hold_count"]) + int(action_stats["switch_count"]) > 0, (
        "hold_count + switch_count == 0"
    )
    assert int(action_stats["phase_set_count"]) > 0, "RL evaluation did not set any TLS phase"

    fixed_stats = fixed.get("baseline_stats") or fixed.get("action_stats")
    assert isinstance(fixed_stats, dict), "eval_fixed.json has neither baseline_stats nor action_stats"
    assert int(fixed_stats.get("phase_set_count", -1)) == 0, "fixed baseline called setPhase"

    if real is not None:
        real_stats = real.get("baseline_stats") or {}
        assert int(real_stats.get("phase_set_count", -1)) == 0, "real_timing baseline called setPhase"
        assert "timing_source" in real, "eval_real_timing.json has no timing_source"

    metric_keys = [
        "avg_queue",
        "avg_waiting_time",
        "total_waiting_time",
        "throughput",
        "total_reward",
        "arrived",
    ]
    identical = all(abs(float(rl[key]) - float(fixed[key])) <= 1e-6 for key in metric_keys)
    assert not identical, "RL and fixed metrics are identical"

    for name, data in [("eval_rl.json", rl), ("eval_fixed.json", fixed)]:
        assert "pedestrian_metrics" in data, f"{name} has no pedestrian_metrics"
        assert "scenario_signature" in data, f"{name} has no scenario_signature"
        assert "seed" in data, f"{name} has no seed"
        assert "episode_seconds" in data, f"{name} has no episode_seconds"
        assert "step_length" in data, f"{name} has no step_length"
        assert "metrics" in data, f"{name} has no metrics block"

    rl_ped = rl["pedestrian_metrics"]
    fixed_ped = fixed["pedestrian_metrics"]
    assert (
        float(rl_ped.get("departed", 0)) > 0 or float(fixed_ped.get("departed", 0)) > 0
    ), "pedestrian_departed is zero for both RL and fixed"

    generated_peds = ROOT / "scenario" / "autogenerated_pedestrians.rou.xml"
    assert generated_peds.exists(), "autogenerated pedestrian demand file does not exist"
    sumocfg = (ROOT / "scenario" / "autogenerated.sumocfg").read_text(encoding="utf-8")
    assert "autogenerated_pedestrians.rou.xml" in sumocfg, (
        "autogenerated pedestrian demand file is not connected in autogenerated.sumocfg"
    )

    timing_file = ROOT / "scenario" / "tls.add.xml"
    if timing_file.exists():
        comparison = ROOT / "logs" / "comparison_rl_vs_real_timing.json"
        assert comparison.exists(), "comparison_rl_vs_real_timing.json does not exist"
        validate_pedestrian_comparison_file(
            comparison,
            ROOT / "logs" / "comparison_rl_vs_real_timing.csv",
            "real_timing",
        )
        native_cfg = (ROOT / "scenario" / "autogenerated_native.sumocfg").read_text(encoding="utf-8")
        real_cfg = (ROOT / "scenario" / "autogenerated_real_timing.sumocfg").read_text(encoding="utf-8")
        rl_cfg = (ROOT / "scenario" / "autogenerated_rl.sumocfg").read_text(encoding="utf-8")
        assert "tls.add.xml" not in native_cfg, "native fixed sumocfg includes tls.add.xml"
        assert "tls.add.xml" in real_cfg, "real timing sumocfg does not include tls.add.xml"
        assert "tls.add.xml" in rl_cfg, "rl sumocfg does not include tls.add.xml"

    native_comparison = ROOT / "logs" / "comparison_rl_vs_native_fixed.json"
    if not native_comparison.exists():
        native_comparison = ROOT / "logs" / "comparison_rl_vs_fixed_native.json"
    assert native_comparison.exists(), "native fixed comparison json does not exist"
    native_csv = ROOT / "logs" / "comparison_rl_vs_native_fixed.csv"
    if not native_csv.exists():
        native_csv = ROOT / "logs" / "comparison_rl_vs_fixed_native.csv"
    validate_pedestrian_comparison_file(native_comparison, native_csv, "native_fixed")
    validate_zero_division_pedestrian_comparison()


def validate_pedestrian_comparison_file(json_path: Path, csv_path: Path, baseline_name: str) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pedestrian = data.get("pedestrian_comparison")
    assert isinstance(pedestrian, dict), f"{json_path.name} has no pedestrian_comparison"
    for key in ["rl", "baseline", "delta", "improvement_pct"]:
        assert key in pedestrian, f"{json_path.name} pedestrian_comparison missing {key}"
    assert pedestrian["baseline"].get("name") == baseline_name, (
        f"{json_path.name} baseline name is {pedestrian['baseline'].get('name')} != {baseline_name}"
    )
    if float(pedestrian["rl"].get("departed") or 0) > 0:
        for key in ["departed", "arrived", "running", "waiting_count", "total_waiting_time", "avg_waiting_time"]:
            assert key in pedestrian["rl"], f"{json_path.name} pedestrian rl missing {key}"
            assert key in pedestrian["baseline"], f"{json_path.name} pedestrian baseline missing {key}"

    assert csv_path.exists(), f"{csv_path.name} does not exist"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
    required_columns = {
        "baseline_name",
        "pedestrian_rl_departed",
        "pedestrian_baseline_departed",
        "pedestrian_departed_delta",
        "pedestrian_rl_arrived",
        "pedestrian_baseline_arrived",
        "pedestrian_arrived_delta",
        "pedestrian_arrived_improvement_pct",
        "pedestrian_rl_running",
        "pedestrian_baseline_running",
        "pedestrian_running_delta",
        "pedestrian_rl_waiting_count",
        "pedestrian_baseline_waiting_count",
        "pedestrian_waiting_count_delta",
        "pedestrian_waiting_count_improvement_pct",
        "pedestrian_rl_total_waiting_time",
        "pedestrian_baseline_total_waiting_time",
        "pedestrian_total_waiting_time_delta",
        "pedestrian_total_waiting_time_improvement_pct",
        "pedestrian_rl_avg_waiting_time",
        "pedestrian_baseline_avg_waiting_time",
        "pedestrian_avg_waiting_time_delta",
        "pedestrian_avg_waiting_time_improvement_pct",
    }
    missing = required_columns.difference(header)
    assert not missing, f"{csv_path.name} missing pedestrian columns: {sorted(missing)}"


def validate_zero_division_pedestrian_comparison() -> None:
    from main import compare_pedestrian_metrics

    result = compare_pedestrian_metrics(
        {
            "departed": 1,
            "arrived": 1,
            "running": 0,
            "waiting_count": 1,
            "total_waiting_time": 1,
            "avg_waiting_time": 1,
        },
        {
            "departed": 0,
            "arrived": 0,
            "running": 0,
            "waiting_count": 0,
            "total_waiting_time": 0,
            "avg_waiting_time": 0,
        },
        "zero_test",
    )
    improvement = result["improvement_pct"]
    assert improvement["arrived"] is None, "arrived improvement should be null for zero baseline"
    assert improvement["avg_waiting_time"] is None, "avg waiting improvement should be null for zero baseline"
    assert improvement["total_waiting_time"] is None, "total waiting improvement should be null for zero baseline"
    assert improvement["waiting_count"] is None, "waiting count improvement should be null for zero baseline"


def run_action_probe() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/action_probe.py"],
        cwd=ROOT,
        env=validation_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
    )
    (ROOT / "logs" / "validation_action_probe_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (ROOT / "logs" / "validation_action_probe_stderr.txt").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, f"action_probe failed: {result.stdout}\n{result.stderr}"


def validate_invalid_signature() -> None:
    meta_path = ROOT / "checkpoints" / "dqn_meta.json"
    original = json.loads(meta_path.read_text(encoding="utf-8"))
    changed = dict(original)
    changed["scenario_signature"] = "INVALID_SIGNATURE_FOR_TEST"
    meta_path.write_text(json.dumps(changed, ensure_ascii=False, indent=2), encoding="utf-8")
    result = run_main("validation_main_invalid_signature")
    stdout = result.stdout
    assert "Checkpoint найден и совместим. Обучение пропущено." not in stdout, (
        "invalid signature was used as compatible checkpoint"
    )
    assert (
        "Checkpoint несовместим" in stdout
        or "Scenario signature mismatch" in stdout
        or "Запускаю обучение заново" in stdout
    ), "invalid signature did not trigger retrain/incompatible message"


def validate_corrupt_checkpoint() -> None:
    checkpoint = ROOT / "checkpoints" / "dqn.pt"
    before_run = time.time()
    checkpoint.write_bytes(b"not a valid torch checkpoint")
    result = run_main("validation_main_corrupt_checkpoint")
    stdout = result.stdout + result.stderr
    assert (
        "Checkpoint повреждён" in stdout
        or "Checkpoint load failed" in stdout
        or "Запускаю обучение заново" in stdout
    ), "corrupt checkpoint did not trigger retrain message"
    assert checkpoint.exists(), "dqn.pt missing after corrupt checkpoint recovery"
    assert checkpoint.stat().st_size > len(b"not a valid torch checkpoint"), (
        "dqn.pt was not replaced after corrupt checkpoint recovery"
    )
    assert checkpoint.stat().st_mtime >= before_run, "dqn.pt was not rewritten after corrupt checkpoint"


def main() -> int:
    try:
        validate_after_checkpoint_delete()
        run_action_probe()
        validate_invalid_signature()
        validate_corrupt_checkpoint()
        print("PIPELINE VALIDATION PASSED")
        return 0
    except Exception as exc:
        print(f"PIPELINE VALIDATION FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
