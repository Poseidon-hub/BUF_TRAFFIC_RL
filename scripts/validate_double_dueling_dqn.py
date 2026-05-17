import json
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import training_params as params
from src.checkpointing import validate_checkpoint_compatibility
from src.config import make_config
from src.dqn import DQNShared, QNetwork


def validation_env() -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "BFU_FAST_TEST": "1",
            "BFU_FORCE_RETRAIN": "0",
            "BFU_EPISODE_SECONDS": "120",
            "BFU_TRAIN_SECONDS": "300",
            "BFU_EVAL_EPISODES": "1",
            "BFU_EVAL_SEEDS": "42",
        }
    )
    return env


def validate_training_params() -> None:
    assert params.ALGORITHM == "double_dueling_dqn"
    assert params.USE_DOUBLE_DQN is True
    assert params.USE_DUELING_DQN is True
    assert params.CHECKPOINT_VERSION == 2


def validate_model_forward() -> None:
    model = QNetwork(
        obs_dim=10,
        action_dim=2,
        hidden_dim=128,
        num_hidden_layers=2,
        use_dueling=True,
        dueling_aggregation="mean",
    )
    x = torch.randn(4, 10)
    y = model(x)
    assert list(y.shape) == [4, 2], f"unexpected model output shape: {list(y.shape)}"
    assert model.value_stream is not None, "dueling model has no value_stream"
    assert model.advantage_stream is not None, "dueling model has no advantage_stream"

    with torch.no_grad():
        features = model.feature_extractor(x)
        value = model.value_stream(features)
        advantage = model.advantage_stream(features)
        expected = value + advantage - advantage.mean(dim=1, keepdim=True)
    assert torch.allclose(y, expected, atol=1e-6), "dueling aggregation is not mean advantage"


def validate_double_target() -> None:
    cfg = make_config(ROOT)
    cfg.algorithm = "double_dueling_dqn"
    cfg.use_double_dqn = True
    cfg.use_dueling_dqn = True
    agent = DQNShared(obs_dim=10, action_dim=2, config=cfg)
    next_states = torch.randn(4, 10, device=agent.device)
    rewards = torch.randn(4, device=agent.device)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0], device=agent.device)

    with torch.no_grad():
        next_actions = agent.q_net(next_states).argmax(dim=1)
        target_next_q = agent.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected = rewards + agent.gamma * (1.0 - dones) * target_next_q
        actual = agent.compute_td_target(rewards, next_states, dones)

    assert list(actual.shape) == list(rewards.shape), "TD target shape does not match rewards shape"
    assert torch.allclose(actual, expected, atol=1e-6), "Double DQN target calculation is wrong"


def run_main_after_checkpoint_delete() -> None:
    checkpoint = ROOT / "checkpoints" / "dqn.pt"
    meta = ROOT / "checkpoints" / "dqn_meta.json"
    checkpoint.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)

    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT,
        env=validation_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=360,
    )
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "validation_double_dueling_main_stdout.txt").write_text(
        result.stdout,
        encoding="utf-8",
    )
    (ROOT / "logs" / "validation_double_dueling_main_stderr.txt").write_text(
        result.stderr,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert checkpoint.exists() and checkpoint.stat().st_size > 0, "dqn.pt was not created"
    assert meta.exists() and meta.stat().st_size > 0, "dqn_meta.json was not created"


def validate_checkpoint_metadata_and_incompatibility() -> None:
    cfg = make_config(ROOT)
    meta_path = cfg.checkpoint_meta_path
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ["algorithm", "use_double_dqn", "use_dueling_dqn", "checkpoint_version"]:
        assert key in metadata, f"dqn_meta.json missing {key}"
    assert metadata["algorithm"] == "double_dueling_dqn"
    assert metadata["use_double_dqn"] is True
    assert metadata["use_dueling_dqn"] is True
    assert int(metadata["checkpoint_version"]) == 2

    original = dict(metadata)
    changed = dict(metadata)
    changed["algorithm"] = "dqn"
    meta_path.write_text(json.dumps(changed, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        compatible, reason = validate_checkpoint_compatibility(cfg, original)
        assert not compatible, "old algorithm metadata was considered compatible"
        assert "algorithm" in reason, f"unexpected incompatibility reason: {reason}"
    finally:
        meta_path.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    try:
        validate_training_params()
        validate_model_forward()
        validate_double_target()
        run_main_after_checkpoint_delete()
        validate_checkpoint_metadata_and_incompatibility()
        print("DOUBLE DUELING DQN VALIDATION PASSED")
        return 0
    except Exception as exc:
        print(f"DOUBLE DUELING DQN VALIDATION FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
