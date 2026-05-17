import importlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_PARAMS = [
    "SEED",
    "USE_GUI",
    "FORCE_RETRAIN",
    "AUTO_RETRAIN_ON_SCENARIO_CHANGE",
    "STEP_LENGTH",
    "EPISODE_SECONDS",
    "TRAIN_EPISODES",
    "EVAL_EPISODES",
    "EVAL_SEEDS",
    "DECISION_INTERVAL",
    "MIN_GREEN",
    "ALGORITHM",
    "USE_DOUBLE_DQN",
    "USE_DUELING_DQN",
    "DUELING_AGGREGATION",
    "MODEL_HIDDEN_DIM",
    "MODEL_NUM_HIDDEN_LAYERS",
    "MODEL_ACTIVATION",
    "CHECKPOINT_VERSION",
    "FORCE_RETRAIN_ON_ALGORITHM_CHANGE",
    "GAMMA",
    "LEARNING_RATE",
    "BATCH_SIZE",
    "REPLAY_SIZE",
    "START_LEARNING_AFTER",
    "TRAIN_FREQ",
    "TARGET_UPDATE_FREQ",
    "GRAD_CLIP_NORM",
    "HIDDEN_DIM",
    "NUM_HIDDEN_LAYERS",
    "EPS_START",
    "EPS_END",
    "EPS_DECAY_STEPS",
    "EVAL_EPSILON",
    "REWARD_ALPHA_QUEUE",
    "REWARD_BETA_NEIGHBOR",
    "REWARD_USE_NEIGHBORS",
    "REWARD_NORMALIZE",
    "PEDESTRIAN_REWARD_ENABLED",
    "PEDESTRIAN_WAITING_PENALTY",
    "REWARD_USE_PEDESTRIANS",
    "REWARD_VEHICLE_QUEUE_WEIGHT",
    "REWARD_VEHICLE_WAIT_WEIGHT",
    "REWARD_NEIGHBOR_WEIGHT",
    "REWARD_PEDESTRIAN_WAIT_WEIGHT",
    "REWARD_PEDESTRIAN_RUNNING_WEIGHT",
    "REWARD_PEDESTRIAN_BLOCKED_WEIGHT",
    "PEDESTRIAN_REWARD_NORMALIZATION",
    "REWARD_CAR_PRIORITY_RATIO_NOTE",
    "PEDESTRIAN_REWARD_SCOPE",
    "STRICT_REWARD_VALIDATION",
    "REAL_TIMING_FILE",
    "USE_REAL_TIMING_BASELINE",
    "RL_USE_REAL_TIMING_PROGRAM_AS_BASE",
    "RUN_REAL_TIMING_BASELINE",
    "RUN_NATIVE_FIXED_BASELINE",
    "PRIMARY_BASELINE",
    "STRICT_REAL_TIMING_VALIDATION",
    "SCENARIO_DIR",
    "CHECKPOINT_DIR",
    "CHECKPOINT_PATH",
    "CHECKPOINT_META_PATH",
    "LOGS_DIR",
    "SAVE_TRAIN_LOGS",
    "SAVE_EVAL_LOGS",
    "PRINT_ACTION_DIAGNOSTICS",
    "PRINT_SCENARIO_INFO",
]


def main() -> int:
    try:
        path = ROOT / "src" / "training_params.py"
        assert path.exists(), "src/training_params.py does not exist"
        module = importlib.import_module("src.training_params")
        missing = [name for name in REQUIRED_PARAMS if not hasattr(module, name)]
        assert not missing, f"missing params: {missing}"

        text = path.read_text(encoding="utf-8")
        comment_count = sum(1 for line in text.splitlines() if line.strip().startswith("#"))
        assert comment_count >= 20, f"expected >=20 comments, got {comment_count}"

        for module_path in ["main.py", "src/train.py", "src/eval.py", "src/sumo_env.py"]:
            source = (ROOT / module_path).read_text(encoding="utf-8")
            assert (
                "training_params" in source or "make_config" in source or "config" in source
            ), f"{module_path} is not connected to config/training_params"

        forbidden = ["300", "3600", "0.001", "0.99", "10000", "64"]
        scan_paths = [ROOT / name for name in ["main.py", "src/train.py", "src/eval.py", "src/sumo_env.py"]]
        violations = []
        for scan_path in scan_paths:
            for lineno, line in enumerate(scan_path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "training_params" in str(scan_path).replace("\\", "/"):
                    continue
                code = stripped.split("#", 1)[0]
                for token in forbidden:
                    if re.search(rf"(?<![\w.]){re.escape(token)}(?![\w.])", code):
                        violations.append(f"{scan_path.relative_to(ROOT)}:{lineno}: {token}")
        assert not violations, "magic training numbers outside training_params: " + "; ".join(violations)

        print("TRAINING PARAMS VALIDATION PASSED")
        return 0
    except Exception as exc:
        print(f"TRAINING PARAMS VALIDATION FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
