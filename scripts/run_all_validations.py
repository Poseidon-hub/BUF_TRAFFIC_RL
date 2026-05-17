import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VALIDATIONS = [
    "scripts/validate_training_params.py",
    "scripts/validate_double_dueling_dqn.py",
    "scripts/validate_pedestrian_reward.py",
    "scripts/validate_scenario_signature.py",
    "scripts/action_probe.py",
    "scripts/validate_pedestrians.py",
    "scripts/validate_real_timing_baseline.py",
    "scripts/validate_pipeline.py",
]


def main() -> int:
    for script in VALIDATIONS:
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0:
            print(f"VALIDATION FAILED: {script}")
            return 1
    print("ALL VALIDATIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
