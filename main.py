from pathlib import Path

from src.diploma_runner import run


if __name__ == "__main__":
    raise SystemExit(run(Path(__file__).resolve().parent))
