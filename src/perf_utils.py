import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


LOG_LEVELS = {
    "silent": 0,
    "compact": 1,
    "verbose": 2,
    "debug": 3,
}


def configure_console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def normalize_log_level(value: Any) -> str:
    normalized = str(value or "compact").strip().lower()
    return normalized if normalized in LOG_LEVELS else "compact"


def should_log(config: Any, level: str = "compact") -> bool:
    current = LOG_LEVELS.get(normalize_log_level(getattr(config, "log_level", "compact")), 1)
    required = LOG_LEVELS.get(normalize_log_level(level), 1)
    return current >= required


def is_silent(config: Any) -> bool:
    return LOG_LEVELS.get(normalize_log_level(getattr(config, "log_level", "compact")), 1) <= 0


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.start)


class PerformanceTracker:
    def __init__(self) -> None:
        self._total = Timer()
        self.stages: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        timer = Timer()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + timer.elapsed()

    def summary(self, device_info: Optional[dict] = None) -> dict:
        summary = dict(device_info or {})
        for key, value in self.stages.items():
            summary[f"{key}_seconds"] = round(float(value), 6)
        summary["total_seconds"] = round(self._total.elapsed(), 6)
        for expected in (
            "scenario_loading_seconds",
            "sumo_smoke_test_seconds",
            "graph_building_seconds",
            "training_seconds",
            "eval_rl_seconds",
            "eval_real_timing_seconds",
            "eval_native_fixed_seconds",
            "comparison_saving_seconds",
        ):
            summary.setdefault(expected, 0.0)
        return summary


def memory_usage_mb() -> Optional[float]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def configure_torch_runtime(config: Any) -> dict:
    try:
        import torch
    except Exception:
        return {
            "cpu_count": os.cpu_count(),
            "torch_num_threads": None,
            "torch_num_interop_threads": None,
        }

    cpu_count = os.cpu_count() or 1
    threads_cfg = str(getattr(config, "torch_num_threads", "auto")).lower()
    interop_cfg = str(getattr(config, "torch_num_interop_threads", "auto")).lower()
    torch_threads = min(cpu_count, 8) if threads_cfg == "auto" else max(1, int(threads_cfg))
    interop_threads = 1 if interop_cfg == "auto" else max(1, int(interop_cfg))

    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass

    return {
        "cpu_count": cpu_count,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def resolve_torch_device(config: Any):
    import torch

    requested = str(getattr(config, "device", "auto") or "auto").strip().lower()
    require_cuda = bool(getattr(config, "require_cuda", False))
    cuda_available = bool(torch.cuda.is_available())
    if (requested == "cuda" or require_cuda) and not cuda_available:
        raise RuntimeError("DEVICE='cuda' was requested, but torch.cuda.is_available() is False.")
    if require_cuda:
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


def collect_device_info(config: Any = None, device: Any = None) -> dict:
    info = {
        "device": str(device) if device is not None else None,
        "cuda_available": False,
        "cuda_device_name": None,
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "memory_usage_mb": memory_usage_mb(),
    }
    try:
        import torch

        resolved = device if device is not None else resolve_torch_device(config)
        info["device"] = str(resolved)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_num_threads"] = int(torch.get_num_threads())
        info["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
        if info["cuda_available"]:
            idx = resolved.index if getattr(resolved, "index", None) is not None else 0
            info["cuda_device_name"] = torch.cuda.get_device_name(idx)
            info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 3)
            info["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(idx) / (1024 * 1024), 3)
    except Exception:
        pass
    return info


def save_performance_summary(path: Path, summary: dict) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


class ProgressBar:
    def __init__(self, total: int, desc: str, enabled: bool = True, width: int = 24):
        self.total = max(1, int(total))
        self.desc = desc
        self.enabled = bool(enabled)
        self.width = max(8, int(width))
        self._last_percent = -1
        self._closed = False
        self._tqdm = None
        if not self.enabled:
            return
        try:
            from tqdm import tqdm

            self._tqdm = tqdm(total=self.total, desc=self.desc, leave=True, dynamic_ncols=True)
            self._current = 0
        except Exception:
            self._current = 0

    def update(self, current: int, suffix: str = "") -> None:
        if not self.enabled or self._closed:
            return
        current = max(0, min(self.total, int(current)))
        if self._tqdm is not None:
            delta = current - self._current
            if delta > 0:
                self._tqdm.update(delta)
                self._current = current
            if suffix:
                self._tqdm.set_postfix_str(suffix)
            return

        percent = int(current * 100 / self.total)
        if percent == self._last_percent and current < self.total:
            return
        self._last_percent = percent
        filled = int(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        text = f"\r{self.desc}: [{bar}] {percent:3d}%"
        if suffix:
            text += f" | {suffix}"
        print(text, end="", flush=True)

    def close(self, suffix: str = "done") -> None:
        if not self.enabled or self._closed:
            return
        if self._tqdm is not None:
            self._tqdm.close()
        else:
            self.update(self.total, suffix=suffix)
            print()
        self._closed = True


def format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)}m {rem:.1f}s"
