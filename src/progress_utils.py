import time
from typing import Optional


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def render_progress_bar(percent: float, width: int = 20) -> str:
    width = max(8, int(width))
    pct = max(0.0, min(100.0, float(percent)))
    filled = int(round(width * pct / 100.0))
    return "#" * filled + "-" * (width - filled)


class ProgressPrinter:
    def __init__(
        self,
        total: int,
        label: str,
        style: str = "single_line",
        width: int = 20,
        min_interval: float = 0.5,
        min_percent_delta: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.total = max(1, int(total))
        self.label = str(label)
        self.style = str(style or "single_line").lower()
        self.width = max(8, int(width))
        self.min_interval = max(0.0, float(min_interval))
        self.min_percent_delta = max(0.0, float(min_percent_delta))
        self.enabled = bool(enabled) and self.style != "none"
        self.started = time.perf_counter()
        self._last_print_time = 0.0
        self._last_percent: Optional[float] = None
        self._closed = False

    def update(self, current: int, suffix: str = "", force: bool = False) -> None:
        if not self.enabled or self._closed:
            return
        current = max(0, min(self.total, int(current)))
        percent = current * 100.0 / self.total
        now = time.perf_counter()
        percent_delta = 100.0 if self._last_percent is None else abs(percent - self._last_percent)
        should_print = (
            force
            or current >= self.total
            or self._last_percent is None
            or percent_delta >= self.min_percent_delta
            or now - self._last_print_time >= self.min_interval
        )
        if not should_print:
            return
        self._last_percent = percent
        self._last_print_time = now
        elapsed = max(0.0, now - self.started)
        eta = 0.0 if current <= 0 else elapsed * max(0, self.total - current) / max(1, current)
        text = (
            f"{self.label}: [{render_progress_bar(percent, self.width)}] "
            f"{percent:3.0f}% | {suffix}"
        ).rstrip()
        if current < self.total:
            text = f"{text} | ETA {format_seconds(eta)}"
        if self.style == "milestones":
            milestone = int(percent)
            if milestone in {0, 25, 50, 75, 100}:
                print(text, flush=True)
            return
        print("\r" + text, end="", flush=True)

    def close(self, suffix: str = "done") -> None:
        if not self.enabled or self._closed:
            return
        self.update(self.total, suffix=suffix, force=True)
        if self.style == "single_line":
            print()
        self._closed = True
