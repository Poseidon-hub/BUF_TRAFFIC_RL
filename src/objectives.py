from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from . import training_params as params


def metric_value(metrics: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    if name in metrics:
        value = metrics.get(name)
    else:
        value = (metrics.get("metrics") or {}).get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def add_normalized_metrics(metrics: dict) -> dict:
    departed = max(0.0, metric_value(metrics, "departed"))
    arrived = max(0.0, metric_value(metrics, "arrived"))
    cumulative_wait = metric_value(
        metrics,
        "cumulative_waiting_vehicle_seconds",
        metric_value(metrics, "legacy_total_waiting_time"),
    )
    total_time_loss = metric_value(metrics, "total_time_loss")

    metrics["normalized_waiting_time_per_departed"] = (
        cumulative_wait / departed if departed > 0.0 else None
    )
    metrics["normalized_waiting_time_per_arrived"] = (
        cumulative_wait / arrived if arrived > 0.0 else None
    )
    metrics["normalized_time_loss_per_departed"] = (
        total_time_loss / departed if departed > 0.0 else None
    )
    metrics["normalized_time_loss_per_arrived"] = (
        total_time_loss / arrived if arrived > 0.0 else None
    )
    return metrics


def weighted_mobility_score(
    metrics: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]] = None,
    weights: Optional[Mapping[str, float]] = None,
    normalization: Optional[Mapping[str, float]] = None,
    eps: float = params.OBJECTIVE_EPS,
) -> float:
    weights = dict(weights or params.OBJECTIVE_WEIGHTS)
    normalization = dict(normalization or params.OBJECTIVE_NORMALIZATION)
    score = 0.0
    for key, weight in weights.items():
        value = metric_value(metrics, key)
        if baseline is not None:
            denom = max(abs(metric_value(baseline, key)), float(eps))
        else:
            denom = max(abs(float(normalization.get(key, 1.0))), float(eps))
        score += float(weight) * (value / denom)
    return float(score)


def objective_payload(
    metrics: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]] = None,
    weights: Optional[Mapping[str, float]] = None,
    normalization: Optional[Mapping[str, float]] = None,
) -> dict:
    score = weighted_mobility_score(
        metrics,
        baseline=baseline,
        weights=weights,
        normalization=normalization,
    )
    return {
        "primary_objective": params.PRIMARY_OBJECTIVE,
        "objective_score": score,
        "objective_direction": "minimize",
        "objective_weights": dict(weights or params.OBJECTIVE_WEIGHTS),
        "objective_normalized_against": "baseline" if baseline is not None else "fixed_constants",
    }


def objective_improvement_pct(baseline_score: float, rl_score: float) -> Optional[float]:
    denom = max(abs(float(baseline_score)), params.OBJECTIVE_EPS)
    if not math.isfinite(denom) or denom <= 0.0:
        return None
    return (float(baseline_score) - float(rl_score)) / denom * 100.0
