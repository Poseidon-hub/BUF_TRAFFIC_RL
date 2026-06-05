from typing import Any, Dict

import numpy as np


def _cfg_value(obs_cfg: Any, name: str, default: float) -> float:
    if obs_cfg is None:
        return default
    if isinstance(obs_cfg, dict):
        return obs_cfg.get(name, default)
    return getattr(obs_cfg, name, default)


def encode_phase_onehot(phase_idx: int, num_phases: int, max_phases_global: int) -> np.ndarray:
    size = max(1, int(max_phases_global))
    onehot = np.zeros(size, dtype=np.float32)
    if num_phases > 0:
        idx = int(phase_idx) % int(num_phases)
        if idx < size:
            onehot[idx] = 1.0
    return onehot


def build_observation(
    tls_id: str,
    local_stats: Dict[str, float],
    neighbor_aggs: Dict[str, float],
    phase_onehot: np.ndarray,
    time_since_switch: float,
    obs_cfg: Any,
) -> np.ndarray:
    queue_norm = max(1e-6, float(_cfg_value(obs_cfg, "queue_norm", 20.0)))
    waiting_norm = max(1e-6, float(_cfg_value(obs_cfg, "waiting_norm", 120.0)))
    time_norm = max(
        1e-6,
        float(
            _cfg_value(
                obs_cfg,
                "time_since_switch_norm",
                _cfg_value(obs_cfg, "time_norm", 60.0),
            )
        ),
    )
    lane_norm = max(1e-6, float(_cfg_value(obs_cfg, "lane_norm", 16.0)))
    vehicle_norm = max(1e-6, float(_cfg_value(obs_cfg, "vehicle_norm", 20.0)))
    speed_norm = max(1e-6, float(_cfg_value(obs_cfg, "speed_norm", 15.0)))
    pressure_norm = max(
        1e-6,
        float(_cfg_value(obs_cfg, "pressure_norm", _cfg_value(obs_cfg, "reward_pressure_norm", 20.0))),
    )

    base = np.array(
        [
            float(local_stats.get("queue", 0.0)) / queue_norm,
            float(local_stats.get("waiting_time", 0.0)) / waiting_norm,
            float(local_stats.get("avg_speed", 0.0)) / speed_norm,
            float(time_since_switch) / time_norm,
            float(local_stats.get("incoming_vehicle_count", local_stats.get("vehicle_count", 0.0))) / vehicle_norm,
            float(local_stats.get("outgoing_vehicle_count", 0.0)) / vehicle_norm,
            float(local_stats.get("pressure", 0.0)) / pressure_norm,
            float(local_stats.get("phase_pressure", 0.0)) / pressure_norm,
            float(neighbor_aggs.get("mean_queue", 0.0)) / queue_norm,
            float(neighbor_aggs.get("mean_wait", 0.0)) / waiting_norm,
            float(neighbor_aggs.get("mean_pressure", 0.0)) / pressure_norm,
            float(neighbor_aggs.get("max_queue", 0.0)) / queue_norm,
            float(local_stats.get("num_lanes", 0.0)) / lane_norm,
        ],
        dtype=np.float32,
    )
    obs = np.concatenate([base, phase_onehot.astype(np.float32, copy=False)])
    return obs.astype(np.float32, copy=False)
