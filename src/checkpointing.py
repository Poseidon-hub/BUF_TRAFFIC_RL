import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .logging_utils import save_json


def checkpoint_file_exists(path: Path) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_checkpoint_header(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    if not checkpoint_file_exists(path):
        return None, "checkpoint file is missing or empty"
    try:
        import torch

        checkpoint = torch.load(Path(path), map_location="cpu")
        return {
            "obs_dim": int(checkpoint.get("obs_dim", -1)),
            "action_dim": int(checkpoint.get("action_dim", -1)),
            "steps_done": int(checkpoint.get("steps_done", 0)),
            "global_step": int(checkpoint.get("global_step", checkpoint.get("steps_done", 0))),
            "algorithm": str(checkpoint.get("algorithm", "dqn")),
            "use_double_dqn": bool(checkpoint.get("use_double_dqn", False)),
            "use_dueling_dqn": bool(checkpoint.get("use_dueling_dqn", False)),
            "checkpoint_version": int(checkpoint.get("checkpoint_version", 1)),
        }, None
    except Exception as exc:
        return None, f"Checkpoint поврежден или несовместим: {type(exc).__name__}: {exc}"


def load_checkpoint_metadata(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    path = Path(path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return None, "checkpoint metadata file is missing or empty"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"checkpoint metadata is unreadable: {type(exc).__name__}: {exc}"


def save_checkpoint_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    save_json(tmp_path, metadata)
    tmp_path.replace(path)


def build_scenario_signature(
    scenario,
    tls_ids,
    obs_dim: int,
    action_dim: int,
    num_phases_per_tls: Dict[str, int],
) -> str:
    payload = {
        "files": [_file_fingerprint(path, scenario.scenario_dir) for path in _scenario_files(scenario)],
        "tls_ids": list(tls_ids),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "num_phases_per_tls": {str(k): int(v) for k, v in sorted(num_phases_per_tls.items())},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint_metadata(config, scenario, tls_ids, obs_dim: int, num_phases_per_tls: Dict[str, int]) -> dict:
    action_dim = int(getattr(config, "action_size", 2))
    signature = build_scenario_signature(
        scenario,
        tls_ids=tls_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_phases_per_tls=num_phases_per_tls,
    )
    return {
        "scenario_signature": signature,
        "scenario_dir": str(Path(scenario.scenario_dir).resolve()),
        "sumocfg_path": _path_value(scenario.sumocfg),
        "net_file": _path_value(scenario.net),
        "route_files": [_path_value(path) for path in scenario.route_files],
        "additional_files": [_path_value(path) for path in scenario.additional_files],
        "timing_files": [_path_value(path) for path in getattr(scenario, "timing_files", [])],
        "real_timing_file": _path_value(getattr(scenario, "real_timing_file", None)),
        "mode_sumocfgs": {
            str(mode): _path_value(path)
            for mode, path in getattr(scenario, "mode_sumocfgs", {}).items()
        },
        "tls_ids": list(tls_ids),
        "tls_count": len(tls_ids),
        "algorithm": str(getattr(config, "algorithm", "dqn")),
        "use_double_dqn": bool(getattr(config, "use_double_dqn", False)),
        "use_dueling_dqn": bool(getattr(config, "use_dueling_dqn", False)),
        "dueling_aggregation": str(getattr(config, "dueling_aggregation", "mean")),
        "checkpoint_version": int(getattr(config, "checkpoint_version", 1)),
        "model_hidden_dim": int(getattr(config, "model_hidden_dim", getattr(config, "hidden_dim", 128))),
        "model_num_hidden_layers": int(
            getattr(config, "model_num_hidden_layers", getattr(config, "num_hidden_layers", 2))
        ),
        "model_activation": str(getattr(config, "model_activation", "relu")),
        "initial_switch_bias": float(getattr(config, "initial_switch_bias", 0.0)),
        "obs_dim": int(obs_dim),
        "action_dim": action_dim,
        "num_phases_per_tls": {str(k): int(v) for k, v in sorted(num_phases_per_tls.items())},
        "training_params": _training_params_snapshot(config),
        "run_config": _json_safe(getattr(config, "run_config", getattr(config, "last_run_setup", {}))),
        "demand_mode": _json_safe(getattr(config, "demand_mode", None)),
        "actual_vehicle_count": _json_safe(
            (getattr(config, "demand_info", {}) or {}).get("actual_vehicle_count")
        ),
        "pedestrian_count": _json_safe(getattr(config, "pedestrian_demand_count", None)),
        "pedestrian_mode": _json_safe(getattr(config, "pedestrian_mode", None)),
        "eval_seconds": _json_safe(getattr(config, "evaluation_seconds", None)),
        "train_episodes": _json_safe(getattr(config, "train_episodes", None)),
        "reward_variant": _json_safe(getattr(config, "reward_variant", None)),
        "learning_rate": _json_safe(getattr(config, "lr", None)),
        "gamma": _json_safe(getattr(config, "gamma", None)),
        "min_green": _json_safe(getattr(config, "min_green", None)),
        "max_green": _json_safe(getattr(config, "max_green", None)),
        "control_decision_interval": _json_safe(getattr(config, "control_decision_interval", None)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_checkpoint_compatibility(config, current_metadata: dict) -> Tuple[bool, str]:
    checkpoint_path = Path(config.checkpoint_path)
    meta_path = Path(config.checkpoint_meta_path)
    header, header_error = load_checkpoint_header(checkpoint_path)
    if header_error:
        return False, header_error

    saved_meta, meta_error = load_checkpoint_metadata(meta_path)
    if meta_error:
        return False, meta_error

    checks = [
        ("scenario_signature", saved_meta.get("scenario_signature"), current_metadata.get("scenario_signature")),
        ("algorithm", saved_meta.get("algorithm", "dqn"), current_metadata.get("algorithm")),
        ("checkpoint_version", saved_meta.get("checkpoint_version", 1), current_metadata.get("checkpoint_version")),
        ("use_double_dqn", saved_meta.get("use_double_dqn", False), current_metadata.get("use_double_dqn")),
        ("use_dueling_dqn", saved_meta.get("use_dueling_dqn", False), current_metadata.get("use_dueling_dqn")),
        (
            "model_hidden_dim",
            saved_meta.get("model_hidden_dim"),
            current_metadata.get("model_hidden_dim"),
        ),
        (
            "model_num_hidden_layers",
            saved_meta.get("model_num_hidden_layers"),
            current_metadata.get("model_num_hidden_layers"),
        ),
        ("model_activation", saved_meta.get("model_activation"), current_metadata.get("model_activation")),
        ("initial_switch_bias", saved_meta.get("initial_switch_bias"), current_metadata.get("initial_switch_bias")),
        ("obs_dim", saved_meta.get("obs_dim"), current_metadata.get("obs_dim")),
        ("action_dim", saved_meta.get("action_dim"), current_metadata.get("action_dim")),
        ("tls_ids", saved_meta.get("tls_ids"), current_metadata.get("tls_ids")),
        ("tls_count", saved_meta.get("tls_count"), current_metadata.get("tls_count")),
        (
            "num_phases_per_tls",
            saved_meta.get("num_phases_per_tls"),
            current_metadata.get("num_phases_per_tls"),
        ),
    ]
    for name, saved, current in checks:
        if saved != current:
            return False, f"{name} mismatch: checkpoint={saved}, current={current}"

    if int(header.get("obs_dim", -1)) != int(current_metadata.get("obs_dim", -2)):
        return False, f"obs_dim mismatch in dqn.pt: {header.get('obs_dim')} != {current_metadata.get('obs_dim')}"
    if int(header.get("action_dim", -1)) != int(current_metadata.get("action_dim", -2)):
        return False, (
            f"action_dim mismatch in dqn.pt: "
            f"{header.get('action_dim')} != {current_metadata.get('action_dim')}"
        )
    header_checks = [
        ("algorithm", header.get("algorithm"), current_metadata.get("algorithm")),
        ("checkpoint_version", header.get("checkpoint_version"), current_metadata.get("checkpoint_version")),
        ("use_double_dqn", header.get("use_double_dqn"), current_metadata.get("use_double_dqn")),
        ("use_dueling_dqn", header.get("use_dueling_dqn"), current_metadata.get("use_dueling_dqn")),
    ]
    for name, saved, current in header_checks:
        if saved != current:
            return False, f"{name} mismatch in dqn.pt: {saved} != {current}"

    return True, "checkpoint is compatible"


def _scenario_files(scenario) -> list:
    paths = []
    for path in [
        scenario.net,
        *scenario.route_files,
        *scenario.additional_files,
        *getattr(scenario, "timing_files", []),
    ]:
        if path is not None:
            paths.append(Path(path))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _file_fingerprint(path: Path, base_dir: Path) -> dict:
    path = Path(path)
    try:
        stat = path.stat()
        content_hash = _sha256_file(path)
        try:
            rel = path.resolve().relative_to(Path(base_dir).resolve()).as_posix()
        except Exception:
            rel = str(path.resolve())
        return {
            "path": rel,
            "size": int(stat.st_size),
            "sha256": content_hash,
        }
    except Exception as exc:
        return {"path": str(path), "missing": True, "error": str(exc)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_params_snapshot(config) -> dict:
    keys = [
        "seed",
        "step_length",
        "episode_seconds",
        "train_steps",
        "eval_episodes",
        "eval_seeds",
        "decision_interval",
        "min_green",
        "algorithm",
        "use_double_dqn",
        "use_dueling_dqn",
        "dueling_aggregation",
        "checkpoint_version",
        "scenario_preferred_prefix",
        "allow_mixed_scenario_files",
        "kaliningrad_validation_enabled",
        "model_hidden_dim",
        "model_num_hidden_layers",
        "model_activation",
        "gamma",
        "lr",
        "batch_size",
        "replay_size",
        "start_learning_after",
        "train_freq",
        "target_update_freq",
        "grad_clip_norm",
        "hidden_dim",
        "num_hidden_layers",
        "eps_start",
        "eps_end",
        "eps_decay_steps",
        "eval_epsilon",
        "alpha",
        "beta",
        "reward_use_neighbors",
        "reward_normalize",
        "pedestrian_reward_enabled",
        "pedestrian_waiting_penalty",
        "reward_use_pedestrians",
        "reward_vehicle_queue_weight",
        "reward_vehicle_wait_weight",
        "reward_neighbor_weight",
        "reward_pedestrian_wait_weight",
        "reward_pedestrian_running_weight",
        "reward_pedestrian_blocked_weight",
        "pedestrian_priority_max_share",
        "pedestrian_reward_normalization",
        "pedestrian_reward_scope",
        "strict_reward_validation",
    ]
    return {key: _json_safe(getattr(config, key, None)) for key in keys}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _path_value(path) -> Optional[str]:
    return str(Path(path).resolve()) if path else None
