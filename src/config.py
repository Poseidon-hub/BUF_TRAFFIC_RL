from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Dict, Any

from . import training_params as params


OBS_BASE_SIZE = 6
DEFAULT_MAX_PHASES = 16
OBS_SIZE = OBS_BASE_SIZE + DEFAULT_MAX_PHASES
ACTION_SIZE = 2

SCENARIO_DIR = params.SCENARIO_DIR
CHECKPOINT_DIR = params.CHECKPOINT_DIR
LOG_DIR = params.LOGS_DIR

SCENARIO_NET_FILE = None
SCENARIO_ROUTE_FILE = None
SCENARIO_SUMOCFG_FILE = None

AUTO_GENERATE_SUMOCFG = True
AUTO_RETRAIN_ON_SHAPE_MISMATCH = params.AUTO_RETRAIN_ON_SCENARIO_CHANGE

STEP_LENGTH = params.STEP_LENGTH
EPISODE_SECONDS = params.EPISODE_SECONDS
TRAIN_SECONDS = params.TRAIN_SECONDS
EVAL_EPISODES = params.EVAL_EPISODES

MIN_GREEN = params.MIN_GREEN
SEED = params.SEED
USE_GUI = params.USE_GUI

SUMO_EXTRA_ARGS = list(params.SUMO_EXTRA_ARGS)

QUEUE_NORM = params.QUEUE_NORM
WAIT_NORM = params.WAIT_NORM
TIME_SINCE_SWITCH_NORM = params.TIME_SINCE_SWITCH_NORM
LANE_NORM = params.LANE_NORM

DEBUG_SCENARIO = params.DEBUG_SCENARIO


@dataclass
class Config:
    project_root: Path = Path(".")

    step_length: float = STEP_LENGTH
    episode_seconds: int = EPISODE_SECONDS
    train_steps: int = TRAIN_SECONDS
    eval_episodes: int = EVAL_EPISODES
    eval_seeds: Any = None
    decision_interval: int = params.DECISION_INTERVAL
    min_green: int = MIN_GREEN

    algorithm: str = params.ALGORITHM
    use_double_dqn: bool = params.USE_DOUBLE_DQN
    use_dueling_dqn: bool = params.USE_DUELING_DQN
    dueling_aggregation: str = params.DUELING_AGGREGATION
    checkpoint_version: int = params.CHECKPOINT_VERSION
    force_retrain_on_algorithm_change: bool = params.FORCE_RETRAIN_ON_ALGORITHM_CHANGE

    gamma: float = params.GAMMA
    lr: float = params.LEARNING_RATE
    batch_size: int = params.BATCH_SIZE
    replay_size: int = params.REPLAY_SIZE
    start_learning_after: int = params.START_LEARNING_AFTER
    train_freq: int = params.TRAIN_FREQ
    target_update_freq: int = params.TARGET_UPDATE_FREQ
    grad_clip_norm: float = params.GRAD_CLIP_NORM
    hidden_dim: int = params.HIDDEN_DIM
    num_hidden_layers: int = params.NUM_HIDDEN_LAYERS
    model_hidden_dim: int = params.MODEL_HIDDEN_DIM
    model_num_hidden_layers: int = params.MODEL_NUM_HIDDEN_LAYERS
    model_activation: str = params.MODEL_ACTIVATION

    eps_start: float = params.EPS_START
    eps_end: float = params.EPS_END
    eps_decay_steps: int = params.EPS_DECAY_STEPS
    eval_epsilon: float = params.EVAL_EPSILON

    alpha: float = params.REWARD_ALPHA_QUEUE
    beta: float = params.REWARD_BETA_NEIGHBOR
    reward_use_neighbors: bool = params.REWARD_USE_NEIGHBORS
    reward_normalize: bool = params.REWARD_NORMALIZE
    pedestrian_reward_enabled: bool = params.PEDESTRIAN_REWARD_ENABLED
    pedestrian_waiting_penalty: float = params.PEDESTRIAN_WAITING_PENALTY
    reward_use_pedestrians: bool = params.REWARD_USE_PEDESTRIANS
    reward_vehicle_queue_weight: float = params.REWARD_VEHICLE_QUEUE_WEIGHT
    reward_vehicle_wait_weight: float = params.REWARD_VEHICLE_WAIT_WEIGHT
    reward_neighbor_weight: float = params.REWARD_NEIGHBOR_WEIGHT
    reward_pedestrian_wait_weight: float = params.REWARD_PEDESTRIAN_WAIT_WEIGHT
    reward_pedestrian_running_weight: float = params.REWARD_PEDESTRIAN_RUNNING_WEIGHT
    reward_pedestrian_blocked_weight: float = params.REWARD_PEDESTRIAN_BLOCKED_WEIGHT
    pedestrian_reward_normalization: float = params.PEDESTRIAN_REWARD_NORMALIZATION
    reward_car_priority_ratio_note: str = params.REWARD_CAR_PRIORITY_RATIO_NOTE
    pedestrian_reward_scope: str = params.PEDESTRIAN_REWARD_SCOPE
    strict_reward_validation: bool = params.STRICT_REWARD_VALIDATION
    reward_stuck_phase_penalty_enabled: bool = params.REWARD_STUCK_PHASE_PENALTY_ENABLED
    reward_stuck_phase_after_seconds: int = params.REWARD_STUCK_PHASE_AFTER_SECONDS
    reward_stuck_phase_penalty: float = params.REWARD_STUCK_PHASE_PENALTY
    reward_switch_penalty: float = params.REWARD_SWITCH_PENALTY
    quality_gate_enabled: bool = params.QUALITY_GATE_ENABLED
    quality_gate_max_retrain_rounds: int = params.QUALITY_GATE_MAX_RETRAIN_ROUNDS
    quality_gate_extra_episodes: int = params.QUALITY_GATE_EXTRA_EPISODES
    min_rl_phase_set_count: int = params.MIN_RL_PHASE_SET_COUNT
    strict_dev_validation: bool = params.STRICT_DEV_VALIDATION
    require_rl_phase_switches: bool = params.REQUIRE_RL_PHASE_SWITCHES
    require_rl_fixed_metric_difference: bool = params.REQUIRE_RL_FIXED_METRIC_DIFFERENCE
    metric_diff_tolerance: float = params.METRIC_DIFF_TOLERANCE
    auto_generate_pedestrians_if_missing: bool = params.AUTO_GENERATE_PEDESTRIANS_IF_MISSING
    pedestrian_demand_count: int = params.PEDESTRIAN_DEMAND_COUNT
    pedestrian_demand_begin: int = params.PEDESTRIAN_DEMAND_BEGIN
    pedestrian_demand_end: int = params.PEDESTRIAN_DEMAND_END
    pedestrian_demand_prefix: str = params.PEDESTRIAN_DEMAND_PREFIX
    require_nonzero_pedestrian_metrics: bool = params.REQUIRE_NONZERO_PEDESTRIAN_METRICS
    real_timing_file: str = params.REAL_TIMING_FILE
    use_real_timing_baseline: bool = params.USE_REAL_TIMING_BASELINE
    rl_use_real_timing_program_as_base: bool = params.RL_USE_REAL_TIMING_PROGRAM_AS_BASE
    run_real_timing_baseline: bool = params.RUN_REAL_TIMING_BASELINE
    run_native_fixed_baseline: bool = params.RUN_NATIVE_FIXED_BASELINE
    primary_baseline: str = params.PRIMARY_BASELINE
    strict_real_timing_validation: bool = params.STRICT_REAL_TIMING_VALIDATION

    seed: int = SEED
    log_freq: int = 100

    queue_norm: float = QUEUE_NORM
    waiting_norm: float = WAIT_NORM
    time_norm: float = TIME_SINCE_SWITCH_NORM
    time_since_switch_norm: float = TIME_SINCE_SWITCH_NORM
    lane_norm: float = LANE_NORM
    max_phases_global: int = DEFAULT_MAX_PHASES
    obs_size: int = OBS_SIZE
    action_size: int = ACTION_SIZE

    scenario_dir_name: str = SCENARIO_DIR
    checkpoints_dir_name: str = CHECKPOINT_DIR
    logs_dir_name: str = LOG_DIR
    checkpoint_name: str = "dqn.pt"
    checkpoint_meta_name: str = "dqn_meta.json"
    scenario_net_file: Any = SCENARIO_NET_FILE
    scenario_route_file: Any = SCENARIO_ROUTE_FILE
    scenario_sumocfg_file: Any = SCENARIO_SUMOCFG_FILE
    auto_generate_sumocfg: bool = AUTO_GENERATE_SUMOCFG
    auto_retrain_on_shape_mismatch: bool = AUTO_RETRAIN_ON_SHAPE_MISMATCH
    force_retrain: bool = params.FORCE_RETRAIN
    use_gui: bool = USE_GUI
    sumo_extra_args: Any = None
    debug_scenario: bool = DEBUG_SCENARIO
    save_train_logs: bool = params.SAVE_TRAIN_LOGS
    save_eval_logs: bool = params.SAVE_EVAL_LOGS
    print_action_diagnostics: bool = params.PRINT_ACTION_DIAGNOSTICS
    print_scenario_info: bool = params.PRINT_SCENARIO_INFO

    @property
    def scenario_dir(self) -> Path:
        return self.project_root / self.scenario_dir_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / self.checkpoints_dir_name

    @property
    def logs_dir(self) -> Path:
        return self.project_root / self.logs_dir_name

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoints_dir / self.checkpoint_name

    @property
    def checkpoint_meta_path(self) -> Path:
        return self.checkpoints_dir / self.checkpoint_meta_name

    @property
    def train_seconds(self) -> int:
        return self.train_steps

    def set_observation_phases(self, max_phases_global: int) -> None:
        self.max_phases_global = max(1, int(max_phases_global))
        self.obs_size = OBS_BASE_SIZE + self.max_phases_global

    def to_json_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["project_root"] = str(self.project_root)
        data["scenario_dir"] = str(self.scenario_dir)
        data["checkpoints_dir"] = str(self.checkpoints_dir)
        data["logs_dir"] = str(self.logs_dir)
        data["checkpoint_path"] = str(self.checkpoint_path)
        return data


def make_config(project_root: Path = Path(".")) -> Config:
    cfg = Config(project_root=Path(project_root).resolve())
    cfg.eval_seeds = list(params.EVAL_SEEDS)
    cfg.sumo_extra_args = list(SUMO_EXTRA_ARGS)
    _apply_env_overrides(cfg)
    cfg.set_observation_phases(DEFAULT_MAX_PHASES)
    return cfg


def _apply_env_overrides(cfg: Config) -> None:
    if _env_bool("BFU_FAST_TEST", False):
        cfg.episode_seconds = 120
        cfg.train_steps = 120
        cfg.eval_episodes = 1
        cfg.eval_seeds = [cfg.seed]
        cfg.debug_scenario = False

    cfg.force_retrain = _env_bool("BFU_FORCE_RETRAIN", cfg.force_retrain)
    cfg.strict_dev_validation = _env_bool("BFU_STRICT_DEV_VALIDATION", cfg.strict_dev_validation)
    cfg.episode_seconds = _env_int("BFU_EPISODE_SECONDS", cfg.episode_seconds)
    cfg.train_steps = _env_int("BFU_TRAIN_SECONDS", cfg.train_steps)
    train_episodes = os.environ.get("BFU_TRAIN_EPISODES")
    if train_episodes is not None and "BFU_TRAIN_SECONDS" not in os.environ:
        cfg.train_steps = max(1, int(train_episodes)) * int(cfg.episode_seconds)
    cfg.eval_episodes = _env_int("BFU_EVAL_EPISODES", cfg.eval_episodes)
    eval_seeds = os.environ.get("BFU_EVAL_SEEDS")
    if eval_seeds:
        cfg.eval_seeds = [int(item.strip()) for item in eval_seeds.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)
