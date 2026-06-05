from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Dict, Any

from . import training_params as params


OBS_BASE_SIZE = 13
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

    log_level: str = params.LOG_LEVEL
    run_mode: str = params.RUN_MODE
    step_length: float = STEP_LENGTH
    episode_seconds: int = EPISODE_SECONDS
    evaluation_seconds: int = params.SMOKE_EVAL_SECONDS
    train_steps: int = TRAIN_SECONDS
    eval_episodes: int = EVAL_EPISODES
    eval_seeds: Any = None
    decision_interval: int = params.DECISION_INTERVAL
    control_decision_interval: int = params.CONTROL_DECISION_INTERVAL
    min_green: int = MIN_GREEN
    max_green: int = params.MAX_GREEN
    force_switch_after_max_green: bool = params.FORCE_SWITCH_AFTER_MAX_GREEN

    algorithm: str = params.ALGORITHM
    use_double_dqn: bool = params.USE_DOUBLE_DQN
    use_dueling_dqn: bool = params.USE_DUELING_DQN
    dueling_aggregation: str = params.DUELING_AGGREGATION
    checkpoint_version: int = params.CHECKPOINT_VERSION
    force_retrain_on_algorithm_change: bool = params.FORCE_RETRAIN_ON_ALGORITHM_CHANGE
    scenario_preferred_prefix: str = params.SCENARIO_PREFERRED_PREFIX
    allow_mixed_scenario_files: bool = params.ALLOW_MIXED_SCENARIO_FILES
    bfu_fast_test_default_episodes: int = params.BFU_FAST_TEST_DEFAULT_EPISODES
    bfu_fast_test_episode_seconds: int = params.BFU_FAST_TEST_EPISODE_SECONDS
    kaliningrad_validation_enabled: bool = params.KALININGRAD_VALIDATION_ENABLED

    fast_train_episodes: int = params.FAST_TRAIN_EPISODES
    fast_episode_seconds: int = params.FAST_EPISODE_SECONDS
    smoke_train_episodes: int = params.SMOKE_TRAIN_EPISODES
    smoke_episode_seconds: int = params.SMOKE_EPISODE_SECONDS
    smoke_max_train_steps: int = params.SMOKE_MAX_TRAIN_STEPS
    smoke_eval_seconds: int = params.SMOKE_EVAL_SECONDS
    smoke_eval_episodes: int = params.SMOKE_EVAL_EPISODES
    smoke_max_total_runtime_seconds: int = params.SMOKE_MAX_TOTAL_RUNTIME_SECONDS
    smoke_replay_size: int = params.SMOKE_REPLAY_SIZE
    smoke_batch_size: int = params.SMOKE_BATCH_SIZE
    smoke_max_controlled_tls: int = params.SMOKE_MAX_CONTROLLED_TLS
    smoke_skip_tls_graph: bool = params.SMOKE_SKIP_TLS_GRAPH
    smoke_eval_max_seconds: int = params.SMOKE_EVAL_MAX_SECONDS
    smoke_min_departed_for_metrics: int = params.SMOKE_MIN_DEPARTED_FOR_METRICS
    smoke_min_running_for_metrics: int = params.SMOKE_MIN_RUNNING_FOR_METRICS
    smoke_stop_eval_when_metrics_ready: bool = params.SMOKE_STOP_EVAL_WHEN_METRICS_READY
    smoke_progress_update_every_steps: int = params.SMOKE_PROGRESS_UPDATE_EVERY_STEPS
    progress_bar_style: str = params.PROGRESS_BAR_STYLE
    dev_train_episodes: int = params.DEV_TRAIN_EPISODES
    dev_episode_seconds: int = params.DEV_EPISODE_SECONDS
    experiment_train_episodes: int = params.EXPERIMENT_TRAIN_EPISODES
    experiment_episode_seconds: int = params.EXPERIMENT_EPISODE_SECONDS
    tuned_train_episodes: int = params.TUNED_TRAIN_EPISODES
    tuned_train_episode_seconds: int = params.TUNED_TRAIN_EPISODE_SECONDS
    tuned_eval_seconds: int = params.TUNED_EVAL_SECONDS
    tuned_eval_seeds: Any = None
    experiment_eval_seconds: int = params.EXPERIMENT_EVAL_SECONDS
    experiment_eval_seeds: Any = None

    device: str = params.DEVICE
    require_cuda: bool = params.REQUIRE_CUDA
    use_amp: bool = params.USE_AMP
    pin_memory: bool = params.PIN_MEMORY
    non_blocking_device_transfer: bool = params.NON_BLOCKING_DEVICE_TRANSFER
    torch_num_threads: Any = params.TORCH_NUM_THREADS
    torch_num_interop_threads: Any = params.TORCH_NUM_INTEROP_THREADS
    sumo_parallel_eval: bool = params.SUMO_PARALLEL_EVAL
    max_parallel_eval_workers: int = params.MAX_PARALLEL_EVAL_WORKERS

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
    min_switch_action_prob_during_train: float = params.MIN_SWITCH_ACTION_PROB_DURING_TRAIN
    eval_epsilon: float = params.EVAL_EPSILON
    initial_switch_bias: float = 0.02
    eval_switch_tie_break_margin: float = 0.0

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
    pedestrian_priority_max_share: float = params.PEDESTRIAN_PRIORITY_MAX_SHARE
    pedestrian_reward_normalization: float = params.PEDESTRIAN_REWARD_NORMALIZATION
    reward_car_priority_ratio_note: str = params.REWARD_CAR_PRIORITY_RATIO_NOTE
    pedestrian_reward_scope: str = params.PEDESTRIAN_REWARD_SCOPE
    strict_reward_validation: bool = params.STRICT_REWARD_VALIDATION
    reward_stuck_phase_penalty_enabled: bool = params.REWARD_STUCK_PHASE_PENALTY_ENABLED
    reward_stuck_phase_after_seconds: int = params.REWARD_STUCK_PHASE_AFTER_SECONDS
    reward_stuck_phase_penalty: float = params.REWARD_STUCK_PHASE_PENALTY
    reward_switch_penalty: float = params.REWARD_SWITCH_PENALTY
    reward_use_delta_wait: bool = params.REWARD_USE_DELTA_WAIT
    reward_use_absolute_wait: bool = params.REWARD_USE_ABSOLUTE_WAIT
    reward_use_queue: bool = params.REWARD_USE_QUEUE
    reward_use_throughput_bonus: bool = params.REWARD_USE_THROUGHPUT_BONUS
    reward_throughput_weight: float = params.REWARD_THROUGHPUT_WEIGHT
    reward_use_time_loss: bool = params.REWARD_USE_TIME_LOSS
    reward_time_loss_weight: float = params.REWARD_TIME_LOSS_WEIGHT
    reward_use_speed_diagnostic: bool = params.REWARD_USE_SPEED_DIAGNOSTIC
    reward_variant: str = params.REWARD_VARIANT
    reward_pressure_norm: float = params.REWARD_PRESSURE_NORM
    reward_queue_norm: float = params.REWARD_QUEUE_NORM
    reward_wait_norm: float = params.REWARD_WAIT_NORM
    reward_time_loss_norm: float = params.REWARD_TIME_LOSS_NORM
    w_pressure_improvement: float = params.W_PRESSURE_IMPROVEMENT
    w_queue_improvement: float = params.W_QUEUE_IMPROVEMENT
    w_wait_improvement: float = params.W_WAIT_IMPROVEMENT
    w_queue_level: float = params.W_QUEUE_LEVEL
    w_wait_level: float = params.W_WAIT_LEVEL
    w_time_loss: float = params.W_TIME_LOSS
    w_neighbor: float = params.W_NEIGHBOR
    w_switch: float = params.W_SWITCH
    w_stuck: float = params.W_STUCK
    reward_switch_penalty_enabled: bool = params.REWARD_SWITCH_PENALTY_ENABLED
    primary_objective: str = params.PRIMARY_OBJECTIVE
    objective_weights: Any = None
    primary_acceptance_metrics: Any = None
    objective_normalization: Any = None
    validation_every_episodes: int = params.VALIDATION_EVERY_EPISODES
    validation_seconds: int = params.VALIDATION_SECONDS
    validation_seeds: Any = None
    strict_rl_quality_gate: bool = params.STRICT_RL_QUALITY_GATE
    accept_min_objective_improvement_pct: float = params.ACCEPT_MIN_OBJECTIVE_IMPROVEMENT_PCT
    max_allowed_queue_regression_pct: float = params.MAX_ALLOWED_QUEUE_REGRESSION_PCT
    max_allowed_wait_regression_pct: float = params.MAX_ALLOWED_WAIT_REGRESSION_PCT
    max_allowed_throughput_regression: int = params.MAX_ALLOWED_THROUGHPUT_REGRESSION
    max_allowed_throughput_regression_pct: float = params.MAX_ALLOWED_THROUGHPUT_REGRESSION_PCT
    tuning_max_trials: int = params.TUNING_MAX_TRIALS
    quality_gate_enabled: bool = params.QUALITY_GATE_ENABLED
    quality_gate_baseline: str = params.QUALITY_GATE_BASELINE
    quality_gate_metric: str = params.QUALITY_GATE_METRIC
    quality_gate_min_improvement: float = params.QUALITY_GATE_MIN_IMPROVEMENT
    quality_gate_max_rounds: int = params.QUALITY_GATE_MAX_ROUNDS
    quality_gate_max_retrain_rounds: int = params.QUALITY_GATE_MAX_RETRAIN_ROUNDS
    quality_gate_extra_episodes: int = params.QUALITY_GATE_EXTRA_EPISODES
    save_best_checkpoint: bool = params.SAVE_BEST_CHECKPOINT
    strict_require_rl_better: bool = params.STRICT_REQUIRE_RL_BETTER
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
    pedestrian_mode: str = params.PEDESTRIAN_MODE
    require_nonzero_pedestrian_metrics: bool = params.REQUIRE_NONZERO_PEDESTRIAN_METRICS
    real_timing_file: str = params.REAL_TIMING_FILE
    use_real_timing_baseline: bool = params.USE_REAL_TIMING_BASELINE
    rl_use_real_timing_program_as_base: bool = params.RL_USE_REAL_TIMING_PROGRAM_AS_BASE
    run_real_timing_baseline: bool = params.RUN_REAL_TIMING_BASELINE
    run_native_fixed_baseline: bool = params.RUN_NATIVE_FIXED_BASELINE
    run_evaluation_in_smoke: bool = params.RUN_EVALUATION_IN_SMOKE
    run_baselines_in_smoke: bool = params.RUN_BASELINES_IN_SMOKE
    run_real_timing_in_smoke: bool = params.RUN_REAL_TIMING_IN_SMOKE
    run_native_fixed_in_smoke: bool = params.RUN_NATIVE_FIXED_IN_SMOKE
    run_validations_in_smoke: bool = params.RUN_VALIDATIONS_IN_SMOKE
    skip_baselines: bool = False
    primary_baseline: str = params.PRIMARY_BASELINE
    strict_real_timing_validation: bool = params.STRICT_REAL_TIMING_VALIDATION
    controlled_tls_mode: str = params.CONTROLLED_TLS_MODE
    active_tls_scan_seconds: int = params.ACTIVE_TLS_SCAN_SECONDS
    min_vehicle_observations_for_active_tls: int = params.MIN_VEHICLE_OBSERVATIONS_FOR_ACTIVE_TLS

    seed: int = SEED
    log_freq: int = 100

    queue_norm: float = QUEUE_NORM
    waiting_norm: float = WAIT_NORM
    time_norm: float = TIME_SINCE_SWITCH_NORM
    time_since_switch_norm: float = TIME_SINCE_SWITCH_NORM
    lane_norm: float = LANE_NORM
    obs_base_size: int = OBS_BASE_SIZE
    vehicle_norm: float = 20.0
    speed_norm: float = 15.0
    pressure_norm: float = params.REWARD_PRESSURE_NORM
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
    cfg.tuned_eval_seeds = list(params.TUNED_EVAL_SEEDS)
    cfg.experiment_eval_seeds = list(params.EXPERIMENT_EVAL_SEEDS)
    cfg.validation_seeds = list(
        params.TUNED_VALIDATION_SEEDS
        if str(params.RUN_MODE).lower() == "tuned_metrics"
        else params.VALIDATION_SEEDS
    )
    cfg.objective_weights = dict(params.OBJECTIVE_WEIGHTS)
    cfg.primary_acceptance_metrics = list(params.PRIMARY_ACCEPTANCE_METRICS)
    cfg.objective_normalization = dict(params.OBJECTIVE_NORMALIZATION)
    cfg.sumo_extra_args = list(SUMO_EXTRA_ARGS)
    _apply_env_overrides(cfg)
    cfg.set_observation_phases(DEFAULT_MAX_PHASES)
    return cfg


def _apply_env_overrides(cfg: Config) -> None:
    cfg.log_level = os.environ.get("BFU_LOG_LEVEL", cfg.log_level).strip().lower()
    cfg.run_mode = os.environ.get("BFU_RUN_MODE", cfg.run_mode).strip().lower()
    cfg.device = os.environ.get("BFU_DEVICE", cfg.device).strip().lower()
    cfg.require_cuda = _env_bool("BFU_REQUIRE_CUDA", cfg.require_cuda)
    cfg.use_amp = _env_bool("BFU_USE_AMP", cfg.use_amp)
    cfg.progress_bar_style = os.environ.get("BFU_PROGRESS_BAR_STYLE", cfg.progress_bar_style).strip().lower()
    cfg.controlled_tls_mode = os.environ.get("BFU_CONTROLLED_TLS_MODE", cfg.controlled_tls_mode).strip().lower()
    cfg.active_tls_scan_seconds = _env_int("BFU_ACTIVE_TLS_SCAN_SECONDS", cfg.active_tls_scan_seconds)
    cfg.control_decision_interval = _env_int(
        "BFU_CONTROL_DECISION_INTERVAL", cfg.control_decision_interval
    )
    cfg.decision_interval = int(cfg.control_decision_interval)
    cfg.min_green = _env_int("BFU_MIN_GREEN", cfg.min_green)
    cfg.max_green = _env_int("BFU_MAX_GREEN", cfg.max_green)
    cfg.reward_variant = os.environ.get("BFU_REWARD_VARIANT", cfg.reward_variant).strip()
    cfg.initial_switch_bias = _env_float("BFU_INITIAL_SWITCH_BIAS", cfg.initial_switch_bias)
    cfg.eval_switch_tie_break_margin = _env_float(
        "BFU_EVAL_SWITCH_TIE_BREAK_MARGIN",
        cfg.eval_switch_tie_break_margin,
    )
    cfg.lr = _env_float("BFU_LR", cfg.lr)
    cfg.gamma = _env_float("BFU_GAMMA", cfg.gamma)
    cfg.w_switch = _env_float("BFU_W_SWITCH", cfg.w_switch)
    cfg.w_stuck = _env_float("BFU_W_STUCK", cfg.w_stuck)
    cfg.reward_switch_penalty = _env_float("BFU_REWARD_SWITCH_PENALTY", cfg.reward_switch_penalty)
    cfg.reward_stuck_phase_penalty = _env_float(
        "BFU_REWARD_STUCK_PHASE_PENALTY", cfg.reward_stuck_phase_penalty
    )
    cfg.tuning_max_trials = _env_int("BFU_TUNING_MAX_TRIALS", cfg.tuning_max_trials)
    cfg.validation_seconds = _env_int("BFU_VALIDATION_SECONDS", cfg.validation_seconds)
    validation_seeds = os.environ.get("BFU_VALIDATION_SEEDS")
    if validation_seeds:
        cfg.validation_seeds = [
            int(item.strip()) for item in validation_seeds.split(",") if item.strip()
        ]
    cfg.sumo_parallel_eval = _env_bool("BFU_SUMO_PARALLEL_EVAL", cfg.sumo_parallel_eval)
    cfg.max_parallel_eval_workers = _env_int(
        "BFU_MAX_PARALLEL_EVAL_WORKERS", cfg.max_parallel_eval_workers
    )
    cfg.quality_gate_enabled = _env_bool("BFU_QUALITY_GATE_ENABLED", cfg.quality_gate_enabled)
    cfg.quality_gate_max_rounds = _env_int("BFU_QUALITY_GATE_MAX_ROUNDS", cfg.quality_gate_max_rounds)
    cfg.quality_gate_max_retrain_rounds = _env_int(
        "BFU_QUALITY_GATE_MAX_RETRAIN_ROUNDS", cfg.quality_gate_max_retrain_rounds
    )
    cfg.quality_gate_extra_episodes = _env_int(
        "BFU_QUALITY_GATE_EXTRA_EPISODES", cfg.quality_gate_extra_episodes
    )
    cfg.strict_require_rl_better = _env_bool(
        "BFU_STRICT_REQUIRE_RL_BETTER", cfg.strict_require_rl_better
    )

    if _env_bool("BFU_FAST_TEST", False):
        cfg.run_mode = "fast"
        cfg.episode_seconds = int(cfg.bfu_fast_test_episode_seconds)
        cfg.train_steps = int(cfg.bfu_fast_test_default_episodes) * int(cfg.episode_seconds)
        cfg.eval_episodes = 1
        cfg.eval_seeds = [cfg.seed]
        cfg.debug_scenario = False
        cfg.quality_gate_enabled = False
        cfg.run_real_timing_baseline = False
        cfg.sumo_parallel_eval = False
        cfg.auto_generate_pedestrians_if_missing = False
    else:
        _apply_run_mode_defaults(cfg, cfg.run_mode)

    cfg.force_retrain = _env_bool("BFU_FORCE_RETRAIN", cfg.force_retrain)
    cfg.strict_dev_validation = _env_bool("BFU_STRICT_DEV_VALIDATION", cfg.strict_dev_validation)
    if _env_bool("BFU_DISABLE_QUALITY_GATE", False):
        cfg.quality_gate_enabled = False
    cfg.run_real_timing_baseline = _env_bool(
        "BFU_RUN_REAL_TIMING_BASELINE", cfg.run_real_timing_baseline
    )
    cfg.run_native_fixed_baseline = _env_bool(
        "BFU_RUN_NATIVE_FIXED_BASELINE", cfg.run_native_fixed_baseline
    )
    cfg.skip_baselines = _env_bool("BFU_SKIP_BASELINES", cfg.skip_baselines)
    cfg.run_evaluation_in_smoke = _env_bool(
        "BFU_RUN_EVALUATION_IN_SMOKE", cfg.run_evaluation_in_smoke
    )
    cfg.run_baselines_in_smoke = _env_bool("BFU_RUN_BASELINES_IN_SMOKE", cfg.run_baselines_in_smoke)
    cfg.run_real_timing_in_smoke = _env_bool(
        "BFU_RUN_REAL_TIMING_IN_SMOKE", cfg.run_real_timing_in_smoke
    )
    cfg.run_native_fixed_in_smoke = _env_bool(
        "BFU_RUN_NATIVE_FIXED_IN_SMOKE", cfg.run_native_fixed_in_smoke
    )
    cfg.run_validations_in_smoke = _env_bool(
        "BFU_RUN_VALIDATIONS_IN_SMOKE", cfg.run_validations_in_smoke
    )
    cfg.auto_generate_pedestrians_if_missing = _env_bool(
        "BFU_AUTO_GENERATE_PEDESTRIANS_IF_MISSING",
        cfg.auto_generate_pedestrians_if_missing,
    )
    if "BFU_PEDESTRIAN_MODE" in os.environ:
        cfg.pedestrian_mode = os.environ["BFU_PEDESTRIAN_MODE"].strip().lower()
    cfg.scenario_preferred_prefix = os.environ.get(
        "BFU_SCENARIO_PREFERRED_PREFIX", cfg.scenario_preferred_prefix
    )
    cfg.allow_mixed_scenario_files = _env_bool(
        "BFU_ALLOW_MIXED_SCENARIO_FILES", cfg.allow_mixed_scenario_files
    )
    cfg.episode_seconds = _env_int("BFU_EPISODE_SECONDS", cfg.episode_seconds)
    cfg.episode_seconds = _env_int("BFU_TRAIN_EPISODE_SECONDS", cfg.episode_seconds)
    cfg.train_steps = _env_int("BFU_TRAIN_SECONDS", cfg.train_steps)
    train_episodes = os.environ.get("BFU_TRAIN_EPISODES")
    if train_episodes is not None and "BFU_TRAIN_SECONDS" not in os.environ:
        cfg.train_steps = max(1, int(train_episodes)) * int(cfg.episode_seconds)
    cfg.eval_episodes = _env_int("BFU_EVAL_EPISODES", cfg.eval_episodes)
    eval_seeds = os.environ.get("BFU_EVAL_SEEDS")
    if eval_seeds:
        cfg.eval_seeds = [int(item.strip()) for item in eval_seeds.split(",") if item.strip()]
    if "BFU_EVAL_SECONDS" in os.environ:
        cfg.evaluation_seconds = _env_int("BFU_EVAL_SECONDS", getattr(cfg, "evaluation_seconds", cfg.episode_seconds))

    if cfg.log_level in {"verbose", "debug"}:
        cfg.debug_scenario = cfg.log_level == "debug"
    if cfg.run_mode == "smoke":
        cfg.skip_baselines = not bool(cfg.run_baselines_in_smoke)
        cfg.run_real_timing_baseline = bool(cfg.run_baselines_in_smoke) and bool(cfg.run_real_timing_in_smoke)
        cfg.run_native_fixed_baseline = bool(cfg.run_baselines_in_smoke) and bool(cfg.run_native_fixed_in_smoke)
        cfg.quality_gate_enabled = False
        cfg.strict_dev_validation = False


def _apply_run_mode_defaults(cfg: Config, run_mode: str) -> None:
    mode = str(run_mode or "dev").strip().lower()
    if mode == "smoke":
        cfg.run_mode = "smoke"
        cfg.episode_seconds = int(cfg.smoke_episode_seconds)
        cfg.evaluation_seconds = int(cfg.smoke_eval_seconds)
        cfg.train_steps = int(cfg.smoke_max_train_steps)
        cfg.eval_episodes = int(cfg.smoke_eval_episodes)
        cfg.eval_seeds = [cfg.seed]
        cfg.replay_size = int(cfg.smoke_replay_size)
        cfg.batch_size = int(cfg.smoke_batch_size)
        cfg.start_learning_after = min(int(cfg.start_learning_after), int(cfg.smoke_batch_size))
        cfg.log_freq = max(1, int(cfg.smoke_max_train_steps) // 4)
        cfg.train_freq = 4
        cfg.target_update_freq = 8
        cfg.device = "cuda"
        cfg.require_cuda = True
        cfg.use_gui = False
        cfg.sumo_parallel_eval = False
        cfg.smoke_max_controlled_tls = int(getattr(cfg, "smoke_max_controlled_tls", 20))
        cfg.smoke_skip_tls_graph = bool(getattr(cfg, "smoke_skip_tls_graph", True))
        cfg.quality_gate_enabled = False
        cfg.strict_dev_validation = False
        cfg.strict_require_rl_better = False
        cfg.require_rl_phase_switches = False
        cfg.require_rl_fixed_metric_difference = False
        cfg.require_nonzero_pedestrian_metrics = False
        cfg.auto_generate_pedestrians_if_missing = False
        cfg.rl_use_real_timing_program_as_base = False
        cfg.run_real_timing_baseline = bool(getattr(cfg, "run_real_timing_in_smoke", False))
        cfg.run_native_fixed_baseline = bool(getattr(cfg, "run_native_fixed_in_smoke", True))
        cfg.skip_baselines = not bool(getattr(cfg, "run_baselines_in_smoke", False))
        cfg.force_retrain = True
    elif mode == "fast":
        cfg.run_mode = "fast"
        cfg.episode_seconds = int(cfg.fast_episode_seconds)
        cfg.evaluation_seconds = int(cfg.fast_episode_seconds)
        cfg.train_steps = int(cfg.fast_train_episodes) * int(cfg.episode_seconds)
        cfg.eval_episodes = 1
        cfg.eval_seeds = [cfg.seed]
        cfg.quality_gate_enabled = False
        cfg.run_real_timing_baseline = False
        cfg.sumo_parallel_eval = False
        cfg.auto_generate_pedestrians_if_missing = False
    elif mode == "tuned_metrics":
        cfg.run_mode = "tuned_metrics"
        cfg.episode_seconds = int(cfg.tuned_train_episode_seconds)
        cfg.train_steps = int(cfg.tuned_train_episodes) * int(cfg.episode_seconds)
        cfg.evaluation_seconds = int(cfg.tuned_eval_seconds)
        cfg.eval_episodes = 1
        cfg.eval_seeds = list(cfg.tuned_eval_seeds or params.TUNED_EVAL_SEEDS)
        cfg.validation_seeds = list(params.TUNED_VALIDATION_SEEDS)
        cfg.controlled_tls_mode = str(params.TUNED_CONTROLLED_TLS)
        cfg.quality_gate_enabled = False
        cfg.run_real_timing_baseline = False
        cfg.run_native_fixed_baseline = True
        cfg.sumo_parallel_eval = False
        cfg.auto_generate_pedestrians_if_missing = True
    elif mode == "experiment":
        cfg.run_mode = "experiment"
        cfg.episode_seconds = int(cfg.experiment_episode_seconds)
        cfg.train_steps = int(cfg.experiment_train_episodes) * int(cfg.episode_seconds)
        cfg.evaluation_seconds = int(cfg.experiment_eval_seconds)
        cfg.eval_seeds = list(cfg.experiment_eval_seeds or params.EXPERIMENT_EVAL_SEEDS)
    else:
        cfg.run_mode = "dev"
        cfg.episode_seconds = int(cfg.dev_episode_seconds)
        cfg.evaluation_seconds = int(cfg.dev_episode_seconds)
        cfg.train_steps = int(cfg.dev_train_episodes) * int(cfg.episode_seconds)


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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)
