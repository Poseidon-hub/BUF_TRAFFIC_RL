"""
Ключевые параметры обучения и evaluation.

Этот файл рассчитан на обычную настройку эксперимента без изменения кода.
Для разработки значения намеренно короткие: запуск `python main.py` не должен
автоматически уходить в многочасовую SUMO-симуляцию.
"""

import os

# -------------------------
# Основные параметры запуска
# -------------------------

SEED = 42
# Базовый seed для SUMO, numpy, random и torch. Меняйте для проверки устойчивости
# результата к случайности.

USE_GUI = False
# True запускает sumo-gui, False запускает консольный sumo. Для обучения обычно
# нужен False, потому что GUI заметно замедляет симуляцию.

FORCE_RETRAIN = False
# True заставляет игнорировать существующий checkpoint и обучать DQN заново.
# Полезно после изменения reward, observation или параметров модели.

AUTO_RETRAIN_ON_SCENARIO_CHANGE = True

LOG_LEVEL = os.getenv("BFU_LOG_LEVEL", "compact")
# Уровень консольного вывода: "silent", "compact", "verbose" или "debug".
# compact оставляет только сводку, progress bar и итоговые сравнения.

RUN_MODE = os.getenv("BFU_RUN_MODE", "smoke")
# Режим запуска:
# "smoke" — сверхбыстрая техническая проверка;
# "fast" — быстрая проверка для разработки;
# "dev" — обычная разработка на укороченных эпизодах;
# "experiment" — более длинный запуск для дипломного эксперимента.

SMOKE_TRAIN_EPISODES = 1
SMOKE_EPISODE_SECONDS = 5
SMOKE_MAX_TRAIN_STEPS = 10
SMOKE_EVAL_SECONDS = 60
SMOKE_EVAL_EPISODES = 1
SMOKE_MAX_TOTAL_RUNTIME_SECONDS = 15
SMOKE_REPLAY_SIZE = 256
SMOKE_BATCH_SIZE = 4
SMOKE_MAX_CONTROLLED_TLS = 20
SMOKE_SKIP_TLS_GRAPH = True
SMOKE_EVAL_MAX_SECONDS = 120
SMOKE_MIN_DEPARTED_FOR_METRICS = 10
SMOKE_MIN_RUNNING_FOR_METRICS = 5
SMOKE_STOP_EVAL_WHEN_METRICS_READY = True
SMOKE_PROGRESS_UPDATE_EVERY_STEPS = 1
PROGRESS_BAR_STYLE = os.getenv("BFU_PROGRESS_BAR_STYLE", "single_line")

FAST_TRAIN_EPISODES = 1
# Количество episode в fast-режиме.

FAST_EPISODE_SECONDS = 60
# Длина episode в fast-режиме.

DEV_TRAIN_EPISODES = 20
# Количество episode в dev-режиме.

DEV_EPISODE_SECONDS = 900
# Длина episode в dev-режиме. Для большой карты 300 секунд часто недостаточно.

DEV_METRICS_TRAIN_EPISODES = 3
DEV_METRICS_TRAIN_EPISODE_SECONDS = 300
DEV_METRICS_EVAL_SECONDS = 600
DEV_METRICS_MAX_TRAIN_STEPS = 600

TUNED_TRAIN_EPISODES = 30
TUNED_TRAIN_EPISODE_SECONDS = 600
TUNED_EVAL_SECONDS = 1200
TUNED_CONTROLLED_TLS = "active_only"
TUNED_EVAL_SEEDS = [42, 43, 44]

EXPERIMENT_TRAIN_EPISODES = 100
# Количество episode в experiment-режиме.

EXPERIMENT_EPISODE_SECONDS = 900
# Длина episode для полноценного эксперимента.

EXPERIMENT_EVAL_SECONDS = 3600
EXPERIMENT_EVAL_SEEDS = [42, 43, 44, 45, 46]

SCENARIO_PREFERRED_PREFIX = "kaliningrad"
# Приоритетный префикс SUMO-сценария. Если в scenario/ есть kaliningrad.net.xml,
# он выбирается раньше старых тестовых карт и пользовательских scenario.sumocfg.

ALLOW_MIXED_SCENARIO_FILES = False
# Если False, выбранная карта kaliningrad.net.xml не смешивается с route-файлами
# от другого сценария, например routes.rou.xml от старой тестовой карты.

BFU_FAST_TEST_DEFAULT_EPISODES = 1
# Число коротких episode для BFU_FAST_TEST, если BFU_TRAIN_SECONDS не задан явно.

BFU_FAST_TEST_EPISODE_SECONDS = 60
# Длина episode для BFU_FAST_TEST по умолчанию. Переменная BFU_EPISODE_SECONDS
# может переопределить это значение из окружения.

KALININGRAD_VALIDATION_ENABLED = True
# Включает validation-скрипты, рассчитанные на наличие файлов kaliningrad.* в scenario/.
# True автоматически переобучает модель, если checkpoint не совпадает с текущей
# картой, TLS, числом фаз, observation/action space или сигнатурой сценария.


# -------------------------
# SUMO / episode
# -------------------------

STEP_LENGTH = 1.0
# Шаг SUMO в секундах. Обычно 1.0 достаточно для traffic-light RL.

if RUN_MODE == "smoke":
    EPISODE_SECONDS = SMOKE_EPISODE_SECONDS
    TRAIN_EPISODES = SMOKE_TRAIN_EPISODES
elif RUN_MODE == "fast":
    EPISODE_SECONDS = FAST_EPISODE_SECONDS
    TRAIN_EPISODES = FAST_TRAIN_EPISODES
elif RUN_MODE == "tuned_metrics":
    EPISODE_SECONDS = TUNED_TRAIN_EPISODE_SECONDS
    TRAIN_EPISODES = TUNED_TRAIN_EPISODES
elif RUN_MODE == "experiment":
    EPISODE_SECONDS = EXPERIMENT_EPISODE_SECONDS
    TRAIN_EPISODES = EXPERIMENT_TRAIN_EPISODES
else:
    EPISODE_SECONDS = DEV_EPISODE_SECONDS
    TRAIN_EPISODES = DEV_TRAIN_EPISODES
# Длительность и число episode выбираются по RUN_MODE. При необходимости их можно
# переопределить переменными BFU_EPISODE_SECONDS и BFU_TRAIN_EPISODES.

TRAIN_SECONDS = EPISODE_SECONDS * TRAIN_EPISODES
# Общий лимит шагов обучения для текущей реализации train-loop.

EVAL_EPISODES = 1
# Сколько episode evaluation запускать на каждый seed.

EVAL_SEEDS = [SEED]
# Список seed для evaluation. Для MVP обычно один seed; список оставлен для
# дальнейшего усреднения по нескольким запускам.

CONTROL_DECISION_INTERVAL = 5
DECISION_INTERVAL = CONTROL_DECISION_INTERVAL
# Как часто агент принимает решение, в шагах SUMO. Текущая среда вызывает action
# каждый step; значение 1 сохраняет это поведение.

MIN_GREEN = 10
# Минимальное время между переключениями фаз TLS. Слишком маленькое значение
# приводит к частому дерганию фаз, слишком большое блокирует действия агента.

MAX_GREEN = 90
FORCE_SWITCH_AFTER_MAX_GREEN = False

USE_SUMO_YELLOW_PHASES = True
# True означает, что yellow phases берутся из существующей SUMO TLS-программы.
# Агент не создает собственные yellow phases.

YELLOW_TIME = 3
# Справочное значение для сценариев, где yellow time задается при генерации сети.
# В текущем коде новые TLS-программы не создаются.


# -------------------------
# DQN
# -------------------------

DEVICE = "cuda"
# Smoke/dev training in this project is CUDA-only. SUMO/TraCI remain CPU-bound.

REQUIRE_CUDA = True
# If True, main.py exits before training when torch.cuda.is_available() is False.

USE_AMP = True
# Mixed precision используется только на CUDA. При проблемах обучение автоматически остается в fp32.

PIN_MEMORY = True
# Зарезервировано для ускоренной передачи batch на GPU при наличии pinned memory.

NON_BLOCKING_DEVICE_TRANSFER = True
# Использовать non_blocking=True при переносе tensor на GPU, где это применимо.

TORCH_NUM_THREADS = "auto"
# При auto PyTorch получает min(os.cpu_count(), 8), чтобы не перегружать SUMO.

TORCH_NUM_INTEROP_THREADS = "auto"
# Количество interop threads PyTorch; auto выбирает умеренное значение.

SUMO_PARALLEL_EVAL = False
# Baseline/evaluation могут запускаться независимо, но при нестабильности TraCI используется последовательный fallback.

MAX_PARALLEL_EVAL_WORKERS = 2
# Верхняя граница параллельных SUMO evaluation процессов.

ALGORITHM = "double_dueling_dqn"
# Main training algorithm.
# Supported values:
# "dqn" - vanilla DQN;
# "double_dqn" - Double DQN target;
# "dueling_dqn" - Dueling Q-network;
# "double_dueling_dqn" - Double DQN target + Dueling Q-network.

USE_DOUBLE_DQN = True
# If True, next action is selected by the online network and evaluated by the target network.

USE_DUELING_DQN = True
# If True, use a shared feature extractor with separate value and advantage streams.

DUELING_AGGREGATION = "mean"
# Q(s,a) = V(s) + A(s,a) - mean(A(s,*)).

MODEL_HIDDEN_DIM = 128
MODEL_NUM_HIDDEN_LAYERS = 2
MODEL_ACTIVATION = "relu"

CHECKPOINT_VERSION = 2
# Incremented because the model architecture and checkpoint metadata changed.

FORCE_RETRAIN_ON_ALGORITHM_CHANGE = True
# Do not silently reuse checkpoints trained with a different algorithm or architecture.

GAMMA = 0.99
# Discount factor: насколько сильно будущая награда влияет на Q-value.

LEARNING_RATE = 0.0003
# Скорость обучения Adam optimizer.

BATCH_SIZE = SMOKE_BATCH_SIZE if RUN_MODE == "smoke" else 128
# Размер mini-batch из replay buffer.

REPLAY_SIZE = SMOKE_REPLAY_SIZE if RUN_MODE == "smoke" else 50_000
# Максимальное число transitions в replay buffer.

START_LEARNING_AFTER = 4 if RUN_MODE == "smoke" else 256
# Сколько transitions накопить перед первыми gradient updates.

TRAIN_FREQ = 4 if RUN_MODE == "smoke" else 1
# Делать update сети каждые N environment steps.

TARGET_UPDATE_FREQ = 8 if RUN_MODE == "smoke" else 1000
# Как часто копировать веса online-сети в target-сеть.

GRAD_CLIP_NORM = 10.0
# Ограничение нормы градиента для стабильности DQN.

HIDDEN_DIM = MODEL_HIDDEN_DIM
# Размер скрытых слоев QNetwork.

NUM_HIDDEN_LAYERS = MODEL_NUM_HIDDEN_LAYERS
# Число скрытых Linear+ReLU блоков в QNetwork.


# -------------------------
# Exploration
# -------------------------

EPS_START = 1.0
# Начальный epsilon для epsilon-greedy во время обучения.

EPS_END = 0.05
# Минимальный epsilon после decay.

EPS_DECAY_STEPS = 30000 if RUN_MODE == "tuned_metrics" else 10000
# За сколько train steps epsilon спадает от EPS_START к EPS_END.

MIN_SWITCH_ACTION_PROB_DURING_TRAIN = 0.05
# Во время обучения сохраняет небольшой шанс action=1, чтобы policy не выродилась в always-hold.

EVAL_EPSILON = 0.0
# Evaluation выполняется жадно, без случайного улучшения метрик.


# -------------------------
# Reward
# -------------------------

REWARD_ALPHA_QUEUE = 0.2
# Вес queue в локальном автомобильном reward.

REWARD_VARIANT = "pressure_wait_time_loss"
REWARD_PRESSURE_NORM = 20.0
REWARD_QUEUE_NORM = 20.0
REWARD_WAIT_NORM = 300.0
REWARD_TIME_LOSS_NORM = 300.0

W_PRESSURE_IMPROVEMENT = 1.0
W_QUEUE_IMPROVEMENT = 0.8
W_WAIT_IMPROVEMENT = 0.8
W_QUEUE_LEVEL = 0.3
W_WAIT_LEVEL = 0.3
W_TIME_LOSS = 0.4
W_NEIGHBOR = 0.2
W_SWITCH = 0.02
W_STUCK = 0.05

REWARD_BETA_NEIGHBOR = 0.3
# Вес mean neighbor penalty в reward.

REWARD_USE_NEIGHBORS = True
# True включает соседние TLS в reward, если граф соседей удалось построить.

REWARD_NORMALIZE = False
# Зарезервировано для будущего нормирования reward.

PEDESTRIAN_REWARD_ENABLED = True
# Пешеходные метрики сейчас не входят в reward. Включайте только после отдельной
# проверки формулы reward.

PEDESTRIAN_WAITING_PENALTY = 0.0
# Будущий штраф за ожидание пешеходов. При PEDESTRIAN_REWARD_ENABLED=False не влияет.

REWARD_USE_PEDESTRIANS = True
# If True, reward includes a small pedestrian penalty.

REWARD_VEHICLE_QUEUE_WEIGHT = 1.0
# Vehicle queue remains a primary optimization target.

REWARD_VEHICLE_WAIT_WEIGHT = 1.0
# Vehicle waiting time remains a primary optimization target.

REWARD_NEIGHBOR_WEIGHT = 0.2
# Neighbor vehicle penalty weight.

REWARD_PEDESTRIAN_WAIT_WEIGHT = 0.10
# Smaller than vehicle weights so vehicle terms dominate the reward.

W_PEDESTRIAN_WAIT = 0.10
# Alias used by runtime reward diagnostics.

REWARD_PEDESTRIAN_RUNNING_WEIGHT = 0.02
# Small penalty for pedestrians still active in the network.

REWARD_PEDESTRIAN_BLOCKED_WEIGHT = 0.05
# Small penalty for pedestrians with positive waiting time.

W_PEDESTRIAN_QUEUE = 0.05
# Alias for pedestrian waiting-count term.

PEDESTRIAN_PRIORITY_MAX_SHARE = 0.20
# Cap pedestrian reward contribution at 20 percent of total reward scale.

PEDESTRIAN_REWARD_NORMALIZATION = 30.0
# Normalizes pedestrian penalty so it does not dominate vehicle terms.

REWARD_CAR_PRIORITY_RATIO_NOTE = "Vehicle terms should dominate pedestrian terms."

PEDESTRIAN_REWARD_SCOPE = "global"
# Supported values:
# "global" - same pedestrian penalty is added to all TLS agents;
# "local" - reserved for future TLS/crossing matching.

STRICT_REWARD_VALIDATION = True
# Validation fails if the pedestrian component dominates the reward.

REWARD_STUCK_PHASE_PENALTY_ENABLED = True
# Включает маленький штраф за слишком долгое удержание одной фазы при наличии очередей.

REWARD_STUCK_PHASE_AFTER_SECONDS = 60
# После скольких секунд с последнего переключения считать фазу потенциально stuck.

REWARD_STUCK_PHASE_PENALTY = W_STUCK
# Малый штраф за stuck phase. Значение намеренно небольшое, чтобы не провоцировать хаос.

REWARD_SWITCH_PENALTY_ENABLED = True

REWARD_SWITCH_PENALTY = W_SWITCH
# Малый штраф за фактическое переключение, чтобы агент не дергал фазы каждую секунду.

REWARD_USE_DELTA_WAIT = True
# Учитывать изменение накопленного ожидания относительно прошлого шага.

REWARD_USE_ABSOLUTE_WAIT = True
# Учитывать абсолютное среднее ожидание автомобилей.

REWARD_USE_QUEUE = True
# Учитывать очередь автомобилей, нормированную по числу controlled lanes.

REWARD_USE_THROUGHPUT_BONUS = True
# Добавлять небольшой бонус за прибытия автомобилей, если они есть.

REWARD_THROUGHPUT_WEIGHT = 0.1
# Вес бонуса throughput; он меньше основных штрафов за очереди и ожидание.

REWARD_USE_TIME_LOSS = True
REWARD_TIME_LOSS_WEIGHT = W_TIME_LOSS
REWARD_USE_SPEED_DIAGNOSTIC = True


# -------------------------
# Training quality gates
# -------------------------

PRIMARY_OBJECTIVE = "weighted_mobility_score"
OBJECTIVE_EPS = 1e-6
OBJECTIVE_WEIGHTS = {
    "avg_queue": 1.0,
    "avg_waiting_time": 1.0,
    "avg_time_loss": 1.25,
    "normalized_time_loss_per_departed": 0.75,
    "throughput": -1.0,
}
PRIMARY_ACCEPTANCE_METRICS = [
    "avg_queue",
    "avg_waiting_time",
    "avg_time_loss",
    "total_time_loss",
    "throughput",
]
OBJECTIVE_NORMALIZATION = {
    "avg_queue": 1.0,
    "avg_waiting_time": 30.0,
    "avg_time_loss": 120.0,
    "normalized_time_loss_per_departed": 300.0,
    "throughput": 50.0,
}

VALIDATION_EVERY_EPISODES = 5
VALIDATION_SECONDS = 600
VALIDATION_SEEDS = [42]
TUNED_VALIDATION_SEEDS = [42, 43]

STRICT_RL_QUALITY_GATE = True
ACCEPT_MIN_OBJECTIVE_IMPROVEMENT_PCT = 1.0
MAX_ALLOWED_QUEUE_REGRESSION_PCT = 10.0
MAX_ALLOWED_WAIT_REGRESSION_PCT = 10.0
MAX_ALLOWED_THROUGHPUT_REGRESSION = 5
MAX_ALLOWED_THROUGHPUT_REGRESSION_PCT = 10.0

TUNING_MAX_TRIALS = 12

QUALITY_GATE_ENABLED = False
# После обучения делает короткую validation evaluation и сравнивает RL с baseline.

QUALITY_GATE_BASELINE = "native_fixed"
# Baseline для quality gate: "native_fixed" или "real_timing".

QUALITY_GATE_METRIC = "total_reward"
# Метрика quality gate. Для total_reward большее значение считается лучше.

QUALITY_GATE_MIN_IMPROVEMENT = 0.0
# Минимальный требуемый выигрыш RL относительно baseline.

QUALITY_GATE_MAX_ROUNDS = 1
# Максимум дополнительных раундов обучения, если RL хуже baseline.

QUALITY_GATE_MAX_RETRAIN_ROUNDS = 1
# Совместимость со старыми проверками; используется как alias QUALITY_GATE_MAX_ROUNDS.

QUALITY_GATE_EXTRA_EPISODES = 1
# Сколько дополнительных episode добавить за одну попытку quality gate.

SAVE_BEST_CHECKPOINT = True
# Сохранять лучший checkpoint по validation metric в checkpoints/best_dqn.pt.

STRICT_REQUIRE_RL_BETTER = False
# Если True, запуск завершается ошибкой, когда RL не превысил baseline после quality gate.

MIN_RL_PHASE_SET_COUNT = 1
# Минимальное число setPhase в RL evaluation для dev-сценария.

STRICT_DEV_VALIDATION = False if RUN_MODE == "smoke" else True
# В dev-режиме main.py завершается ошибкой, если RL не управляет TLS или нет pedestrians.

REQUIRE_RL_PHASE_SWITCHES = True
# Требовать, чтобы RL evaluation вызвал setPhase хотя бы один раз.

REQUIRE_RL_FIXED_METRIC_DIFFERENCE = True
# Требовать, чтобы RL и fixed метрики не были полностью одинаковыми.

METRIC_DIFF_TOLERANCE = 1e-6
# Допуск при сравнении RL/fixed метрик.

AUTO_GENERATE_PEDESTRIANS_IF_MISSING = False if RUN_MODE == "smoke" else True
# Если в scenario нет person/personFlow/walk, создать dev pedestrian demand.

PEDESTRIAN_DEMAND_COUNT = 30
# Сколько pedestrians создать для dev-проверки.

PEDESTRIAN_DEMAND_BEGIN = 0
# Начало генерации pedestrian demand.

PEDESTRIAN_DEMAND_END = 300
# Конец генерации pedestrian demand.

PEDESTRIAN_DEMAND_PREFIX = "auto_ped"
# Префикс deterministic person ids.

PEDESTRIAN_MODE = "crossing_focused"
# random_walks | crossing_focused | existing_persons.

REQUIRE_NONZERO_PEDESTRIAN_METRICS = True
# В dev-режиме требовать pedestrian_departed > 0.


# -------------------------
# Реальные тайминги светофоров
# -------------------------

REAL_TIMING_FILE = "tls.add.xml"
# Имя файла с реальными/заданными программами светофоров в формате SUMO additional.
# Обычно лежит в scenario/ и содержит теги <tlLogic>.

USE_REAL_TIMING_BASELINE = True
# Если True, программа использует внешний timing profile как отдельный baseline.

RL_USE_REAL_TIMING_PROGRAM_AS_BASE = False if RUN_MODE == "smoke" else True
# Если True, RL evaluation загружает ту же TLS-программу фаз, что и real_timing baseline,
# но действия модели всё равно применяются через TraCI setPhase.

RUN_EVALUATION_IN_SMOKE = True
RUN_BASELINES_IN_SMOKE = True
RUN_REAL_TIMING_IN_SMOKE = False
RUN_NATIVE_FIXED_IN_SMOKE = True
RUN_VALIDATIONS_IN_SMOKE = False

RUN_REAL_TIMING_BASELINE = False if RUN_MODE == "smoke" else True
# Запускать ли baseline с программами светофоров из REAL_TIMING_FILE.

RUN_NATIVE_FIXED_BASELINE = False if RUN_MODE == "smoke" else True
# Запускать ли baseline с исходными программами светофоров из net.xml без REAL_TIMING_FILE.

PRIMARY_BASELINE = "real_timing"
# Основной baseline для сравнения. Возможные значения: "real_timing", "fixed_native".

STRICT_REAL_TIMING_VALIDATION = False
# Если True, ошибки в tls.add.xml или несовпадение TLS считаются критическими при validation.
# По умолчанию False, чтобы старая тестовая tls.add.xml не ломала запуск Калининграда до подключения реальных таймингов.


# -------------------------
# Paths
# -------------------------

CONTROLLED_TLS_MODE = "active_only"
ACTIVE_TLS_SCAN_SECONDS = 300
MIN_VEHICLE_OBSERVATIONS_FOR_ACTIVE_TLS = 1

SCENARIO_DIR = "scenario"
# Папка с SUMO-сценарием: *.sumocfg или *.net.xml + route/additional XML.

CHECKPOINT_DIR = "checkpoints"
# Папка для весов модели и metadata.

CHECKPOINT_PATH = "checkpoints/dqn.pt"
# Основной файл весов DQN.

CHECKPOINT_META_PATH = "checkpoints/dqn_meta.json"
# Metadata checkpoint: сигнатура сценария, TLS ids, размеры observation/action.

LOGS_DIR = "logs"
# Папка для train/eval/comparison logs.


# -------------------------
# Logging/debug
# -------------------------

SAVE_TRAIN_LOGS = True
# Сохранять logs/train_metrics.csv и logs/train_metrics.jsonl.

SAVE_EVAL_LOGS = True
# Сохранять logs/eval_rl.* и logs/eval_fixed.*.

PRINT_ACTION_DIAGNOSTICS = True
# Печатать диагностику действий RL после evaluation.

PRINT_SCENARIO_INFO = True
# Печатать найденный sumocfg/net/routes/TLS при запуске main.py.

DEBUG_SCENARIO = False
# Печатать дополнительные сведения, например граф соседей TLS.


# -------------------------
# Observation normalization
# -------------------------

QUEUE_NORM = 20.0
# Делитель для нормализации очереди в observation.

WAIT_NORM = 300.0
# Делитель для нормализации waiting time в observation.

TIME_SINCE_SWITCH_NORM = 120.0
# Делитель для нормализации времени с последнего переключения.

LANE_NORM = 16.0
# Делитель для нормализации числа controlled vehicle lanes.


# -------------------------
# SUMO extra args
# -------------------------

SUMO_EXTRA_ARGS = [
    "--no-step-log",
    "true",
    "--quit-on-end",
    "true",
    "--time-to-teleport",
    "-1",
]
# Аргументы, добавляемые к SUMO при запуске через sumocfg.
