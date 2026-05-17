"""
Ключевые параметры обучения и evaluation.

Этот файл рассчитан на обычную настройку эксперимента без изменения кода.
Для разработки значения намеренно короткие: запуск `python main.py` не должен
автоматически уходить в многочасовую SUMO-симуляцию.
"""

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
# True автоматически переобучает модель, если checkpoint не совпадает с текущей
# картой, TLS, числом фаз, observation/action space или сигнатурой сценария.


# -------------------------
# SUMO / episode
# -------------------------

STEP_LENGTH = 1.0
# Шаг SUMO в секундах. Обычно 1.0 достаточно для traffic-light RL.

EPISODE_SECONDS = 300
# Длительность одного episode в секундах SUMO. Для полноценного эксперимента
# можно увеличить до 3600.

TRAIN_EPISODES = 10
# Сколько episode обучения запускать, если checkpoint отсутствует или включен
# FORCE_RETRAIN. Чем больше значение, тем дольше запуск и потенциально лучше policy.

TRAIN_SECONDS = EPISODE_SECONDS * TRAIN_EPISODES
# Общий лимит шагов обучения для текущей реализации train-loop.

EVAL_EPISODES = 1
# Сколько episode evaluation запускать на каждый seed.

EVAL_SEEDS = [SEED]
# Список seed для evaluation. Для MVP обычно один seed; список оставлен для
# дальнейшего усреднения по нескольким запускам.

DECISION_INTERVAL = 1
# Как часто агент принимает решение, в шагах SUMO. Текущая среда вызывает action
# каждый step; значение 1 сохраняет это поведение.

MIN_GREEN = 5
# Минимальное время между переключениями фаз TLS. Слишком маленькое значение
# приводит к частому дерганию фаз, слишком большое блокирует действия агента.

USE_SUMO_YELLOW_PHASES = True
# True означает, что yellow phases берутся из существующей SUMO TLS-программы.
# Агент не создает собственные yellow phases.

YELLOW_TIME = 3
# Справочное значение для сценариев, где yellow time задается при генерации сети.
# В текущем коде новые TLS-программы не создаются.


# -------------------------
# DQN
# -------------------------

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

LEARNING_RATE = 0.0005
# Скорость обучения Adam optimizer.

BATCH_SIZE = 64
# Размер mini-batch из replay buffer.

REPLAY_SIZE = 50_000
# Максимальное число transitions в replay buffer.

START_LEARNING_AFTER = 256
# Сколько transitions накопить перед первыми gradient updates.

TRAIN_FREQ = 1
# Делать update сети каждые N environment steps.

TARGET_UPDATE_FREQ = 250
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

EPS_DECAY_STEPS = 1500
# За сколько train steps epsilon спадает от EPS_START к EPS_END.

EVAL_EPSILON = 0.05
# Epsilon во время RL evaluation. Небольшое значение в dev-режиме помогает явно
# проверить action=1/setPhase; Q-diagnostics отдельно показывает greedy policy.


# -------------------------
# Reward
# -------------------------

REWARD_ALPHA_QUEUE = 0.2
# Вес queue в локальном автомобильном reward.

REWARD_BETA_NEIGHBOR = 0.3
# Вес mean neighbor penalty в reward.

REWARD_USE_NEIGHBORS = True
# True включает соседние TLS в reward, если граф соседей удалось построить.

REWARD_NORMALIZE = False
# Зарезервировано для будущего нормирования reward.

PEDESTRIAN_REWARD_ENABLED = False
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

REWARD_PEDESTRIAN_RUNNING_WEIGHT = 0.02
# Small penalty for pedestrians still active in the network.

REWARD_PEDESTRIAN_BLOCKED_WEIGHT = 0.05
# Small penalty for pedestrians with positive waiting time.

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

REWARD_STUCK_PHASE_PENALTY = 0.05
# Малый штраф за stuck phase. Значение намеренно небольшое, чтобы не провоцировать хаос.

REWARD_SWITCH_PENALTY = 0.01
# Малый штраф за фактическое переключение, чтобы агент не дергал фазы каждую секунду.


# -------------------------
# Training quality gates
# -------------------------

QUALITY_GATE_ENABLED = True
# После обучения делает короткую RL evaluation и проверяет, что агент реально управляет TLS.

QUALITY_GATE_MAX_RETRAIN_ROUNDS = 3
# Максимум дополнительных попыток обучения, если policy не проходит quality gate.

QUALITY_GATE_EXTRA_EPISODES = 5
# Сколько дополнительных episode добавить за одну попытку quality gate.

MIN_RL_PHASE_SET_COUNT = 1
# Минимальное число setPhase в RL evaluation для dev-сценария.

STRICT_DEV_VALIDATION = True
# В dev-режиме main.py завершается ошибкой, если RL не управляет TLS или нет pedestrians.

REQUIRE_RL_PHASE_SWITCHES = True
# Требовать, чтобы RL evaluation вызвал setPhase хотя бы один раз.

REQUIRE_RL_FIXED_METRIC_DIFFERENCE = True
# Требовать, чтобы RL и fixed метрики не были полностью одинаковыми.

METRIC_DIFF_TOLERANCE = 1e-6
# Допуск при сравнении RL/fixed метрик.

AUTO_GENERATE_PEDESTRIANS_IF_MISSING = True
# Если в scenario нет person/personFlow/walk, создать dev pedestrian demand.

PEDESTRIAN_DEMAND_COUNT = 30
# Сколько pedestrians создать для dev-проверки.

PEDESTRIAN_DEMAND_BEGIN = 0
# Начало генерации pedestrian demand.

PEDESTRIAN_DEMAND_END = 300
# Конец генерации pedestrian demand.

PEDESTRIAN_DEMAND_PREFIX = "auto_ped"
# Префикс deterministic person ids.

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

RL_USE_REAL_TIMING_PROGRAM_AS_BASE = True
# Если True, RL evaluation загружает ту же TLS-программу фаз, что и real_timing baseline,
# но действия модели всё равно применяются через TraCI setPhase.

RUN_REAL_TIMING_BASELINE = True
# Запускать ли baseline с программами светофоров из REAL_TIMING_FILE.

RUN_NATIVE_FIXED_BASELINE = True
# Запускать ли baseline с исходными программами светофоров из net.xml без REAL_TIMING_FILE.

PRIMARY_BASELINE = "real_timing"
# Основной baseline для сравнения. Возможные значения: "real_timing", "fixed_native".

STRICT_REAL_TIMING_VALIDATION = True
# Если True, ошибки в tls.add.xml или несовпадение TLS считаются критическими при validation.


# -------------------------
# Paths
# -------------------------

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

DEBUG_SCENARIO = True
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
