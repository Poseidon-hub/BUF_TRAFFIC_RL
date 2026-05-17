# Техническое описание программной реализации

## 1. Назначение программного комплекса

Программный комплекс предназначен для моделирования и предварительной оценки подхода к оптимизации режимов работы светофорных объектов с использованием модели обучения с подкреплением. В рамках проекта рассматривается дорожная сеть, заданная в формате SUMO, а объектами управления являются контроллеры светофоров Traffic Light System, далее TLS.

SUMO используется как микроскопический транспортный симулятор: он позволяет воспроизводить движение отдельных транспортных средств, задавать маршруты, подключать светофорные программы и получать через TraCI текущие значения очередей, ожидания, скоростей, прибытия и выбытия участников движения. TraCI применяется как интерфейс между Python-программой и симуляцией: через него программа считывает состояние сети и в RL-режиме управляет фазами светофоров.

Использование reinforcement learning обосновано тем, что задача управления светофорами является последовательной задачей принятия решений: действие на текущем шаге влияет на состояние очередей и задержек в последующие моменты времени. В текущей реализации используется DQN-подход с разделением параметров между агентами. Каждый TLS рассматривается как отдельный агент, но все агенты используют одну общую Q-сеть.

Экспериментальная логика программы строится вокруг сравнения трех режимов: RL-управления обученной DQN-моделью, native fixed-time baseline на основе исходных программ светофоров из `net.xml`, а также real timing baseline на основе внешнего файла `tls.add.xml`. Такое сравнение позволяет проверять, отличается ли поведение RL-модели от стандартной логики SUMO и от заданных вручную или потенциально реальных таймингов.

## 2. Общая архитектура проекта

Основной pipeline расположен в `main.py`. При запуске программа проверяет наличие SUMO и Python-зависимостей, формирует конфигурацию, обнаруживает сценарий в папке `scenario/`, выполняет короткую проверку запуска SUMO, определяет совместимость checkpoint, при необходимости запускает обучение DQN, затем выполняет evaluation для RL и baseline-режимов. По результатам evaluation формируются JSON/CSV-логи и сравнительные файлы.

Связь основных компонентов следующая: SUMO-сценарий задает сеть, маршруты и светофорные программы; `SumoMultiAgentEnv` запускает SUMO и предоставляет multi-agent окружение; DQN-модель выбирает действия для каждого TLS; `train.py` обучает модель и сохраняет checkpoint; `eval.py` загружает модель или запускает baseline-режимы; `main.py` сравнивает метрики и сохраняет результаты.

| Модуль | Назначение |
|---|---|
| `main.py` | Оркестрация полного pipeline: проверки окружения, обнаружение сценария, проверка checkpoint, обучение, evaluation, сравнение RL с baseline и сохранение результатов. |
| `src/sumo_env.py` | Multi-agent окружение SUMO/TraCI: запуск симуляции, сбор observation, применение RL-действий через `setPhase`, расчет reward, автомобильных и пешеходных метрик. |
| `src/train.py` | Цикл обучения shared DQN: взаимодействие с окружением, заполнение replay buffer, обновление Q-сети, запись train-логов и сохранение checkpoint. |
| `src/eval.py` | Evaluation для режимов `rl`, `fixed_native`, `real_timing` и `fixed_override`; агрегация метрик, action statistics, pedestrian metrics и baseline statistics. |
| `src/features.py` | Формирование вектора observation: локальные признаки, агрегаты соседей, время с последнего переключения, число controlled lanes и one-hot текущей фазы. |
| `src/graph_utils.py` | Построение графа соседства TLS из `net.xml` через `sumolib` или XML fallback; агрегация признаков соседних TLS. |
| `src/training_params.py` | Централизованные параметры запуска, обучения, reward, baseline, pedestrian generation, логирования и нормализации признаков. |
| `src/config.py` | Объект конфигурации, связывающий параметры из `training_params.py` с путями проекта и поддержкой переменных окружения для быстрых запусков. |
| `src/dqn.py` | Реализация `QNetwork`, общего replay buffer и класса `DQNShared` с online/target сетями. |
| `src/checkpointing.py` | Создание и проверка metadata checkpoint: scenario signature, TLS ids, размеры observation/action, число фаз и параметры обучения. |
| `src/scenario.py` | Обнаружение SUMO-сценария, классификация XML-файлов, генерация mode-specific `.sumocfg` и dev-файла пешеходного спроса. |
| `src/timing_profiles.py` | Парсинг `tls.add.xml` и представление внешних программ светофоров как timing profile. |
| `src/baseline.py` | Заложенный ручной `FixedTimeController` для режима `fixed_override`; основной native baseline в `main.py` работает без ручного `setPhase`. |
| `src/logging_utils.py` | Запись CSV, JSON, JSONL и печать метрик. |
| `src/utils.py` | Проверка SUMO, добавление `SUMO_HOME/tools` в `sys.path`, поиск SUMO-команд и установка seed. |

## 3. Использование SUMO-сценариев

Поиск сценария выполняется в `src/scenario.py` функцией `discover_scenario`. Сначала учитываются явно заданные пути из конфигурации, затем обычные пользовательские `*.sumocfg` в папке `scenario/`, исключая autogenerated-файлы. Если пользовательский `.sumocfg` не найден, программа ищет `*.net.xml` и маршрутные файлы `*.rou.xml`, `*.routes.xml`, `*.trips.xml`. Если полноценный сценарий не найден, в `main.py` заложена возможность создать demo-сценарий через `src/demo_scenario.py`.

В текущей структуре проекта в `scenario/` присутствуют:

- `scenario.sumocfg` - пользовательская конфигурация, указывающая `two_tls_corridor.net.xml`, `routes.rou.xml` и `tls.add.xml`;
- `two_tls_corridor.net.xml` - тестовая дорожная сеть с TLS `B1` и `C1`;
- `routes.rou.xml` - маршрутный файл с 3600 vehicle-маршрутами;
- `trips.trips.xml` - исходный trip-файл с 3600 trips, сгенерированный SUMO `randomTrips.py`; в текущих metadata/evaluation активным route-файлом является `routes.rou.xml`;
- `tls.add.xml` - SUMO additional-файл с внешними программами светофоров для `B1` и `C1`;
- `autogenerated_pedestrians.rou.xml` - автоматически созданный dev-файл пешеходного спроса на 30 `person`;
- `autogenerated.sumocfg`, `autogenerated_native.sumocfg`, `autogenerated_real_timing.sumocfg`, `autogenerated_rl.sumocfg` - конфигурации, создаваемые программой для разных режимов запуска.

Отдельный обычный `add.xml` в текущей версии проекта не найден. Файл `tls.add.xml` классифицируется как timing-файл, поскольку содержит теги `<tlLogic>`.

Mode-specific `.sumocfg` создаются функцией `write_mode_sumocfgs`:

- `autogenerated_native.sumocfg` подключает `two_tls_corridor.net.xml`, `routes.rou.xml` и `autogenerated_pedestrians.rou.xml`, но не подключает `tls.add.xml`;
- `autogenerated_real_timing.sumocfg` подключает те же route-файлы и дополнительно `tls.add.xml`;
- `autogenerated_rl.sumocfg` в текущей конфигурации также подключает `tls.add.xml`, потому что `RL_USE_REAL_TIMING_PROGRAM_AS_BASE=True`;
- `autogenerated.sumocfg` является legacy-конфигурацией и соответствует native-набору файлов без `tls.add.xml`.

Различие режимов следующее. Native fixed-time использует исходные программы светофоров из `net.xml` и не вызывает `setPhase` из Python-кода. Real timing baseline использует внешние программы из `tls.add.xml`, подключенные как SUMO additional-файл, но также не применяет RL-действия. RL-режим использует обученную DQN-модель и применяет действия к TLS через TraCI; при этом в текущей конфигурации он стартует с той же фазовой программы, что и real timing baseline.

В текущем `two_tls_corridor.net.xml` для TLS `B1` и `C1` заданы исходные static-программы `programID="0"` с 6 фазами. В `tls.add.xml` для тех же TLS заданы static-программы `programID="manual_fixed"` с 6 фазами, длительностями 40, 4, 3, 35, 5 и 3 секунды; для `C1` задан offset 9 секунд. По комментариям в `tls.add.xml`, эти тайминги являются manual fixed-time программами для тестового коридора, а не подтвержденными реальными городскими таймингами.

## 4. Модель управления светофорами

В проекте реализована multi-agent постановка: каждый TLS из `traci.trafficlight.getIDList()` рассматривается как отдельный агент. В текущем тестовом сценарии фактически обнаруживаются два агента: `B1` и `C1`. При этом используется shared DQN, то есть все агенты имеют одну общую Q-сеть и одну target-сеть. Такой подход уменьшает число параметров и позволяет всем светофорам обучаться на общем наборе transition.

Общий replay buffer реализован в `src/dqn.py` классом `ReplayBuffer` и хранит transition от всех TLS: observation, action, reward, next observation и done. В `src/train.py` после каждого шага окружения transition каждого TLS добавляется в один общий буфер. Обновление DQN выполняется по mini-batch, случайно выбранному из этого общего буфера.

Action space имеет размер 2:

- `action = 0` - удерживать текущую фазу;
- `action = 1` - переключиться на следующую зеленую фазу.

При `action = 1` окружение проверяет ограничение `min_green`. Если с момента последнего переключения прошло меньше `MIN_GREEN`, действие блокируется и счетчик `blocked_by_min_green_count` увеличивается. Если переключение разрешено, окружение ищет следующую зеленую фазу в активной TLS-программе и вызывает `traci.trafficlight.setPhase(tls_id, next_phase)`.

При `action = 0` текущая версия `src/sumo_env.py` также вызывает `setPhase(tls_id, current_phase)`, чтобы явно удержать текущую фазу. Поэтому в RL-логах `phase_set_count` включает как `hold_phase_set_count`, так и `switch_phase_set_count`. Это относится только к RL-режиму и ручному `fixed_override`; native fixed-time и real timing baseline передают пустой словарь действий, поэтому `_apply_actions` не вызывает `setPhase`.

Ограничение `MIN_GREEN=5` задано в `src/training_params.py`. Оно защищает модель от слишком частых переключений. Дополнительно в reward используется малый штраф за фактическое переключение, если TLS был переключен на текущем шаге.

## 5. Состояние агента / observation

Observation формируется в `src/features.py` функцией `build_observation`. Он состоит из 6 базовых признаков и one-hot вектора текущей фазы. Размер one-hot части равен максимальному числу фаз среди TLS в текущем сценарии. В текущем checkpoint `obs_dim=12`, потому что базовая часть имеет размер 6, а у TLS `B1` и `C1` по 6 фаз.

Фактически используемые признаки:

- локальная очередь `queue` - сумма `lane.getLastStepHaltingNumber` по входящим vehicle lanes текущего TLS;
- локальное среднее время ожидания `waiting_time` - сумма `vehicle.getWaitingTime` по автомобилям на входящих lanes, деленная на число автомобилей;
- средняя очередь соседних TLS `mean_queue`;
- среднее ожидание соседних TLS `mean_wait`;
- `time_since_switch` - время с последнего изменения фазы TLS;
- `num_lanes` - число входящих controlled lanes, отфильтрованных как автомобильные;
- one-hot код текущей фазы TLS.

Входящие lanes определяются через `trafficlight.getControlledLanes` или, если список пуст, через `getControlledLinks`. Затем lanes фильтруются: исключаются internal lanes, pedestrian-only lanes, walkingarea и crossing lanes. Это важно, чтобы пешеходные дорожки не попадали в автомобильные очереди.

### 5.1. Нормализация признаков

Нормализация выполняется в `src/features.py` делением базовых признаков на коэффициенты из конфигурации:

- `QUEUE_NORM = 20.0` - нормализация локальной и соседской очереди;
- `WAIT_NORM = 300.0` - нормализация локального и соседского waiting time;
- `TIME_SINCE_SWITCH_NORM = 120.0` - нормализация времени с последнего переключения;
- `LANE_NORM = 16.0` - нормализация числа controlled vehicle lanes.

One-hot код текущей фазы не нормируется, так как уже состоит из значений 0 и 1.

## 6. Графовая структура дорожной сети

Граф соседства TLS строится в `src/graph_utils.py`. Основной способ использует `sumolib.net.readNet`: для каждого traffic light извлекаются входящие и исходящие edges, затем TLS связываются как соседи, если их edges или junctions показывают прямую дорожную связь. Если `sumolib` недоступен или построение не удалось, используется XML fallback: из `net.xml` извлекаются `tlLogic` и junctions типа `traffic_light`, а затем анализируются обычные edges между ними.

Соседние TLS используются не как вход полноценной графовой нейросети, а как источник агрегированных признаков. Функция `aggregate_neighbors` считает средние значения очереди и ожидания по соседним TLS. Эти два значения входят в observation, а также фактически участвуют в reward как neighbor penalty.

Если граф построить невозможно или для TLS нет соседей, список соседей остается пустым. В этом случае neighbor features равны нулю, а neighbor-компонента reward не влияет на итоговую награду. Полноценная GNN в текущей версии не реализована.

## 7. Функция награды

Reward рассчитывается в `src/sumo_env.py` функцией `_compute_rewards`. Для каждого TLS сначала считается локальный штраф:

```text
local_term = waiting_time + alpha * queue
```

где `alpha = REWARD_ALPHA_QUEUE = 0.2`. Затем для соседних TLS рассчитываются аналогичные значения, усредняются и добавляются с весом `beta = REWARD_BETA_NEIGHBOR = 0.3`:

```text
reward = -local_term - beta * neighbor_mean
```

Таким образом, модель максимизирует отрицательную величину задержек и очередей: чем меньше очередь и время ожидания, тем выше reward. Neighbor penalty учитывает состояние соседних светофоров, если граф соседства построен и соседи присутствуют в текущем `stats`.

В коде также есть два дополнительных штрафа:

- `REWARD_STUCK_PHASE_PENALTY_ENABLED=True`: если фаза удерживается не менее `REWARD_STUCK_PHASE_AFTER_SECONDS=60` секунд и при этом есть очередь или ожидание, из reward вычитается `REWARD_STUCK_PHASE_PENALTY=0.05`;
- если TLS фактически переключился на текущем шаге, из reward вычитается `REWARD_SWITCH_PENALTY=0.01`.

Пешеходные показатели в текущей версии не включены в функцию награды. В `training_params.py` задано `PEDESTRIAN_REWARD_ENABLED=False`, а `PEDESTRIAN_WAITING_PENALTY=0.0`. В коде `_compute_rewards` оставлен TODO для будущего добавления pedestrian penalty. В текущей версии пешеходные показатели используются для оценки качества, но не включены в функцию награды. Это оставляет возможность дальнейшего расширения модели.

Параметр `REWARD_USE_NEIGHBORS=True` присутствует в конфигурации и metadata, однако текущая формула reward напрямую не проверяет этот флаг. Neighbor-компонента фактически определяется наличием соседей в `self.neighbors` и ненулевым `beta`.

## 8. Обучение DQN

Обучение запускается из `main.py`, если checkpoint отсутствует, поврежден, несовместим с текущим сценарием или если включен `FORCE_RETRAIN=True`. Проверка совместимости выполняется через `src/checkpointing.py`: программа сравнивает scenario signature, `obs_dim`, `action_dim`, список TLS, число TLS и число фаз по TLS.

Файл `checkpoints/dqn.pt` содержит веса DQN и служебное состояние: `obs_dim`, `action_dim`, `q_net`, `target_net`, состояние optimizer и `steps_done`. Файл `checkpoints/dqn_meta.json` хранит metadata: scenario signature, пути к `sumocfg`, `net`, route/additional/timing файлам, `tls_ids`, `tls_count`, `obs_dim`, `action_dim`, `num_phases_per_tls`, snapshot training parameters и время создания. В текущем `dqn_meta.json` указаны TLS `B1` и `C1`, `obs_dim=12`, `action_dim=2`, по 6 фаз на каждый TLS.

Если `dqn.pt` или `dqn_meta.json` удалить, `main.py` определит checkpoint как отсутствующий и запустит обучение заново. Если карта, маршруты, timing-файлы, число фаз, TLS ids или размеры observation/action изменились, scenario signature или metadata перестанут совпадать. При `AUTO_RETRAIN_ON_SCENARIO_CHANGE=True` программа автоматически переобучает модель; если автоматическое переобучение отключить, запуск завершится с сообщением о необходимости удалить checkpoint или включить авто-переобучение.

DQN реализована как MLP в `src/dqn.py`: несколько блоков `Linear + ReLU`, затем выходной слой на 2 действия. Используются online Q-сеть, target-сеть, Adam optimizer, replay buffer, Huber loss (`smooth_l1_loss`) и gradient clipping.

| Параметр | Значение в текущем коде | Назначение |
|---|---:|---|
| `TRAIN_EPISODES` | 10 | Число episode обучения при стандартном запуске без готового checkpoint. |
| `EPISODE_SECONDS` | 300 | Длительность одного episode в секундах SUMO. |
| `TRAIN_SECONDS` | 3000 | Общий лимит шагов обучения: `EPISODE_SECONDS * TRAIN_EPISODES`. |
| `GAMMA` | 0.99 | Discount factor для будущих reward. |
| `LEARNING_RATE` | 0.001 | Скорость обучения Adam optimizer. |
| `BATCH_SIZE` | 64 | Размер mini-batch из replay buffer. |
| `REPLAY_SIZE` | 50000 | Максимальная емкость общего replay buffer. |
| `START_LEARNING_AFTER` | 200 | Минимальное число transition перед первыми gradient updates. |
| `TRAIN_FREQ` | 4 | Частота обновления Q-сети в шагах окружения. |
| `TARGET_UPDATE_FREQ` | 250 | Частота копирования весов online-сети в target-сеть. |
| `EPS_START` | 1.0 | Начальное значение epsilon при обучении. |
| `EPS_END` | 0.05 | Минимальное значение epsilon после decay. |
| `EPS_DECAY_STEPS` | 1500 | Число шагов decay epsilon. |
| `EVAL_EPSILON` | 0.05 | Epsilon при RL evaluation в текущей конфигурации. |
| `MIN_GREEN` | 5 | Минимальное время между переключениями фазы TLS. |
| `HIDDEN_DIM` | 128 | Размер скрытых слоев Q-сети. |
| `NUM_HIDDEN_LAYERS` | 2 | Число скрытых `Linear + ReLU` блоков. |

После обучения включен quality gate: `main.py` запускает короткую RL evaluation и проверяет, что `phase_set_count` не меньше `MIN_RL_PHASE_SET_COUNT=1`. Если policy не управляет фазами, программа может выполнить дополнительные раунды обучения, число которых ограничено `QUALITY_GATE_MAX_RETRAIN_ROUNDS=3`.

## 9. Baseline-режимы

### 9.1. RL evaluation

RL evaluation вызывается как `evaluate(cfg, mode="rl", episodes=cfg.eval_episodes)`. Этот режим требует существующий совместимый `checkpoints/dqn.pt`. Для каждого TLS вычисляются Q-values, выбирается действие через `agent.act`, затем действие применяется окружением. В текущем коде используется `epsilon=config.eval_epsilon`, то есть `EVAL_EPSILON=0.05`, а не строго 0.0. Дополнительно сохраняются `action_stats` и `q_value_stats`: число hold/switch решений, блокировки из-за `min_green`, число вызовов `setPhase`, средние Q-values и доля greedy hold/switch.

### 9.2. Native fixed-time baseline

Native fixed-time baseline запускается как `mode="fixed_native"` или legacy-алиас `mode="fixed"`. В этом режиме `eval.py` передает в окружение пустой словарь действий. Поэтому `_apply_actions` не вызывает `setPhase`, а SUMO сам выполняет программы светофоров из `net.xml`. В логах baseline описан как `controlled_by: "SUMO native tlLogic from net.xml"`, а `phase_set_count` должен быть равен 0.

### 9.3. Real timing baseline

Real timing baseline запускается как `mode="real_timing"`. Он использует `autogenerated_real_timing.sumocfg`, куда подключен `tls.add.xml`. В `src/sumo_env.py` additionally загружается timing profile, проверяется наличие программ для TLS, активные `programID`, число фаз и длительности фаз. При этом RL-действия не применяются: `actions = {}`, поэтому `setPhase` не вызывается, а SUMO выполняет static-программы из `tls.add.xml`.

Real timing baseline важен для дипломного эксперимента, потому что он позволяет сравнивать RL-модель не только с исходным fixed-time режимом из `net.xml`, но и с внешним набором таймингов. В дальнейшем такой файл может быть заменен на реальные или экспертно заданные программы светофоров, полученные для исследуемой дорожной сети.

## 10. Пешеходы в симуляции

Пешеходы обнаруживаются в `src/scenario.py` через поиск тегов `person`, `personFlow` и `walk` в route-файлах. В текущих `routes.rou.xml` и `trips.trips.xml` такие теги не обнаружены. Поскольку `AUTO_GENERATE_PEDESTRIANS_IF_MISSING=True`, программа создала `scenario/autogenerated_pedestrians.rou.xml` для dev-проверки. В текущем файле находится 30 `person` с id вида `auto_ped_0`, `auto_ped_1` и так далее.

Генерация dev-пешеходов выполняется через `sumolib`: выбираются edges, разрешающие класс `pedestrian`, затем создаются маршруты `<walk from="..." to="..."/>`. После генерации выполняется smoke-test SUMO. Если межреберный маршрут не проходит, предусмотрен fallback на walk по одному edge.

Пешеходы участвуют в SUMO-симуляции как обычные участники движения из route-файла, но в текущей версии они не являются отдельными RL-агентами. Модель управляет только TLS. Пешеходные дорожки также специально исключаются из автомобильных controlled lanes, чтобы не искажать автомобильные очереди и ожидания.

Пешеходные метрики собираются в `src/sumo_env.py`:

- `pedestrian_departed`;
- `pedestrian_arrived`;
- `pedestrian_running`;
- `pedestrian_running_max`;
- `pedestrian_waiting_count`;
- `pedestrian_waiting_count_sum`;
- `pedestrian_total_waiting_time`;
- `pedestrian_avg_waiting_time`;
- `pedestrian_waiting_time_available`;
- `pedestrian_waiting_time_note`.

В `main.py` реализовано сравнение pedestrian metrics для `RL vs real_timing` и `RL vs native_fixed`. Для `arrived` используется процент улучшения как метрика на максимум; для `avg_waiting_time`, `total_waiting_time` и `waiting_count` используется процент улучшения как метрика на минимум; для `running` выводится только delta, поскольку число пешеходов в сети в конкретный момент не всегда напрямую означает улучшение или ухудшение.

## 11. Метрики качества

Автомобильные метрики собираются в `src/sumo_env.py` и агрегируются в `src/eval.py`. Основные показатели:

- `avg_queue` - среднее число остановившихся автомобилей в очереди по TLS и шагам evaluation;
- `avg_waiting_time` - среднее время ожидания автомобилей на controlled vehicle lanes;
- `total_waiting_time` - накопленная сумма ожидания автомобилей за episode;
- `total_reward` - сумма reward по всем TLS и шагам episode;
- `throughput` - число прибывших транспортных средств, в коде совпадает с `arrived`;
- `departed` - число отправившихся транспортных средств;
- `arrived` - число прибывших транспортных средств;
- `episode_steps` - число шагов simulation/evaluation;
- `avg_speed` - средняя скорость активных автомобилей;
- `avg_time_loss` - средний time loss активных автомобилей.

Метрики очереди, ожидания, total waiting time и time loss желательно минимизировать. Throughput, arrived, avg_speed и total_reward желательно максимизировать. Для reward в сравнительных файлах используется `reward_delta`, а не процент, потому что reward в текущей формуле отрицательный; процентная интерпретация для отрицательных величин может быть вводящей в заблуждение.

| Метрика | Смысл | Желательное направление |
|---|---|---|
| `avg_queue` | Средняя очередь у TLS. | меньше |
| `avg_waiting_time` | Среднее ожидание автомобилей на controlled lanes. | меньше |
| `total_waiting_time` | Накопленное ожидание автомобилей за episode. | меньше |
| `throughput` | Число автомобилей, завершивших маршрут. | больше |
| `departed` | Число автомобилей, вошедших в сеть. | зависит от сценария, обычно контролируется спросом |
| `arrived` | Число автомобилей, прибывших к назначению. | больше |
| `avg_speed` | Средняя скорость активных автомобилей. | больше |
| `avg_time_loss` | Средняя потеря времени относительно свободного движения. | меньше |
| `total_reward` | Суммарная награда RL-формулы. | больше |

Для `avg_queue` и `avg_waiting_time` процент улучшения считается функцией `_improvement_pct`:

```text
improvement_pct = (baseline_value - rl_value) / baseline_value * 100
```

Если baseline-значение близко к нулю, процент улучшения записывается как `null`. Для throughput используется `throughput_delta = rl_throughput - baseline_throughput`, для reward - `reward_delta = rl_total_reward - baseline_total_reward`.

## 12. Логирование и результаты

Логи сохраняются в папку `logs/`. В текущей структуре присутствуют следующие основные файлы:

| Файл | Назначение |
|---|---|
| `eval_rl.json`, `eval_rl.csv` | Итоги RL evaluation, включая метрики, `action_stats`, `q_value_stats`, timing source и pedestrian metrics. |
| `eval_real_timing.json`, `eval_real_timing.csv` | Итоги real timing baseline с `tls.add.xml`; baseline должен иметь `phase_set_count=0`. |
| `eval_fixed.json` | Legacy-имя для native fixed-time baseline. |
| `eval_fixed_native.json`, `eval_fixed_native.csv` | Итоги native fixed-time baseline на программах из `net.xml`. |
| `comparison_rl_vs_real_timing.json`, `.csv` | Сравнение RL с real timing baseline, включая автомобильные и пешеходные показатели. |
| `comparison_rl_vs_native_fixed.json`, `.csv` | Сравнение RL с native fixed-time baseline. |
| `comparison_rl_vs_fixed_native.json`, `.csv` | Альтернативное имя того же сравнения с native fixed-time baseline. |
| `comparison.json`, `comparison.csv` | Сводное сравнение с primary baseline и блоком pedestrian metrics для всех режимов. |
| `train_metrics.csv`, `train_metrics.jsonl` | Логи обучения: episode, steps, epsilon, loss, replay size, reward, queue/waiting metrics и action counters. |
| `action_probe.json` | Результат проверки того, что action=1 действительно приводит к попыткам переключения и вызовам `setPhase`. |
| `validation_*.txt` | stdout/stderr validation-запусков. |
| `validation_pedestrians.json` | Результат проверки пешеходных метрик во временном сценарии. |
| `vehicle_lane_filter_report.json` | Отчет о фильтрации pedestrian/internal lanes из автомобильных lanes. |

Текущие логи отражают предварительные dev-запуски на 300 секундах тестового сценария. Их нельзя трактовать как финальное подтверждение эффективности модели на реальной городской сети.

## 13. Проверки корректности

В папке `scripts/` находятся validation-скрипты. Они предназначены для проверки технической корректности pipeline. В рамках подготовки этого документа они были прочитаны, но не запускались, чтобы не инициировать дополнительные SUMO-симуляции.

| Скрипт | Что проверяет |
|---|---|
| `run_all_validations.py` | Последовательно запускает все validation-скрипты и останавливается при первой ошибке. |
| `validate_training_params.py` | Проверяет наличие обязательных параметров в `src/training_params.py`, связь основных модулей с конфигурацией и отсутствие ряда magic numbers вне файла параметров. |
| `validate_scenario_signature.py` | Проверяет, что изменение route-файла меняет scenario signature и старый checkpoint становится несовместимым. |
| `action_probe.py` | Запускает окружение с `min_green=0`, несколько раз применяет `action=1` ко всем TLS и проверяет `decision_count`, `switch_count`, `phase_set_count` и фактическое изменение фаз. |
| `validate_pedestrians.py` | Проверяет наличие pedestrian metrics в eval-логах, создает временный сценарий с пешеходом и проверяет, что pedestrian/internal lanes не попадают в автомобильные lanes. |
| `validate_real_timing_baseline.py` | Проверяет, что `tls.add.xml` найден и распарсен, real timing использует нужные программы и длительности фаз, baseline не вызывает `setPhase`, а native `.sumocfg` не подключает `tls.add.xml`. |
| `validate_pipeline.py` | Комплексная проверка: удаление checkpoint запускает обучение заново, создаются `dqn.pt` и `dqn_meta.json`, eval-логи содержат нужные поля, RL вызывает `setPhase`, fixed/real baseline не вызывают `setPhase`, pedestrian metrics не нулевые при наличии пешеходов, corrupt checkpoint и invalid signature приводят к переобучению. |

Особенно важные проверки для дипломной логики:

- удаление `checkpoints/dqn.pt` и `checkpoints/dqn_meta.json` должно запускать обучение заново;
- RL evaluation должна иметь ненулевой `phase_set_count`;
- fixed baseline и real timing baseline должны иметь `phase_set_count=0`;
- `tls.add.xml` должен подключаться в `autogenerated_real_timing.sumocfg` и `autogenerated_rl.sumocfg`, но не в `autogenerated_native.sumocfg`;
- pedestrian metrics должны присутствовать в eval-логах, а при наличии dev-пешеходов `pedestrian_departed` не должен быть нулевым для всех режимов.

## 14. Текущие ограничения реализации

Текущая карта является тестовой сетью `two_tls_corridor.net.xml`, а не реальной городской дорожной сетью. В ней два управляемых TLS: `B1` и `C1`. Поэтому текущие результаты пригодны для проверки работоспособности программного контура, но не для финального вывода о городской эффективности метода.

Текущие тайминги `tls.add.xml` являются manual fixed-time программами для тестового сценария. Они заложены как возможность сравнения с внешними таймингами, но в текущей версии не являются подтвержденными реальными программами светофоров.

Модель еще не обучалась на реальной карте. Короткий dev-режим с `EPISODE_SECONDS=300`, `TRAIN_EPISODES=10` и одним seed не является финальным экспериментом. Результаты на 300 секундах нельзя считать окончательными.

Пешеходы пока не входят в reward и не являются агентами. Они участвуют в SUMO-симуляции и логируются через pedestrian metrics, но модель оптимизируется только по автомобильным очередям, ожиданию, neighbor-компоненте и штрафам за поведение фаз.

В проекте используется DQN с parameter sharing, а не полноценная графовая нейросеть. Граф дорожной сети используется только для агрегирования признаков соседних TLS и neighbor penalty в reward.

`DECISION_INTERVAL` задан в `src/training_params.py` и `src/config.py`, но в текущих циклах `train.py` и `eval.py` действие фактически выбирается на каждом шаге SUMO. Это следует учитывать при описании текущей версии.

Для дипломного эксперимента нужно будет прогнать модель на реальной карте, реальных или приближенных транспортных потоках, внешних таймингах и нескольких seed. Только после этого можно будет формировать итоговые таблицы, графики и статистически аккуратные выводы.

## 15. Как это можно описать в дипломе

Разработанная программная система реализует экспериментальный контур для оценки метода управления светофорными объектами на основе обучения с подкреплением. В качестве среды моделирования используется SUMO, а взаимодействие между Python-программой и транспортной симуляцией осуществляется через интерфейс TraCI. Такой подход позволяет получать текущее состояние дорожной сети на каждом шаге моделирования и программно изменять фазы светофорных объектов в режиме RL-управления.

В программной реализации каждый светофорный контроллер рассматривается как отдельный агент multi-agent системы. При этом агенты используют общую DQN-модель с разделяемыми параметрами и общий replay buffer. Наблюдение агента включает локальные характеристики очереди и ожидания, агрегированные показатели соседних светофоров, время с момента последнего переключения, число контролируемых автомобильных полос и one-hot представление текущей фазы. Граф дорожной сети используется для определения соседних светофоров и вычисления агрегированных признаков, однако полноценная графовая нейронная сеть в текущей версии не применяется.

Экспериментальный контур поддерживает сравнение обучаемого RL-режима с двумя baseline-подходами. Первый baseline использует исходные static-программы светофоров из SUMO-сети и не вмешивается в управление через TraCI. Второй baseline использует внешний файл `tls.add.xml`, что позволяет подключать заданные вручную или в дальнейшем реальные тайминги светофорных программ. Для всех режимов собираются сопоставимые метрики транспортного потока, включая среднюю очередь, среднее и суммарное время ожидания, throughput, число прибывших транспортных средств, суммарную награду, скорость и time loss.

Текущая версия является программным прототипом и прошла предварительную проверку на тестовом SUMO-сценарии. Она не содержит финального эксперимента на реальной городской карте и не доказывает превосходство модели над реальными таймингами. Основное назначение текущей реализации состоит в подготовке воспроизводимого pipeline, который после подключения реальной карты, потоков и таймингов может быть использован для полноценной экспериментальной части дипломной работы.

## 16. Что нужно будет доделать после получения реальной карты

- Заменить тестовый SUMO-сценарий `two_tls_corridor.net.xml` на реальную карту исследуемой дорожной сети.
- Подключить реальные или приближенные маршруты и транспортные потоки.
- Подключить реальные или экспертно заданные тайминги в формате `tls.add.xml`.
- Проверить, что TLS ids в `tls.add.xml` совпадают с TLS ids реальной SUMO-сети.
- Удалить старый checkpoint `checkpoints/dqn.pt` и metadata `checkpoints/dqn_meta.json`.
- Переобучить модель на новой карте и новых потоках.
- Выполнить сравнение `RL vs real_timing`.
- Выполнить сравнение `RL vs native fixed-time`, если такой baseline сохраняет смысл для выбранной сети.
- Провести несколько запусков с разными seed.
- Посчитать средние значения и разброс по ключевым метрикам.
- Подготовить таблицы и графики для диплома.
- Отдельно зафиксировать, входят ли пешеходные показатели только в evaluation или будут добавлены в reward в расширенной версии модели.
