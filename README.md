# Multi-Agent Traffic Light RL for SUMO

Упрощенная дипломная версия проекта. В проекте оставлен только основной эксперимент:

- SUMO-сценарий из `scenario/`;
- генерация спроса машин и пешеходов;
- RL-управление светофорами через TraCI `trafficlight.setPhase`;
- Double/Dueling DQN, Adam, gradient clipping и epsilon-greedy exploration;
- fixed/native baseline без ручного `setPhase`;
- метрики для машин и пешеходов;
- четыре режима запуска и JSON/CSV-логи.

## Запуск

```powershell
python main.py
```

Меню:

```text
1) Быстрый
2) Нормальный
3) Долгий
4) Ручной
Выберите режим:
```

Для автоматической проверки без меню:

```powershell
$env:BFU_NON_INTERACTIVE='1'
$env:BFU_RUN_MODE='validate_fast'
python main.py
```

## Проверки

```powershell
python -m compileall main.py src scripts
python scripts/action_probe.py
python scripts/validate_metrics_nonzero.py
```

Основные результаты сохраняются в `logs/`:

- `eval_rl.json`
- `eval_fixed.json`
- `comparison_rl_vs_fixed_native.json`
- `comparison_rl_vs_fixed_native.csv`
- `run_config_resolved.json`
- `validation_metrics_report.json`
