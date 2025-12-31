# Детекция дефектов речи с помощью скороговорок

# Part 1 (Установка)

### Установка

```bash
git clone https://github.com/dkzmn/speech-defect-detector.git
cd speech-defect-detector
poetry install
poetry shell
pre-commit install
pre-commit run -a
```
### Запуск обучения

```bash
python -m speech_defect_detector.commands train
```

Команда автоматически:
- Скачает данные через DVC
- Загрузит конфигурацию из `configs/config.yaml`
- Запустит обучение модели с использованием PyTorch Lightning
- Залогирует метрики в MLflow
- Сохранит лучшую модель в `checkpoints/`

### Изменение гиперпараметров

Гиперпараметры можно изменить через конфигурационные файлы в директории `configs/`:

- `configs/model/model.yaml` - параметры модели
- `configs/data/data.yaml` - параметры данных
- `configs/training/training.yaml` - параметры обучения
- `configs/logging/logging.yaml` - параметры логирования

Также можно переопределить параметры через командную строку:

```bash
python -m speech_defect_detector.commands train training.batch_size=64 training.learning_rate=0.0005
```

### Структура проекта

```
speech-defect-detector/
├── configs/                 # Конфигурационные файлы Hydra
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   ├── training/
│   └── logging/
├── speech_defect_detector/  # Основной пакет
│   ├── data/                # Модули для работы с данными
│   ├── models/              # Определения моделей
│   ├── training/            # Модули для обучения
│   └── commands.py          # Точка входа CLI
├── data/                    # Данные (управляются через DVC)
├── checkpoints/             # Сохраненные модели
├── pyproject.toml           # Зависимости проекта
├── .pre-commit-config.yaml  # Конфигурация pre-commit
└── README.md
```

### Логирование

Все метрики обучения логируются в MLflow:
- Loss (train/val/test)
- Accuracy
- F1 Score
- ROC-AUC

Также логируются гиперпараметры и git commit ID для воспроизводимости экспериментов.

---

# Part 2 (Постановка задачи)

