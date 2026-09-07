# MLOps-практикум для студентов

> Справочная методичка. Конкретные ключи доступа заменены на placeholders для публикации. Для запуска этой версии проекта используйте README: подключения берутся из локального `.env` или GitHub Secrets, MLflow работает на порту 5000.

Цель: поднять локальный стек (Postgres + MinIO + MLflow), сгенерировать данные, обучить модель, зарегистрировать её и пропустить через Quality Gate.

Стек: **Docker Compose, MLflow, MinIO, DVC, GitHub Actions**.

Ожидаемое время: около 1 часа.

---

## Что должно получиться

1. Три контейнера работают: `mlflow-postgres`, `mlflow-minio`, `mlflow-server`.
2. Файл `data.csv` — 1000 строк, 10 признаков.
3. В MLflow есть эксперимент, run с метрикой `f1_score` и модель `production_classifier`.
4. Quality Gate переводит модель в стадию **Staging**, если `f1_score >= 0.80`.

---

## Требования

- Docker Desktop (должен быть запущен)
- Python **3.10 или 3.11** (не 3.14 — клиент MLflow 2.7.1 с ним несовместим)
- Git
- Репозиторий этого практикума

Проверка версий:

```bash
docker --version
python3.11 --version # или python3.10 --version
```

Если Docker пишет `Cannot connect to the Docker daemon` — открой Docker Desktop и повтори.

---

## Шаг 1. Клонировать репозиторий

```bash
git clone https://github.com/Shahnurhon/MLOps.git
cd MLOps
```

Если репозиторий уже скачан:

```bash
cd /путь/к/MLOps
```

---

## Шаг 2. Поднять инфраструктуру

```bash
docker-compose up -d
```

Что делает команда:

- `postgres` — база метаданных MLflow (пользователь `mlflow`, БД `mlflow_db`)
- `s3` (MinIO) — хранилище артефактов моделей, порты `9000` (API) и `9001` (UI)
- `mlflow` — tracking server, порт **5001**

Проверка:

```bash
docker-compose ps
```

Все три контейнера должны быть `Up`. У `mlflow-postgres` статус `healthy`.

Если `mlflow-server` только что стартовал, подожди 15–30 секунд: внутри контейнера сначала ставится `psycopg2-binary` и `boto3`.

Открой в браузере:

- MLflow UI: http://localhost:5001
- MinIO UI: http://localhost:9001
логин `<YOUR_LOCAL_CREDENTIAL>`, пароль `<YOUR_LOCAL_CREDENTIAL>`

### Если порт 5000/5001 занят

В этом репозитории MLflow слушает **5001**, потому что 5000 часто занят на macOS (Control Center / AirPlay).

Если 5001 тоже занят, в `docker-compose.yml` поменяй и проброс порта, и `--port` у MLflow, затем:

```bash
docker-compose up -d
```

Скрипты по умолчанию ходят на `http://localhost:5000`. Поэтому дальше всегда задаём:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
```

---

## Шаг 3. Создать виртуальное окружение

Системный Python (особенно Homebrew 3.14) пакеты ставить не даст: `externally-managed-environment`. Нужен venv на 3.10/3.11.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

На Windows:

```bat
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
```

Что здесь важно:

- `mlflow==2.7.1` в `requirements.txt` должен совпадать с образом `ghcr.io/mlflow/mlflow:v2.7.1`
- если поставить свежий MLflow 3.x, обучение упадёт с `404` на `/api/2.0/mlflow/logged-models`

Проверка:

```bash
python -c "import sklearn, pandas, mlflow, boto3; print(mlflow.__version__)"
```

Должно напечатать `2.7.1`.

---

## Шаг 4. Сгенерировать датасет

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
python prepare_data.py
```

Скрипт вызывает `sklearn.datasets.make_classification` (1000 строк, 10 фичей) и пишет `data.csv`.

Ожидаемый вывод:

```text
Генерация датасета: 1000 строк, 10 признаков...
Датасет сохранён: .../data.csv (1000 строк)
```

---

## Шаг 5. Обучить модель и зарегистрировать в MLflow

```bash
python train.py
```

Что делает `train.py`:

1. Выставляет ключи MinIO (`<YOUR_LOCAL_CREDENTIAL>` / `<YOUR_LOCAL_CREDENTIAL>`) и `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`
2. Создаёт бакет `s3://mlflow/`, если его ещё нет
3. Ждёт готовности MLflow
4. Проверяет, что файл `data.csv` существует
5. Учит `RandomForestClassifier` (`n_estimators=100`, `max_depth=10`)
6. Логирует параметры и метрику `f1_score`
7. Регистрирует модель в Model Registry под именем `production_classifier`

Ожидаемый вывод (числа могут чуть отличаться):

```text
MLflow tracking URI: http://localhost:5001
Создан S3-бакет: mlflow
MLflow сервер доступен
Загружен датасет: data.csv (1000 строк)
Обучение RandomForestClassifier (n_estimators=100, max_depth=10)...
f1_score: 0.8426
Модель зарегистрирована как 'production_classifier'
```

Проверь в UI:

- http://localhost:5001 → эксперимент `mlops-workshop` → run с `f1_score`
- вкладка **Models** → `production_classifier` версии `1`
- http://localhost:9001 → бакет `mlflow` с артефактами

---

## Шаг 6. Quality Gate

```bash
python eval_gate.py
```

Скрипт:

1. Берёт последнюю версию `production_classifier`
2. Читает `f1_score` из run
3. Если `f1_score >= 0.80` — переводит версию в **Staging** и завершается с кодом `0`
4. Если меньше — печатает ошибку и завершается с кодом `1`

Ожидаемый вывод:

```text
Последняя версия модели: v1
f1_score = 0.8426, порог = 0.80
Quality Gate пройден: 'production_classifier' v1 переведена в Staging
```

В MLflow UI у версии модели стадия должна стать **Staging**.

---

## Шаг 7. Повторить всё одной цепочкой

Если шаги выше уже сделаны по отдельности, этот блок можно пропустить. Это шпаргалка «с нуля»:

```bash
cd MLOps
docker-compose up -d
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://localhost:5001
python prepare_data.py
python train.py
python eval_gate.py
```

---

## Шаг 8. DVC (версия данных)

Данные не кладём в Git (`data.csv` в `.gitignore`). Версионируем через DVC, remote — тот же MinIO.

```bash
dvc init
dvc remote add -d minio s3://mlflow/dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id <YOUR_LOCAL_CREDENTIAL>
dvc remote modify minio secret_access_key <YOUR_LOCAL_CREDENTIAL>
dvc add data.csv
dvc push
```

Что происходит:

- `dvc init` — создаёт служебные файлы DVC
- `dvc remote add` — remote `minio` в бакете `s3://mlflow/dvc`
- `dvc add data.csv` — вместо большого CSV в Git попадает маленький `data.csv.dvc`
- `dvc push` — сам файл уходит в MinIO

Проверка: в MinIO UI в бакете `mlflow` появится префикс `dvc/`.

---

## Частые ошибки

| Симптом | Причина | Что сделать |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker не запущен | Открыть Docker Desktop |
| `Ports are not available ... 5000` | Порт занят | Использовать `5001` и `export MLFLOW_TRACKING_URI=http://localhost:5001` |
| `externally-managed-environment` | pip в системный Python | Создать venv на 3.10/3.11 |
| `No module named 'sklearn'` | Зависимости не установлены | `source .venv311/bin/activate` и `pip install -r requirements.txt` |
| `404 ... /logged-models` | Клиент MLflow 3.x, сервер 2.7.1 | `pip install mlflow==2.7.1` |
| `Файл data.csv не найден` | Не запускали подготовку данных | `python prepare_data.py` |
| `MLflow недоступен` | Сервер ещё ставит пакеты | Подождать и повторить `python train.py` |
| `git push` отклоняет `.github/workflows/ci.yml` | У PAT нет scope `workflow` | Добавить scope `workflow` или залить workflow через GitHub UI |

---

## Что сдавать / что показать преподавателю

1. Скрин MLflow: run с `f1_score` и модель `production_classifier` в стадии **Staging**
2. Скрин MinIO: бакет `mlflow`
3. Вывод терминала `train.py` и `eval_gate.py`
4. (опционально) `data.csv.dvc` после `dvc add`
