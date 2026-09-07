# MLOps-training

Учебный проект: версия датасета в DVC/MinIO → обучение Random Forest → метрики и версии моделей в MLflow → Quality Gate → локальный сервис предсказаний. Основан на [Shahnurhon/MLOps](https://github.com/Shahnurhon/MLOps). Методичка сохранена в [docs/Gude.md](docs/Gude.md), результаты учебных запусков — в [docs/RESULTS.md](docs/RESULTS.md).

## Что хранится где

| Хранилище | Содержимое |
|---|---|
| GitHub | Исходники, Docker Compose, workflow, инструкции, конфигурация DVC и `data.csv.dvc` |
| MinIO, бакет `mlflow`, префикс `dvc/` | Байты датасета, адресуемые по хешу DVC |
| MinIO, остальные префиксы бакета `mlflow` | Артефакты обученных моделей MLflow |
| PostgreSQL | Эксперименты, параметры, метрики, run ID и Model Registry |
| Локальный компьютер | Восстановленный `data.csv`, виртуальное окружение, кеш DVC, локальные настройки |

**Сам `data.csv` в Git не загружается.** `dvc add data.csv` вычисляет хеш локального файла и создаёт небольшой указатель `data.csv.dvc`; `dvc push` отправляет данные в MinIO. После клонирования `dvc pull` читает указатель из Git, получает соответствующий объект из MinIO и восстанавливает имя `data.csv`.

Текущий указатель: MD5 `9eebf690878c4c771dff95214d7f4fb0`, размер 196 763 байта. Объект находится в `s3://mlflow/dvc/files/md5/9e/ebf690878c4c771dff95214d7f4fb0`. Датасет содержит 1000 записей, 10 признаков `feature_0`…`feature_9` и целевой столбец `target`. В MinIO объект называется по хешу, поэтому искать файл с именем `data.csv` в интерфейсе не нужно.

## Требования и адреса

- Windows, PowerShell, Git, Python 3.11.
- Запущенный Docker Desktop с Linux containers и Docker Compose.
- Доступ к MinIO, в котором уже хранится версия данных из `data.csv.dvc`.

| Сервис | Адрес |
|---|---|
| MLflow | http://localhost:5000 |
| MinIO API для DVC/MLflow | http://localhost:9000 |
| MinIO Console | http://localhost:9001 |
| Сервис предсказаний после deploy | http://localhost:5002 |

В исходной методичке встречается порт **5001**. В этой конфигурации MLflow использует **5000**; команды ниже соответствуют `docker-compose.yml`.

## Воспроизведение в PowerShell

### 1. Получить проект и запустить инфраструктуру

```powershell
git clone https://github.com/Akana92/MLOps-training.git
cd MLOps-training
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Заполните в `.env` значения `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` и `POSTGRES_PASSWORD`, затем сохраните файл. Для существующих томов укажите **те же значения, что использует ваш действующий стенд**: смена переменной не меняет пароль уже созданной базы PostgreSQL. Для нового стенда задайте собственные длинные буквенно-цифровые значения. Не коммитьте `.env`: он исключён из Git; публикуется только пустой `.env.example`.

```powershell
docker compose -p mlops up -d
docker compose -p mlops ps
```

Явное имя Compose-проекта `mlops` сохраняет связь с существующими томами учебного стенда. При первом запуске MLflow устанавливает дополнительные пакеты; дождитесь доступности интерфейса. На компьютере с уже запущенным стендом используются существующие контейнеры и данные. Не запускайте второй экземпляр с теми же портами/именами и не выполняйте `down -v`, если нужно сохранить базу и бакет.

### 2. Создать окружение и задать подключение

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip --use-feature=truststore
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
$env:MLFLOW_TRACKING_URI = 'http://localhost:5000'
$env:MLFLOW_S3_ENDPOINT_URL = 'http://localhost:9000'
.\.venv\Scripts\python.exe -c "import mlflow; print(mlflow.__version__)"
```

Ожидается MLflow **2.7.1**, совпадающий с сервером. `requirements.txt` задаёт основные пакеты, а `constraints.txt` закрепляет совместимые версии ключевых библиотек для обучения и сервиса. `requirements-local.lock.txt` — дополнительный полный снимок прежнего Windows-окружения; он служит справкой о том окружении, а актуальные дополнения, включая python-dotenv, устанавливаются основной командой выше. Для другой ОС набор зависимостей нужно проверить отдельно.

Docker Compose, `train.py` и `eval_gate.py` читают локальный `.env`. DVC CLI не загружает его автоматически, поэтому ниже запускается через `python -m dotenv run`. Вместо файла можно задать переменные окружения процесса. Ключи и пароли не коммитьте. `.dvc/config` содержит адрес remote без учётных данных; `.dvc/config.local` исключён из Git. Адрес API — порт 9000, а не порт консоли 9001.

### 3. Восстановить датасет из MinIO

```powershell
.\.venv\Scripts\python.exe -m dotenv run -- .\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\python.exe -m dotenv run -- .\.venv\Scripts\dvc.exe status -c
Get-FileHash data.csv -Algorithm MD5
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('data.csv'); print(d.shape); print(d.columns.tolist())"
git ls-files data.csv data.csv.dvc
```

Ожидаются хеш из указателя, размер таблицы `(1000, 11)` и только `data.csv.dvc` в выводе `git ls-files`. `dvc pull` требует исходный объект в MinIO: новый пустой бакет не восстановит датасет по одному хешу. На другом компьютере `localhost` обозначает другой MinIO; нужен доступ к исходному серверу и локальная настройка endpoint:

```powershell
.\.venv\Scripts\dvc.exe remote modify --local minio endpointurl http://ADDRESS:9000
$env:MLFLOW_S3_ENDPOINT_URL = 'http://ADDRESS:9000'
```

### 4. Обучить и проверить модель

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe eval_gate.py
```

Обучение создаёт новый run в эксперименте `mlops-workshop`, логирует параметры и `f1_score`, сохраняет модель в MinIO и регистрирует новую версию `production_classifier`. Quality Gate допускает модель при **F1 ≥ 0,80** и переводит её в **Staging**. При одинаковых данных и параметрах новая версия может иметь прежнюю метрику: номер версии отражает регистрацию, а не обязательное улучшение качества.

Отдельный `eval_gate.py` из исходного практикума выбирает последнюю версию: запускайте эту пару команд последовательно, без параллельных обучений. Автоматизированный сценарий ниже привязывает проверку к конкретному run ID.

Откройте MLflow → Experiments → `mlops-workshop`, затем Chart для сравнения `f1_score`; версии смотрите в Models → `production_classifier`.

## Первичная загрузка или новая версия данных

Этот сценарий нужен автору датасета при его создании/изменении. При обычном воспроизведении используйте `dvc pull` выше, не генерируйте данные заново.

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\dvc.exe add data.csv
.\.venv\Scripts\python.exe -m dotenv run -- .\.venv\Scripts\dvc.exe push
.\.venv\Scripts\python.exe -m dotenv run -- .\.venv\Scripts\dvc.exe status -c
git add data.csv.dvc .gitignore
git diff --cached --name-only
git commit -m "Update DVC dataset version"
git push
```

Для `dvc push` бакет `mlflow` должен существовать: при пустом стенде создайте его в MinIO Console. DVC уже инициализирован в репозитории, повторные `dvc init` и `dvc remote add` не нужны. Загружайте объект в MinIO **до** публикации нового указателя, иначе другие участники получат ссылку на отсутствующие данные.

## CI/CD на локальном runner

Workflow `.github/workflows/ci.yml` подготовлен для Windows self-hosted runner с меткой **`mlops-local`**. Обычный GitHub-hosted runner не имеет доступа к `localhost` вашего компьютера, где находятся MinIO и MLflow.

Полный сценарий локально проверен: восстановление DVC → обучение v6 → F1 0.8426 → Docker-сервис → проверка предсказаний → Production. Подробности в [результатах](docs/RESULTS.md). Запуск через GitHub Actions требует активации runner, описанной ниже.

Последовательность соответствует учебной схеме:

1. **Setup** — получить код, подготовить Python-окружение и доступ к локальной инфраструктуре.
2. **DVC Pull** — восстановить именно ту версию данных, которая зафиксирована в Git.
3. **Train** — обучить модель, записать run ID и номер созданной версии.
4. **Quality Gate** — проверить F1 именно этой версии; при провале остановить дальнейший deploy.
5. **Deploy** — запустить контейнер предсказаний, проверить здоровье и ответ на тестовый запрос, затем перевести проверенную версию в **Production**.

Связь по точным run ID/версии исключает случайную проверку чужого последнего запуска. Сервис использует отдельный Compose-проект `mlops-serving` с `Dockerfile.serve` и `compose.serve.yml`; инфраструктура остаётся в проекте `mlops`. Новый deploy перезапускает сервис с выбранной версией модели.

### Ручной запуск полного сценария

После настройки окружения и инфраструктуры:

```powershell
.\.venv\Scripts\python.exe -m dotenv run -- .\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\python.exe scripts/pipeline.py
Invoke-WebRequest http://localhost:5002/ping
Get-Content artifacts/pipeline-result.json
```

Сценарий создаёт новую версию модели. После успешного smoke test он переводит её в Production и архивирует прежние версии Production; результат сохраняется локально в `artifacts/pipeline-result.json`. При провале Quality Gate контейнер не обновляется. При ошибке уже начавшегося deploy контейнер мог быть заменён, хотя стадия Production ещё не изменена: автоматический откат и отсутствие простоя здесь не реализованы.

Сервис предоставляет `GET /ping` и `POST /invocations` (формат MLflow `dataframe_split`); это API, а не веб-страница. Pipeline сравнивает предсказания сервиса на пяти строках данных с результатом прямой загрузки той же зарегистрированной версии.

### Активация runner владельцем репозитория

1. В GitHub откройте **Settings → Actions → Runners → New self-hosted runner**, выберите Windows и выполните команды, которые выдаст GitHub, на компьютере с доступом к Docker/MinIO/MLflow.
2. При регистрации добавьте метку `mlops-local`. Запускайте runner под учётной записью, у которой работает Docker Desktop; интерактивный запуск подходит для практикума.
3. Установите Python 3.11, PowerShell 7 (`pwsh`) и Git на runner. Создайте обязательные repository secrets `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` и `POSTGRES_PASSWORD` со значениями действующего локального стенда. Workflow остановится, если любой из них отсутствует; запасных ключей в коде нет. Не сохраняйте регистрационный токен runner или настоящие ключи в репозитории.
4. В **Settings → Secrets and variables → Actions → Variables** создайте repository variable `MLOPS_RUNNER_READY` со значением `true` только после готовности runner.
5. Запустите workflow вручную из ветки `main` либо выполните новый push в `main`. Пока переменная готовности не включена, локальная работа не запускается.

Workflow не предназначен для исполнения непроверенного кода из pull request на вашем компьютере. В этой поставке runner **не установлен и не зарегистрирован автоматически**. Наличие workflow в Git не является доказательством прошедшего CI/CD: подтверждение — успешный конкретный запуск во вкладке Actions и доступный сервис предсказаний.

Документация: [метки self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow), [MLflow model serving](https://mlflow.org/docs/2.11.2/deployment/deploy-model-locally.html). Реальный образ проекта использует MLflow 2.7.1; команды serving проверяются на нём локально.

## Что показать преподавателю

- Репозиторий с исходниками, документацией и `data.csv.dvc`, без CSV/артефактов модели/локальных секретов.
- MinIO → бакет `mlflow` → `dvc/files/md5/9e/` с объектом датасета и отдельно артефакты MLflow.
- MLflow: несколько runs, график F1 и зарегистрированные версии со стадиями.
- После активации runner: успешные шаги Setup → DVC Pull → Train → Quality Gate → Deploy в Actions и ответ сервиса.

## Если что-то не запускается

| Симптом | Проверка |
|---|---|
| Docker daemon недоступен | Запустите Docker Desktop и проверьте `docker info` |
| `dvc pull` сообщает об отсутствующем объекте | Проверьте remote, ключи и наличие объекта в исходном MinIO; пустой бакет недостаточен |
| Access denied в MinIO | Проверьте `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` в текущем процессе |
| MLflow 404 на `/logged-models` | Проверьте, что клиент MLflow имеет версию 2.7.1 |
| Workflow пропущен | Проверьте ветку `main` и variable `MLOPS_RUNNER_READY` |
| Job ждёт runner | Runner должен быть online и иметь Windows и `mlops-local` labels |
| GitHub отклоняет push workflow | Текущая GitHub-авторизация должна разрешать изменение `.github/workflows` (для classic PAT нужен scope `workflow`) |

Учебный стенд требует локально заданные учётные данные и не рассчитан на публикацию сервисов в открытый интернет. Сброс Docker или удаление томов уничтожает локальные данные, если нет резервной копии.
