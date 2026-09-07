"""Quality Gate: проверка f1_score и перевод модели в Staging."""

import os
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "production_classifier"
F1_THRESHOLD = 0.80


def configure_environment() -> None:
    load_dotenv()
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


def wait_for_mlflow(retries: int = 15, delay_sec: float = 2.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            mlflow.search_experiments()
            print("MLflow сервер доступен")
            return
        except Exception as exc:  # noqa: BLE001 — ждём готовности сервиса
            print(f"Ожидание MLflow ({attempt}/{retries}): {exc}")
            time.sleep(delay_sec)
    print("MLflow недоступен. Проверьте docker-compose и адрес http://localhost:5000")
    sys.exit(1)


def get_latest_model_version(client: MlflowClient):
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print(f"Модель '{REGISTERED_MODEL_NAME}' не найдена в Model Registry")
        sys.exit(1)
    latest = max(versions, key=lambda version: int(version.version))
    print(f"Последняя версия модели: v{latest.version} (run_id={latest.run_id})")
    return latest


def evaluate_gate() -> None:
    configure_environment()
    wait_for_mlflow()

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest = get_latest_model_version(client)

    run = client.get_run(latest.run_id)
    score = run.data.metrics.get("f1_score")
    if score is None:
        print(f"Метрика f1_score отсутствует в run {latest.run_id}")
        sys.exit(1)

    print(f"f1_score = {score:.4f}, порог = {F1_THRESHOLD:.2f}")
    if score >= F1_THRESHOLD:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest.version,
            stage="Staging",
        )
        print(
            f"Quality Gate пройден: '{REGISTERED_MODEL_NAME}' "
            f"v{latest.version} переведена в Staging"
        )
        sys.exit(0)

    print(
        f"Quality Gate не пройден: f1_score {score:.4f} < {F1_THRESHOLD:.2f}. "
        "Модель не переведена в Staging"
    )
    sys.exit(1)


if __name__ == "__main__":
    try:
        evaluate_gate()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Ошибка Quality Gate: {exc}")
        sys.exit(1)
