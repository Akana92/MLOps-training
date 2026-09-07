"""Генерация синтетического датасета классификации для практикума."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

OUTPUT_PATH = Path("data.csv")
N_SAMPLES = 1000
N_FEATURES = 10
RANDOM_STATE = 42


def prepare_data() -> Path:
    print(f"Генерация датасета: {N_SAMPLES} строк, {N_FEATURES} признаков...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_STATE,
    )

    columns = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Датасет сохранён: {OUTPUT_PATH.resolve()} ({len(df)} строк)")
    return OUTPUT_PATH


if __name__ == "__main__":
    try:
        prepare_data()
    except OSError as exc:
        print(f"Ошибка записи датасета: {exc}")
        raise SystemExit(1) from exc
