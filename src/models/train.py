# Trains the fraud detection model.

# This module loads engineered features, splits the dataset,
# builds a preprocessing + model pipeline, evaluates performance,
# and saves the trained model artifact.

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor
from src.models.evaluate import evaluate_model, save_metrics


DATA_PATH = "data/processed/train_ready.csv"
MODEL_PATH = "artifacts/models/fraud_model.joblib"
TARGET = "is_fraud"
RANDOM_SEED = 42


def load_data(path: str = DATA_PATH):
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, "transaction_id", "account_id"])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )


def build_pipeline():
    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def save_model(model, path: str = MODEL_PATH):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    print("Loading data...")
    df = load_data()

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    save_metrics(metrics)
    save_model(pipeline)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()