# Provides model loading and prediction logic for the fraud detection API.

# This module loads the trained model artifact, builds engineered features
# for incoming transactions, generates predictions, and maps probabilities
# to human-readable risk labels.

from pathlib import Path

import joblib
import pandas as pd

from src.api.schemas import TransactionRequest
from src.features.build_features import create_features


MODEL_PATH = "artifacts/models/fraud_model.joblib"


def load_model(path: str = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Train the model first with: python -m src.models.train"
        )

    return joblib.load(path)


model = load_model()


def get_risk_label(probability: float) -> str:
    if probability >= 0.80:
        return "high"
    if probability >= 0.50:
        return "medium"
    return "low"


def predict_transaction(payload: TransactionRequest) -> dict:
    input_data = pd.DataFrame([payload.model_dump()])

    input_data["transaction_id"] = "API_REQUEST"
    input_data["account_id"] = "API_ACCOUNT"
    input_data["is_synthetic_account"] = "unknown"
    input_data["is_fraud"] = 0

    features = create_features(input_data)

    features = features.drop(
        columns=[
            "transaction_id",
            "account_id",
            "is_synthetic_account",
            "is_fraud",
        ]
    )

    fraud_probability = float(model.predict_proba(features)[:, 1][0])
    prediction = int(fraud_probability >= 0.5)
    risk_label = get_risk_label(fraud_probability)

    return {
        "prediction": prediction,
        "fraud_probability": round(fraud_probability, 4),
        "risk_label": risk_label,
    }