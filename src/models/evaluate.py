# Evaluates fraud classification models.

# This module calculates common fraud detection metrics such as
# precision, recall, F1 score, ROC-AUC, and confusion matrix.

import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }

    print("\nEvaluation Metrics")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    return metrics


def save_metrics(
    metrics: dict,
    path: str = "artifacts/metrics/train_metrics.json",
):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {path}")