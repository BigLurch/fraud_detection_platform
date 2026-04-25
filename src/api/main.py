# Exposes FastAPI endpoints for fraud detection inference.

# This module provides health checks and prediction endpoints for serving
# the trained fraud detection model as a lightweight API service.

from fastapi import FastAPI

from src.api.schemas import PredictionResponse, TransactionRequest
from src.api.service import predict_transaction


app = FastAPI(
    title="Fraud Detection API",
    description="Inference API for detecting suspicious payment transactions.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "fraud-detection-api"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionRequest):
    return predict_transaction(payload)