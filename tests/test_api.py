# Tests for the FastAPI fraud prediction service.

# These tests verify that the API health endpoint works and that the prediction
# endpoint returns the expected response structure.

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_valid_response():
    payload = {
        "user_age": 42,
        "account_age_days": 850,
        "transaction_amount": 349.0,
        "transaction_hour": 14,
        "ip_risk_score": 0.12,
        "num_prev_transactions_24h": 1,
        "avg_transaction_amount_7d": 520.0,
        "failed_login_attempts_24h": 0,
        "email_domain": "gmail.com",
        "device_type": "mobile",
        "payment_method": "apple_pay",
        "country": "Sweden",
        "is_foreign_transaction": "no",
        "shipping_billing_mismatch": "no",
        "kyc_completed": "yes",
        "has_chargeback_history": "no",
        "source": "test",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "fraud_probability" in data
    assert "risk_label" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_label"] in ["low", "medium", "high"]