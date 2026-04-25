# Defines request and response schemas for the fraud prediction API.

# This module uses Pydantic models to validate incoming transaction data
# and structure prediction responses returned by the FastAPI service.

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    user_age: int = Field(..., ge=18, le=100)
    account_age_days: int = Field(..., ge=0)
    transaction_amount: float = Field(..., gt=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    ip_risk_score: float = Field(..., ge=0, le=1)
    num_prev_transactions_24h: int = Field(..., ge=0)
    avg_transaction_amount_7d: float = Field(..., gt=0)
    failed_login_attempts_24h: int = Field(..., ge=0)

    email_domain: str
    device_type: str
    payment_method: str
    country: str
    is_foreign_transaction: str
    shipping_billing_mismatch: str
    kyc_completed: str
    has_chargeback_history: str


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    risk_label: str