# Defines the dataset schema for the fraud detection platform.

# This module contains the feature groups, target column, ID columns,
# and complete column structure used across data generation, preprocessing,
# model training, API inference, and monitoring pipelines.

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FraudDatasetSchema:
    target_column: str = "is_fraud"

    numerical_features: List[str] = None
    categorical_features: List[str] = None
    id_columns: List[str] = None
    metadata_columns: List[str] = None
    all_columns: List[str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "numerical_features",
            [
                "user_age",
                "account_age_days",
                "transaction_amount",
                "transaction_hour",
                "ip_risk_score",
                "num_prev_transactions_24h",
                "avg_transaction_amount_7d",
                "failed_login_attempts_24h",
            ],
        )

        object.__setattr__(
            self,
            "categorical_features",
            [
                "email_domain",
                "device_type",
                "payment_method",
                "country",
                "is_foreign_transaction",
                "shipping_billing_mismatch",
                "kyc_completed",
                "has_chargeback_history",
            ],
        )

        object.__setattr__(
            self,
            "id_columns",
            [
                "transaction_id",
                "account_id",
            ],
        )

        object.__setattr__(
            self,
            "metadata_columns",
            [
                "is_synthetic_account",
            ],
        )

        object.__setattr__(
            self,
            "all_columns",
            self.id_columns
            + self.numerical_features
            + self.categorical_features
            + self.metadata_columns
            + [self.target_column],
        )


SCHEMA = FraudDatasetSchema()