# Builds preprocessing pipelines for fraud model training.

# This module prepares numerical and categorical features using
# scaling and one-hot encoding inside a reusable sklearn pipeline.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import SCHEMA


def get_feature_columns():
    numerical_features = SCHEMA.numerical_features + [
        "amount_to_avg_ratio",
        "is_night_transaction",
        "high_velocity_flag",
        "new_account_flag",
        "risky_email_domain",
        "foreign_high_amount_flag",
        "login_risk_flag",
    ]

    categorical_features = SCHEMA.categorical_features

    return numerical_features, categorical_features


def build_preprocessor():
    numerical_features, categorical_features = get_feature_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return preprocessor