# Tests for fraud feature engineering.

# These tests verify that engineered fraud-risk features are created correctly
# from raw transaction data.

from src.data.generate_data import create_dataset
from src.features.build_features import create_features


def test_create_features_adds_expected_columns():
    df = create_dataset(n_samples=100, fraud_rate=0.1, random_seed=42)
    featured_df = create_features(df)

    expected_columns = {
        "amount_to_avg_ratio",
        "is_night_transaction",
        "high_velocity_flag",
        "new_account_flag",
        "risky_email_domain",
        "foreign_high_amount_flag",
        "login_risk_flag",
    }

    assert expected_columns.issubset(featured_df.columns)


def test_amount_to_avg_ratio_is_positive():
    df = create_dataset(n_samples=100, fraud_rate=0.1, random_seed=42)
    featured_df = create_features(df)

    assert (featured_df["amount_to_avg_ratio"] > 0).all()


def test_binary_engineered_features_are_binary():
    df = create_dataset(n_samples=100, fraud_rate=0.1, random_seed=42)
    featured_df = create_features(df)

    binary_columns = [
        "is_night_transaction",
        "high_velocity_flag",
        "new_account_flag",
        "risky_email_domain",
        "foreign_high_amount_flag",
        "login_risk_flag",
    ]

    for column in binary_columns:
        assert set(featured_df[column].unique()).issubset({0, 1})