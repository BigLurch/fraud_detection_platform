# Validates the synthetic fraud dataset before training.

# This module checks schema consistency, missing values, duplicate IDs,
# target integrity, and basic data quality rules to ensure the dataset
# is suitable for machine learning workflows.

from pathlib import Path

import pandas as pd

from src.data.schema import SCHEMA


def load_dataset(path: str = "data/raw/synthetic_transactions.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing = set(SCHEMA.all_columns) - set(df.columns)
    extra = set(df.columns) - set(SCHEMA.all_columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if extra:
        print(f"Warning: Extra columns found: {extra}")


def validate_nulls(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    bad = null_counts[null_counts > 0]

    if not bad.empty:
        raise ValueError(f"Null values found:\n{bad}")


def validate_duplicates(df: pd.DataFrame) -> None:
    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction_id values found.")

    if df["account_id"].duplicated().sum() > len(df) * 0.95:
        print("Warning: Many unique accounts generated (expected in simulation).")


def validate_target(df: pd.DataFrame) -> None:
    valid_values = {0, 1}

    if not set(df["is_fraud"].unique()).issubset(valid_values):
        raise ValueError("Target column contains invalid values.")


def validate_ranges(df: pd.DataFrame) -> None:
    if (df["transaction_amount"] < 0).any():
        raise ValueError("Negative transaction amounts found.")

    if (df["ip_risk_score"] < 0).any() or (df["ip_risk_score"] > 1).any():
        raise ValueError("ip_risk_score must be between 0 and 1.")

    if (df["transaction_hour"] < 0).any() or (df["transaction_hour"] > 23).any():
        raise ValueError("transaction_hour must be between 0 and 23.")


def run_validation(path: str = "data/raw/synthetic_transactions.csv") -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = load_dataset(path)

    validate_columns(df)
    validate_nulls(df)
    validate_duplicates(df)
    validate_target(df)
    validate_ranges(df)

    print("Dataset validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    run_validation()