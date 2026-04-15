# Builds engineered features for fraud model training.

# This module transforms raw transaction data into model-ready features
# by creating behavioral risk indicators, ratios, and categorical flags
# that improve fraud detection performance.

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: str = "data/raw/synthetic_transactions.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio between current transaction and user average
    df["amount_to_avg_ratio"] = (
        df["transaction_amount"] / (df["avg_transaction_amount_7d"] + 1)
    )

    # Night transaction flag
    df["is_night_transaction"] = df["transaction_hour"].apply(
        lambda x: 1 if x in [0, 1, 2, 3, 4, 23] else 0
    )

    # High velocity flag
    df["high_velocity_flag"] = df["num_prev_transactions_24h"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    # New account flag
    df["new_account_flag"] = df["account_age_days"].apply(
        lambda x: 1 if x <= 30 else 0
    )

    # Risky email domain flag
    risky_domains = [
        "tempmail.io",
        "fastmailbox.net",
        "quickdrop.cc",
        "burnermail.xyz",
    ]

    df["risky_email_domain"] = df["email_domain"].apply(
        lambda x: 1 if x in risky_domains else 0
    )

    # Foreign + high amount combo
    df["foreign_high_amount_flag"] = np.where(
        (df["is_foreign_transaction"] == "yes")
        & (df["transaction_amount"] > 3000),
        1,
        0,
    )

    # Failed login risk
    df["login_risk_flag"] = df["failed_login_attempts_24h"].apply(
        lambda x: 1 if x >= 3 else 0
    )

    return df


def save_processed_data(
    df: pd.DataFrame,
    path: str = "data/processed/train_ready.csv",
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def main() -> None:
    df = load_dataset()
    df = create_features(df)
    save_processed_data(df)

    print("Feature engineering complete.")
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()