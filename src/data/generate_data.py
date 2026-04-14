from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import SCHEMA


RANDOM_SEED = 42
N_SAMPLES = 8000
FRAUD_RATE = 0.10


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_account_id(index: int) -> str:
    return f"ACC_{100000 + index}"


def generate_transaction_id(index: int) -> str:
    return f"TXN_{500000 + index}"


def pick_email_domain(is_synthetic: bool) -> str:
    legit_domains = ["gmail.com", "outlook.com", "icloud.com", "hotmail.com"]
    risky_domains = ["tempmail.io", "fastmailbox.net", "quickdrop.cc", "burnermail.xyz"]

    if is_synthetic:
        return random.choices(
            population=risky_domains + legit_domains,
            weights=[0.35, 0.25, 0.20, 0.10, 0.04, 0.03, 0.02, 0.01],
            k=1,
        )[0]

    return random.choice(legit_domains)


def pick_country(is_fraud: bool) -> str:
    safe_countries = ["Sweden", "Norway", "Denmark", "Finland", "Germany", "Netherlands"]
    risky_countries = ["Poland", "Romania", "Nigeria", "Russia", "Turkey", "Thailand"]

    if is_fraud:
        return random.choices(
            population=safe_countries + risky_countries,
            weights=[0.10, 0.10, 0.08, 0.08, 0.08, 0.06, 0.12, 0.10, 0.12, 0.08, 0.04, 0.04],
            k=1,
        )[0]

    return random.choices(
        population=safe_countries + risky_countries,
        weights=[0.35, 0.15, 0.10, 0.10, 0.12, 0.10, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005],
        k=1,
    )[0]


def generate_legit_profile() -> dict:
    user_age = int(np.clip(np.random.normal(35, 10), 18, 75))
    account_age_days = int(np.clip(np.random.normal(420, 250), 30, 2500))
    avg_transaction_amount_7d = float(np.clip(np.random.normal(850, 350), 80, 5000))
    transaction_amount = float(np.clip(np.random.normal(avg_transaction_amount_7d, 300), 30, 7000))
    transaction_hour = int(np.clip(np.random.normal(14, 5), 0, 23))
    ip_risk_score = float(np.clip(np.random.normal(0.18, 0.10), 0.01, 0.75))
    num_prev_transactions_24h = int(np.clip(np.random.poisson(2), 0, 8))
    failed_login_attempts_24h = int(np.clip(np.random.poisson(0.4), 0, 3))

    return {
        "user_age": user_age,
        "account_age_days": account_age_days,
        "transaction_amount": round(transaction_amount, 2),
        "transaction_hour": transaction_hour,
        "ip_risk_score": round(ip_risk_score, 3),
        "num_prev_transactions_24h": num_prev_transactions_24h,
        "avg_transaction_amount_7d": round(avg_transaction_amount_7d, 2),
        "failed_login_attempts_24h": failed_login_attempts_24h,
        "email_domain": pick_email_domain(is_synthetic=False),
        "device_type": random.choices(
            ["mobile", "desktop", "tablet"],
            weights=[0.55, 0.35, 0.10],
            k=1,
        )[0],
        "payment_method": random.choices(
            ["card", "apple_pay", "google_pay", "bank_transfer"],
            weights=[0.55, 0.20, 0.15, 0.10],
            k=1,
        )[0],
        "country": pick_country(is_fraud=False),
        "is_foreign_transaction": random.choices(["no", "yes"], weights=[0.92, 0.08], k=1)[0],
        "shipping_billing_mismatch": random.choices(["no", "yes"], weights=[0.94, 0.06], k=1)[0],
        "kyc_completed": random.choices(["yes", "no"], weights=[0.96, 0.04], k=1)[0],
        "has_chargeback_history": random.choices(["no", "yes"], weights=[0.93, 0.07], k=1)[0],
        "is_synthetic_account": "no",
    }


def generate_fraud_profile() -> dict:
    synthetic_account = random.choices(["yes", "no"], weights=[0.65, 0.35], k=1)[0]
    account_takeover_pattern = synthetic_account == "no"

    if synthetic_account == "yes":
        user_age = int(np.clip(np.random.normal(27, 8), 18, 60))
        account_age_days = int(np.clip(np.random.normal(18, 12), 1, 90))
        avg_transaction_amount_7d = float(np.clip(np.random.normal(550, 220), 50, 2500))
        transaction_amount = float(np.clip(np.random.normal(3800, 2200), 250, 15000))
        failed_login_attempts_24h = int(np.clip(np.random.poisson(2), 0, 8))
        num_prev_transactions_24h = int(np.clip(np.random.poisson(6), 1, 20))
        ip_risk_score = float(np.clip(np.random.normal(0.78, 0.15), 0.35, 1.00))
        transaction_hour = random.choice([0, 1, 2, 3, 4, 22, 23])
        email_domain = pick_email_domain(is_synthetic=True)
        kyc_completed = random.choices(["no", "yes"], weights=[0.75, 0.25], k=1)[0]
        has_chargeback_history = random.choices(["yes", "no"], weights=[0.55, 0.45], k=1)[0]
        is_foreign_transaction = random.choices(["yes", "no"], weights=[0.80, 0.20], k=1)[0]
        shipping_billing_mismatch = random.choices(["yes", "no"], weights=[0.70, 0.30], k=1)[0]
    else:
        user_age = int(np.clip(np.random.normal(38, 10), 18, 75))
        account_age_days = int(np.clip(np.random.normal(620, 350), 60, 2500))
        avg_transaction_amount_7d = float(np.clip(np.random.normal(900, 300), 100, 4000))
        transaction_amount = float(np.clip(np.random.normal(5200, 2500), 300, 18000))
        failed_login_attempts_24h = int(np.clip(np.random.poisson(4), 1, 10))
        num_prev_transactions_24h = int(np.clip(np.random.poisson(4), 1, 15))
        ip_risk_score = float(np.clip(np.random.normal(0.67, 0.18), 0.20, 1.00))
        transaction_hour = random.choice([0, 1, 2, 3, 4, 23])
        email_domain = pick_email_domain(is_synthetic=False)
        kyc_completed = "yes"
        has_chargeback_history = random.choices(["yes", "no"], weights=[0.30, 0.70], k=1)[0]
        is_foreign_transaction = random.choices(["yes", "no"], weights=[0.72, 0.28], k=1)[0]
        shipping_billing_mismatch = random.choices(["yes", "no"], weights=[0.45, 0.55], k=1)[0]

    return {
        "user_age": user_age,
        "account_age_days": account_age_days,
        "transaction_amount": round(transaction_amount, 2),
        "transaction_hour": transaction_hour,
        "ip_risk_score": round(ip_risk_score, 3),
        "num_prev_transactions_24h": num_prev_transactions_24h,
        "avg_transaction_amount_7d": round(avg_transaction_amount_7d, 2),
        "failed_login_attempts_24h": failed_login_attempts_24h,
        "email_domain": email_domain,
        "device_type": random.choices(
            ["mobile", "desktop", "tablet"],
            weights=[0.45, 0.45, 0.10],
            k=1,
        )[0],
        "payment_method": random.choices(
            ["card", "apple_pay", "google_pay", "bank_transfer"],
            weights=[0.72, 0.08, 0.05, 0.15],
            k=1,
        )[0],
        "country": pick_country(is_fraud=True),
        "is_foreign_transaction": is_foreign_transaction,
        "shipping_billing_mismatch": shipping_billing_mismatch,
        "kyc_completed": kyc_completed,
        "has_chargeback_history": has_chargeback_history,
        "is_synthetic_account": synthetic_account if not account_takeover_pattern else "no",
    }


def create_dataset(
    n_samples: int = N_SAMPLES,
    fraud_rate: float = FRAUD_RATE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    set_random_seed(random_seed)

    fraud_count = int(n_samples * fraud_rate)
    legit_count = n_samples - fraud_count

    rows = []

    for i in range(legit_count):
        row = generate_legit_profile()
        row["transaction_id"] = generate_transaction_id(i)
        row["account_id"] = generate_account_id(i)
        row["is_fraud"] = 0
        rows.append(row)

    for i in range(fraud_count):
        row = generate_fraud_profile()
        row["transaction_id"] = generate_transaction_id(legit_count + i)
        row["account_id"] = generate_account_id(legit_count + i)
        row["is_fraud"] = 1
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return df[SCHEMA.all_columns]


def save_dataset(df: pd.DataFrame, output_path: str = "data/raw/synthetic_transactions.csv") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\nDataset summary")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts())
    print("\nSynthetic account distribution:")
    print(df["is_synthetic_account"].value_counts())
    print("\nPreview:")
    print(df.head())


def main() -> None:
    df = create_dataset()
    save_dataset(df)
    print_dataset_summary(df)


if __name__ == "__main__":
    main()