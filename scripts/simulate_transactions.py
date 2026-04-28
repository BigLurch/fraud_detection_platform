# Simulates live transaction traffic for the fraud detection platform.

# This script generates random legitimate, suspicious, and high-risk transactions,
# sends them to the FastAPI prediction endpoint, and lets the API log predictions
# for the Streamlit dashboard and monitoring pipeline.

import os
import random
import time

import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


def get_transaction_preset(scenario: str) -> dict:
    presets = {
        "legit": {
            "user_age": random.randint(25, 65),
            "account_age_days": random.randint(300, 1600),
            "transaction_amount": round(random.uniform(100, 1200), 2),
            "transaction_hour": random.randint(8, 20),
            "ip_risk_score": round(random.uniform(0.02, 0.25), 2),
            "num_prev_transactions_24h": random.randint(0, 3),
            "avg_transaction_amount_7d": round(random.uniform(300, 1000), 2),
            "failed_login_attempts_24h": random.randint(0, 1),
            "email_domain": random.choice(["gmail.com", "outlook.com", "icloud.com"]),
            "device_type": random.choice(["mobile", "desktop", "tablet"]),
            "payment_method": random.choice(["card", "apple_pay", "google_pay"]),
            "country": random.choice(["Sweden", "Norway", "Denmark", "Finland", "Germany"]),
            "is_foreign_transaction": "no",
            "shipping_billing_mismatch": "no",
            "kyc_completed": "yes",
            "has_chargeback_history": "no",
        },
        "suspicious": {
            "user_age": random.randint(20, 55),
            "account_age_days": random.randint(20, 180),
            "transaction_amount": round(random.uniform(1500, 5500), 2),
            "transaction_hour": random.choice([0, 1, 2, 3, 22, 23]),
            "ip_risk_score": round(random.uniform(0.40, 0.70), 2),
            "num_prev_transactions_24h": random.randint(3, 7),
            "avg_transaction_amount_7d": round(random.uniform(500, 1500), 2),
            "failed_login_attempts_24h": random.randint(1, 3),
            "email_domain": random.choice(["gmail.com", "outlook.com", "tempmail.io"]),
            "device_type": random.choice(["mobile", "desktop"]),
            "payment_method": random.choice(["card", "bank_transfer"]),
            "country": random.choice(["Poland", "Romania", "Germany", "Netherlands"]),
            "is_foreign_transaction": random.choice(["yes", "no"]),
            "shipping_billing_mismatch": random.choice(["yes", "no"]),
            "kyc_completed": random.choice(["yes", "no"]),
            "has_chargeback_history": random.choice(["yes", "no"]),
        },
        "fraud": {
            "user_age": random.randint(18, 45),
            "account_age_days": random.randint(1, 30),
            "transaction_amount": round(random.uniform(5000, 12000), 2),
            "transaction_hour": random.choice([0, 1, 2, 3, 4, 23]),
            "ip_risk_score": round(random.uniform(0.75, 0.98), 2),
            "num_prev_transactions_24h": random.randint(6, 14),
            "avg_transaction_amount_7d": round(random.uniform(300, 1000), 2),
            "failed_login_attempts_24h": random.randint(3, 8),
            "email_domain": random.choice(["tempmail.io", "burnermail.xyz", "quickdrop.cc"]),
            "device_type": random.choice(["desktop", "mobile"]),
            "payment_method": random.choice(["card", "bank_transfer"]),
            "country": random.choice(["Nigeria", "Turkey", "Romania", "Poland"]),
            "is_foreign_transaction": "yes",
            "shipping_billing_mismatch": "yes",
            "kyc_completed": "no",
            "has_chargeback_history": random.choice(["yes", "no"]),
        },
    }

    return presets[scenario]


def choose_scenario() -> str:
    return random.choices(
        population=["legit", "suspicious", "fraud"],
        weights=[0.70, 0.20, 0.10],
        k=1,
    )[0]


def send_transaction(payload: dict) -> None:
    payload["source"] = "simulator"
    response = requests.post(API_URL, json=payload, timeout=10)
    response.raise_for_status()

    result = response.json()

    print(
        f"{payload['country']:>11} | "
        f"{payload['transaction_amount']:>8.2f} | "
        f"prob={result['fraud_probability']:.2%} | "
        f"risk={result['risk_label']}"
    )


def main() -> None:
    print("Starting live transaction simulator...")
    print(f"Sending traffic to: {API_URL}")
    print("Press Ctrl+C to stop.\n")

    while True:
        scenario = choose_scenario()
        payload = get_transaction_preset(scenario)

        try:
            send_transaction(payload)
        except requests.exceptions.RequestException as error:
            print(f"Request failed: {error}")

        time.sleep(random.uniform(1.0, 3.0))


if __name__ == "__main__":
    main()