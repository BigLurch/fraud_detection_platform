# Provides a Streamlit dashboard for fraud detection.

# This module creates a lightweight user interface for submitting transaction
# data to the FastAPI inference service, displaying fraud risk results,
# and reviewing recent prediction logs.

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"
LOG_PATH = "artifacts/logs/predictions.jsonl"


st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
)


def load_prediction_logs(path: str = LOG_PATH) -> pd.DataFrame:
    log_path = Path(path)

    if not log_path.exists():
        return pd.DataFrame()

    records = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def call_prediction_api(payload: dict) -> dict:
    response = requests.post(API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def render_risk_result(result: dict) -> None:
    probability = result["fraud_probability"]
    risk_label = result["risk_label"]
    prediction = result["prediction"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud probability", f"{probability:.2%}")
    col2.metric("Risk level", risk_label.upper())
    col3.metric("Prediction", "FRAUD" if prediction == 1 else "LEGIT")

    if risk_label == "high":
        st.error("High risk transaction detected. Manual review recommended.")
    elif risk_label == "medium":
        st.warning("Medium risk transaction. Additional verification may be needed.")
    else:
        st.success("Low risk transaction. No immediate action required.")


def main() -> None:
    st.title("🛡️ Fraud Detection Platform")
    st.caption("Modern payment fraud detection demo with FastAPI, MLflow and Streamlit.")

    st.divider()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Transaction Check")

        user_age = st.slider("User age", 18, 100, 32)
        account_age_days = st.number_input("Account age days", min_value=0, value=30)
        transaction_amount = st.number_input(
            "Transaction amount",
            min_value=1.0,
            value=2500.0,
            step=100.0,
        )
        transaction_hour = st.slider("Transaction hour", 0, 23, 14)
        ip_risk_score = st.slider("IP risk score", 0.0, 1.0, 0.45, 0.01)
        num_prev_transactions_24h = st.number_input(
            "Previous transactions last 24h",
            min_value=0,
            value=2,
        )
        avg_transaction_amount_7d = st.number_input(
            "Average transaction amount 7d",
            min_value=1.0,
            value=850.0,
            step=50.0,
        )
        failed_login_attempts_24h = st.number_input(
            "Failed login attempts last 24h",
            min_value=0,
            value=0,
        )

    with right_col:
        st.subheader("Context")

        email_domain = st.selectbox(
            "Email domain",
            ["gmail.com", "outlook.com", "icloud.com", "hotmail.com", "tempmail.io", "burnermail.xyz"],
        )
        device_type = st.selectbox("Device type", ["mobile", "desktop", "tablet"])
        payment_method = st.selectbox(
            "Payment method",
            ["card", "apple_pay", "google_pay", "bank_transfer"],
        )
        country = st.selectbox(
            "Country",
            [
                "Sweden",
                "Norway",
                "Denmark",
                "Finland",
                "Germany",
                "Netherlands",
                "Poland",
                "Romania",
                "Nigeria",
                "Turkey",
            ],
        )
        is_foreign_transaction = st.selectbox("Foreign transaction", ["no", "yes"])
        shipping_billing_mismatch = st.selectbox("Shipping/billing mismatch", ["no", "yes"])
        kyc_completed = st.selectbox("KYC completed", ["yes", "no"])
        has_chargeback_history = st.selectbox("Chargeback history", ["no", "yes"])

    payload = {
        "user_age": user_age,
        "account_age_days": int(account_age_days),
        "transaction_amount": float(transaction_amount),
        "transaction_hour": transaction_hour,
        "ip_risk_score": float(ip_risk_score),
        "num_prev_transactions_24h": int(num_prev_transactions_24h),
        "avg_transaction_amount_7d": float(avg_transaction_amount_7d),
        "failed_login_attempts_24h": int(failed_login_attempts_24h),
        "email_domain": email_domain,
        "device_type": device_type,
        "payment_method": payment_method,
        "country": country,
        "is_foreign_transaction": is_foreign_transaction,
        "shipping_billing_mismatch": shipping_billing_mismatch,
        "kyc_completed": kyc_completed,
        "has_chargeback_history": has_chargeback_history,
    }

    st.divider()

    if st.button("Run Fraud Check", type="primary"):
        try:
            result = call_prediction_api(payload)
            render_risk_result(result)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Make sure the API is running on port 8000.")
        except requests.exceptions.RequestException as error:
            st.error(f"Prediction request failed: {error}")

    st.divider()

    st.subheader("Recent Prediction Logs")

    logs_df = load_prediction_logs()

    if logs_df.empty:
        st.info("No prediction logs found yet. Run a fraud check to create logs.")
    else:
        display_columns = [
            "timestamp",
            "transaction_amount",
            "country",
            "ip_risk_score",
            "fraud_probability",
            "risk_label",
            "prediction",
        ]

        available_columns = [col for col in display_columns if col in logs_df.columns]
        st.dataframe(
            logs_df[available_columns].tail(10).sort_index(ascending=False),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()