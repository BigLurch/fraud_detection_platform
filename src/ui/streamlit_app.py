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


def get_preset(name: str) -> dict:
    presets = {
        "legit": {
            "user_age": 42,
            "account_age_days": 820,
            "transaction_amount": 399.0,
            "transaction_hour": 14,
            "ip_risk_score": 0.08,
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
        },
        "medium": {
            "user_age": 31,
            "account_age_days": 45,
            "transaction_amount": 4200.0,
            "transaction_hour": 23,
            "ip_risk_score": 0.55,
            "num_prev_transactions_24h": 5,
            "avg_transaction_amount_7d": 900.0,
            "failed_login_attempts_24h": 2,
            "email_domain": "outlook.com",
            "device_type": "desktop",
            "payment_method": "card",
            "country": "Poland",
            "is_foreign_transaction": "yes",
            "shipping_billing_mismatch": "yes",
            "kyc_completed": "yes",
            "has_chargeback_history": "no",
        },
        "high": {
            "user_age": 26,
            "account_age_days": 7,
            "transaction_amount": 8900.0,
            "transaction_hour": 2,
            "ip_risk_score": 0.92,
            "num_prev_transactions_24h": 9,
            "avg_transaction_amount_7d": 500.0,
            "failed_login_attempts_24h": 5,
            "email_domain": "tempmail.io",
            "device_type": "desktop",
            "payment_method": "card",
            "country": "Nigeria",
            "is_foreign_transaction": "yes",
            "shipping_billing_mismatch": "yes",
            "kyc_completed": "no",
            "has_chargeback_history": "yes",
        },
    }

    return presets[name]


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

    st.subheader("Risk Decision")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{probability:.2%}")
    col2.metric("Risk Level", risk_label.upper())
    col3.metric("Prediction", "FRAUD" if prediction == 1 else "LEGIT")

    if risk_label == "high":
        st.error(f"🚨 HIGH RISK • {probability:.2%} fraud probability")
    elif risk_label == "medium":
        st.warning(f"⚠️ MEDIUM RISK • {probability:.2%} fraud probability")
    else:
        st.success(f"✅ LOW RISK • {probability:.2%} fraud probability")


def selectbox_with_preset(label: str, options: list[str], preset_value: str) -> str:
    index = options.index(preset_value) if preset_value in options else 0
    return st.selectbox(label, options, index=index)


def main() -> None:
    st.title("🛡️ Fraud Detection Platform")
    st.caption(
        "Modern payment fraud detection demo with FastAPI, MLflow, prediction logging and Streamlit."
    )

    logs_df = load_prediction_logs()

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Predictions", len(logs_df))
    k2.metric(
        "Fraud Alerts",
        len(logs_df[logs_df["prediction"] == 1]) if not logs_df.empty else 0,
    )
    k3.metric(
        "Avg Fraud Probability",
        f"{logs_df['fraud_probability'].mean():.2%}"
        if not logs_df.empty
        else "0%",
    )

    st.divider()

    st.subheader("Quick Scenarios")

    p1, p2, p3 = st.columns(3)

    if p1.button("✅ Legit Customer"):
        st.session_state["preset"] = get_preset("legit")

    if p2.button("⚠️ Suspicious Attempt"):
        st.session_state["preset"] = get_preset("medium")

    if p3.button("🚨 High Risk Attack"):
        st.session_state["preset"] = get_preset("high")

    preset = st.session_state.get("preset", get_preset("legit"))

    st.divider()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Transaction Check")

        user_age = st.slider("User age", 18, 100, preset["user_age"])
        account_age_days = st.number_input(
            "Account age days",
            min_value=0,
            value=preset["account_age_days"],
        )
        transaction_amount = st.number_input(
            "Transaction amount",
            min_value=1.0,
            value=preset["transaction_amount"],
            step=100.0,
        )
        transaction_hour = st.slider(
            "Transaction hour",
            0,
            23,
            preset["transaction_hour"],
        )
        ip_risk_score = st.slider(
            "IP risk score",
            0.0,
            1.0,
            preset["ip_risk_score"],
            0.01,
        )
        num_prev_transactions_24h = st.number_input(
            "Previous transactions last 24h",
            min_value=0,
            value=preset["num_prev_transactions_24h"],
        )
        avg_transaction_amount_7d = st.number_input(
            "Average transaction amount 7d",
            min_value=1.0,
            value=preset["avg_transaction_amount_7d"],
            step=50.0,
        )
        failed_login_attempts_24h = st.number_input(
            "Failed login attempts last 24h",
            min_value=0,
            value=preset["failed_login_attempts_24h"],
        )

    with right_col:
        st.subheader("Transaction Context")

        email_domain = selectbox_with_preset(
            "Email domain",
            [
                "gmail.com",
                "outlook.com",
                "icloud.com",
                "hotmail.com",
                "tempmail.io",
                "burnermail.xyz",
            ],
            preset["email_domain"],
        )
        device_type = selectbox_with_preset(
            "Device type",
            ["mobile", "desktop", "tablet"],
            preset["device_type"],
        )
        payment_method = selectbox_with_preset(
            "Payment method",
            ["card", "apple_pay", "google_pay", "bank_transfer"],
            preset["payment_method"],
        )
        country = selectbox_with_preset(
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
            preset["country"],
        )
        is_foreign_transaction = selectbox_with_preset(
            "Foreign transaction",
            ["no", "yes"],
            preset["is_foreign_transaction"],
        )
        shipping_billing_mismatch = selectbox_with_preset(
            "Shipping/billing mismatch",
            ["no", "yes"],
            preset["shipping_billing_mismatch"],
        )
        kyc_completed = selectbox_with_preset(
            "KYC completed",
            ["yes", "no"],
            preset["kyc_completed"],
        )
        has_chargeback_history = selectbox_with_preset(
            "Chargeback history",
            ["no", "yes"],
            preset["has_chargeback_history"],
        )

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
            st.error(
                "Could not connect to FastAPI. Make sure the API is running on port 8000."
            )
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