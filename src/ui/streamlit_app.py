# Provides a Streamlit dashboard for fraud detection.

# This module creates a lightweight user interface for submitting transaction
# data to the FastAPI inference service, displaying fraud risk results,
# and reviewing recent prediction logs.

import json
from pathlib import Path
import os
import psycopg2
import random
import time

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pydeck as pdk


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_HEALTH_URL = API_URL.replace("/predict", "/health")
LOG_PATH = "artifacts/logs/predictions.jsonl"
DATABASE_URL = os.getenv("DATABASE_URL")

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
)


def wake_up_api() -> bool:
    try:
        response = requests.get(API_HEALTH_URL, timeout=20)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


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
            "user_age": 34,
            "account_age_days": 140,
            "transaction_amount": 2850.0,
            "transaction_hour": 21,
            "ip_risk_score": 0.49,
            "num_prev_transactions_24h": 5,
            "avg_transaction_amount_7d": 1350.0,
            "failed_login_attempts_24h": 4,
            "email_domain": "outlook.com",
            "device_type": "mobile",
            "payment_method": "card",
            "country": "Germany",
            "is_foreign_transaction": "yes",
            "shipping_billing_mismatch": "no",
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
    if DATABASE_URL:
        return load_prediction_logs_from_database()

    return load_prediction_logs_from_jsonl(path)

def ensure_prediction_logs_table() -> None:
    if not DATABASE_URL:
        return

    create_sql = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        source TEXT,
        is_manual BOOLEAN,
        user_age INTEGER,
        account_age_days INTEGER,
        transaction_amount DOUBLE PRECISION,
        transaction_hour INTEGER,
        ip_risk_score DOUBLE PRECISION,
        num_prev_transactions_24h INTEGER,
        avg_transaction_amount_7d DOUBLE PRECISION,
        failed_login_attempts_24h INTEGER,
        email_domain TEXT,
        device_type TEXT,
        payment_method TEXT,
        country TEXT,
        city TEXT,
        lat DOUBLE PRECISION,
        lon DOUBLE PRECISION,
        is_foreign_transaction TEXT,
        shipping_billing_mismatch TEXT,
        kyc_completed TEXT,
        has_chargeback_history TEXT,
        prediction INTEGER,
        fraud_probability DOUBLE PRECISION,
        risk_label TEXT,
        raw_record JSONB
    );
    """

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(create_sql)

def load_prediction_logs_from_database(limit: int = 500) -> pd.DataFrame:
    ensure_prediction_logs_table()

    query = """
    SELECT
        timestamp,
        source,
        is_manual,
        user_age,
        account_age_days,
        transaction_amount,
        transaction_hour,
        ip_risk_score,
        num_prev_transactions_24h,
        avg_transaction_amount_7d,
        failed_login_attempts_24h,
        email_domain,
        device_type,
        payment_method,
        country,
        city,
        lat,
        lon,
        is_foreign_transaction,
        shipping_billing_mismatch,
        kyc_completed,
        has_chargeback_history,
        prediction,
        fraud_probability,
        risk_label
    FROM prediction_logs
    ORDER BY timestamp DESC
    LIMIT %s;
    """

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            return pd.read_sql_query(query, conn, params=(limit,))
    except Exception as error:
        st.warning(f"Could not load prediction logs from database: {error}")
        return pd.DataFrame()


def load_prediction_logs_from_jsonl(path: str = LOG_PATH) -> pd.DataFrame:
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


def get_demo_transaction(scenario: str) -> dict:
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
            "source": "demo_button",
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
            "source": "demo_button",
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
            "source": "demo_button",
        },
    }

    return presets[scenario]


def choose_demo_scenario() -> str:
    return random.choices(
        population=["legit", "suspicious", "fraud"],
        weights=[0.70, 0.20, 0.10],
        k=1,
    )[0]


def generate_demo_traffic(n_transactions: int = 25) -> tuple[int, int]:
    success_count = 0
    error_count = 0

    progress = st.progress(0)
    status = st.empty()

    for index in range(n_transactions):
        scenario = choose_demo_scenario()
        payload = get_demo_transaction(scenario)

        try:
            call_prediction_api(payload)
            success_count += 1
        except requests.exceptions.RequestException:
            error_count += 1

        progress.progress((index + 1) / n_transactions)
        status.caption(f"Generated {index + 1}/{n_transactions} demo transactions")
        time.sleep(0.05)

    progress.empty()
    status.empty()

    return success_count, error_count


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

def render_fraud_map(logs_df: pd.DataFrame) -> None:
    st.subheader("Transaction Origin Map")

    if logs_df.empty or "lat" not in logs_df.columns or "lon" not in logs_df.columns:
        st.info("No geolocation data available yet.")
        return

    map_df = logs_df.dropna(subset=["lat", "lon"]).copy()

    if map_df.empty:
        st.info("No valid map points available yet.")
        return

    map_df["risk_label"] = map_df["risk_label"].fillna("low").str.lower()
    map_df["source"] = map_df.get("source", "manual")
    map_df["is_manual"] = map_df["source"].eq("manual")

    def get_fill_color(risk_label: str) -> list[int]:
        if risk_label == "high":
            return [255, 40, 40, 180]
        if risk_label == "medium":
            return [255, 165, 0, 180]
        return [40, 200, 90, 180]

    def get_line_color(is_manual: bool) -> list[int]:
        if is_manual:
            return [66, 21, 234, 255]
        return [255, 255, 255, 0]

    map_df["color"] = map_df["risk_label"].apply(get_fill_color)
    map_df["line_color"] = map_df["is_manual"].apply(get_line_color)
    map_df["radius"] = 12000
    map_df["line_width"] = map_df["is_manual"].apply(lambda x: 3000 if x else 0)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_line_color="line_color",
        get_radius="radius",
        get_line_width="line_width",
        stroked=True,
        filled=True,
        pickable=True,
        opacity=0.85,
    )

    view_state = pdk.ViewState(
        latitude=50.5,
        longitude=14.0,
        zoom=3,
        pitch=25,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>Country:</b> {country}<br/>
                <b>City:</b> {city}<br/>
                <b>Risk:</b> {risk_label}<br/>
                <b>Source:</b> {source}<br/>
                <b>Amount:</b> {transaction_amount}<br/>
                <b>Fraud probability:</b> {fraud_probability}
            """,
            "style": {
                "backgroundColor": "#111827",
                "color": "white",
            },
        },
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.caption(
        "Map colors: green = low risk, orange = medium risk, red = high risk. "
        "Blue border = manual UI prediction."
    )

def selectbox_with_preset(label: str, options: list[str], preset_value: str) -> str:
    index = options.index(preset_value) if preset_value in options else 0
    return st.selectbox(label, options, index=index)


def main() -> None:
    st.title("🛡️ Fraud Detection Platform")
    st.caption(
        "Map colors: green = low risk, orange = medium risk, red = high risk. "
        "Blue border = manually submitted transaction."
    )

    with st.spinner("Starting backend API..."):
        api_ready = wake_up_api()

    if not api_ready:
        st.warning(
            "Backend API is starting up. This can take up to a minute on Render Free. "
            "Please wait a moment and refresh the page."
        )

    st_autorefresh(interval=7000, key="fraud_dashboard_refresh")

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

    demo_col1, demo_col2 = st.columns([1, 4])

    with demo_col1:
        if st.button("Generate Demo Traffic", type="secondary"):
            success_count, error_count = generate_demo_traffic(25)

            if error_count == 0:
                st.success(f"Generated {success_count} demo transactions.")
            else:
                st.warning(
                    f"Generated {success_count} transactions, "
                    f"but {error_count} requests failed."
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
        "source": "manual",
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

    logs_df = load_prediction_logs()

    render_fraud_map(logs_df)

    st.divider()

    st.subheader("Recent Prediction Logs")

    logs_df = load_prediction_logs()

    if logs_df.empty:
        st.info("No prediction logs found yet. Run a fraud check to create logs.")
    else:
        display_columns = [
            "timestamp",
            "country",
            "city",
            "transaction_amount",
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