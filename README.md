# Fraud Detection Platform

An end-to-end MLOps portfolio project that simulates fraud activity in a modern digital payment platform.

The system generates synthetic transactions and fake accounts, trains a machine learning model to detect fraud, serves predictions through FastAPI, provides a Streamlit dashboard, logs predictions, and monitors model/data drift over time.

---

# Project Vision

This project is designed to feel like a real fraud prevention product used in a modern checkout or payment environment.

It combines two perspectives:

## Payment Platform

* customer transactions
* payment risk scoring
* account abuse prevention
* real-time fraud checks

## Fraud Intelligence Dashboard

* suspicious activity alerts
* transaction monitoring
* risk trends
* geolocation visualizations
* model health monitoring

---

# Current Development Status

## Completed

### Project Foundation

* professional project structure
* uv virtual environment setup
* modular Python package layout
* config-driven architecture

### Synthetic Data Engine

* realistic payment transaction simulation
* legitimate customer behavior patterns
* synthetic / fake account abuse scenarios
* suspicious purchase behavior
* fraud labels for supervised learning

### Data Quality Layer

* schema definition
* dataset validation checks
* null / duplicate / range validation

### Feature Engineering

Created model-ready features such as:

* amount_to_avg_ratio
* is_night_transaction
* high_velocity_flag
* new_account_flag
* risky_email_domain
* foreign_high_amount_flag
* login_risk_flag

---

# Planned Next Steps

## Machine Learning Pipeline

* preprocessing pipeline
* train / test split
* RandomForest baseline model
* model evaluation
* saved artifacts

## MLOps Layer

* MLflow experiment tracking
* model versioning
* inference logging
* Evidently monitoring

## Serving Layer

* FastAPI prediction API
* batch inference endpoint
* health checks

## UI Dashboard

* Streamlit fraud console
* live prediction view
* recent alerts
* world fraud activity map
* model monitoring panel

## Deployment

* Docker
* docker-compose
* GitHub Actions CI

---

# Example Use Cases

The model will learn to detect scenarios such as:

* new account making large foreign purchase
* multiple failed login attempts followed by payment
* synthetic identity using risky email domain
* sudden high-value purchase outside normal behavior
* high velocity transactions in short timeframe

---

# Tech Stack

* Python 3.11
* pandas
* numpy
* scikit-learn
* FastAPI
* Streamlit
* MLflow
* Evidently
* pytest
* Docker

---

# Repository Structure

```text
fraud_detection_platform/
├── artifacts/
├── configs/
├── data/
├── docker/
├── notebooks/
├── scripts/
├── src/
├── tests/
└── README.md
```

# Why This Project Matters

This project demonstrates practical skills for roles such as:
* MLOps Engineer
* Machine Learning Engineer
* AI Engineer
* Data / ML Platform Engineer

It focuses not only on model training, but on the full lifecycle of machine learning systems.