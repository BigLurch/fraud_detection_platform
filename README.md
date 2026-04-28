# Fraud Detection Platform

Production-style end-to-end MLOps project for real-time fraud detection in a modern digital payment platform.

This platform simulates live payment traffic, detects fraudulent transactions using machine learning, exposes predictions through FastAPI, visualizes risk events in a Streamlit dashboard, tracks experiments with MLflow, monitors drift with Evidently, and runs fully containerized with Docker.


# Project Overview

Fraud detection is one of the most valuable machine learning use cases in fintech and e-commerce.

This project was built to simulate how a modern payment company could:

- score incoming transactions in real time
- detect suspicious account behavior
- monitor fraud activity globally
- track model performance over time
- deploy and maintain an ML system in production

The focus is not only the model — but the **full ML lifecycle**.


# Key Features

## Real-Time Fraud Detection API

- FastAPI prediction endpoint
- fraud probability scoring
- low / medium / high risk labels
- health check endpoint
- JSON responses for easy integration

## Synthetic Fraud Traffic Simulator

- generates continuous live transactions
- realistic customer behavior
- suspicious scenarios
- fraud attack patterns
- sends traffic directly to API

## Interactive Fraud Dashboard

Built with Streamlit.

Includes:

- live fraud alerts
- KPI metrics
- recent transaction feed
- real-time geolocation fraud map
- manual fraud testing controls
- highlighted manual events

## Machine Learning Pipeline

- synthetic training dataset generation
- preprocessing pipeline
- feature engineering
- train / test split
- RandomForest fraud classifier
- saved model artifacts

## Experiment Tracking

Using MLflow:

- parameters
- metrics
- model versions
- artifacts

## Monitoring

Using Evidently:

- data drift detection
- feature drift reporting
- production monitoring reports

## Engineering / DevOps

- Docker + Docker Compose
- pytest test suite
- GitHub Actions CI pipeline
- modular production-style structure


# Example Fraud Signals

The model learns to detect patterns such as:

- new account making high-value foreign purchase
- many failed logins before payment
- suspicious email domain
- high transaction velocity
- billing / shipping mismatch
- unusually high purchase amount
- risky IP behavior


# ML Performance

Example latest run:

```json
{
  "precision": 0.9857,
  "recall": 0.8625,
  "f1_score": 0.9200,
  "roc_auc": 0.9938
}
```

Strong fraud detection performance with balanced precision and recall.


# System Architecture

```text
Live Simulator
      ↓
 FastAPI API
      ↓
 Fraud Model
      ↓
Prediction Logs
      ↓
Streamlit Dashboard
      ↓
Evidently Monitoring
```


# Tech Stack

## Core
- Python 3.11
- pandas
- numpy
- scikit-learn
## Serving
- FastAPI
- Uvicorn
## Frontend
- Streamlit
- PyDeck Maps
## MLOps
- MLflow
- Evidently
## Engineering
- pytest
- GitHub Actions
- Docker
- Docker Compose


# Run Locally

## Clone repo

```bash
git clone https://github.com/BigLurch/fraud_detection_platform.git
cd fraud_detection_platform
```

## Start with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```


# Local URLs

## Dashboard

```text
http://localhost:8501
```

## API Docs

```text
http://localhost:8000/docs
```


# Run Tests

```bash
uv run pytest -v
```


# CI/CD

Every push automatically runs:
- dependency install
- dataset generation
- feature pipeline
- model training
- pytest suite

Configured through GitHub Actions.


# Repository Structure

```text
fraud_detection_platform/
├── artifacts/
│   ├── logs/
│   ├── metrics/
│   ├── models/
│   └── reports/
├── configs/
├── data/
├── docker/
├── notebooks/
├── scripts/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── ui/
├── tests/
└── README.md
```


# Why This Project Matters

This project demonstrates practical skills relevant for roles such as:
- MLOps Engineer
- Machine Learning Engineer
- AI Engineer
- ML Platform Engineer
- Backend Engineer (AI systems)

It shows ability to build not only models — but complete, production-style machine learning systems.


# Future Improvements

- cloud deployment (Render / AWS / GCP)
- PostgreSQL prediction storage
- Kafka live event streaming
- automated retraining pipeline
- alert notifications
- role-based analyst dashboard
- advanced anomaly detection models


# Author

Built and designed by **Jonas Johansson** as a portfolio project focused on production-grade ML systems, fraud detection, and MLOps engineering.