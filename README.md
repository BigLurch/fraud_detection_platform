# Fraud Detection Platform

An end-to-end MLOps portfolio project that simulates fraudulent transactions and fake accounts, trains a fraud detection model, serves predictions through FastAPI, provides a Streamlit user interface, logs predictions, and monitors data drift with Evidently.

## Project goal

The goal of this project is to build a production-inspired fraud detection platform that demonstrates practical skills in:

- machine learning
- feature engineering
- experiment tracking
- model versioning
- API development
- testing
- monitoring
- containerization

This project is designed as a strong portfolio case for roles in:

- MLOps Engineering
- ML Engineering
- AI Engineering

## Planned architecture

The platform will include:

- synthetic fraud data generation
- preprocessing and feature engineering
- model training in Python
- experiment tracking with MLflow
- saved model artifacts
- inference API with FastAPI
- Streamlit demo UI
- prediction logging
- monitoring with Evidently
- Docker support
- automated tests

## Project structure

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
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Initial MVP scope

The first MVP version will support:
* generation of synthetic transaction data
* fake and legitimate account patterns
* fraud classification model training
* single prediction through API
* prediction display in Streamlit
* prediction logging to file
* drift monitoring report generation

## Tech stack
* Python 3.11
* pandas
* numpy
* scikit-learn
* MLflow
* FastAPI
* Streamlit
* Evidently
* pytest
* Docker

## Status
Project setup in progress.

Next step:
* build synthetic transaction and fake account data generator

## Future improvements

Possible future extensions:
* batch inference endpoint
* model registry integration
* CI/CD pipeline
* Prometheus/Grafana metrics
* cloud deployment
* feature store simulation# fraud_detection_platform
