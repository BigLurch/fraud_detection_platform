# Tests for fraud model training utilities.

# These tests verify that the training pipeline can be built and that
# the train/test split preserves feature and target structure.

from src.data.generate_data import create_dataset
from src.features.build_features import create_features
from src.models.train import build_pipeline, split_data


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_split_data_returns_train_test_sets():
    df = create_dataset(n_samples=200, fraud_rate=0.1, random_seed=42)
    df = create_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)