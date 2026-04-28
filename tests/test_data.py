# Tests for synthetic fraud data generation.

# These tests verify that the generated dataset has the expected structure,
# target values, and basic data quality properties.

from src.data.generate_data import create_dataset
from src.data.schema import SCHEMA


def test_create_dataset_has_expected_shape():
    df = create_dataset(n_samples=100, fraud_rate=0.1, random_seed=42)

    assert len(df) == 100
    assert set(SCHEMA.all_columns).issubset(df.columns)


def test_target_contains_only_binary_values():
    df = create_dataset(n_samples=100, fraud_rate=0.1, random_seed=42)

    assert set(df["is_fraud"].unique()).issubset({0, 1})


def test_generated_dataset_has_no_nulls():
    df = create_dataset(n_samples=100, fraud_rate=0.1, random_seed=42)

    assert df.isnull().sum().sum() == 0


def test_fraud_rate_is_reasonable():
    df = create_dataset(n_samples=1000, fraud_rate=0.1, random_seed=42)

    fraud_rate = df["is_fraud"].mean()

    assert 0.08 <= fraud_rate <= 0.12