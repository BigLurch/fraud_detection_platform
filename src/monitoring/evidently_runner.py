# Generates monitoring reports using Evidently.

# This module compares training reference data with recent prediction logs
# to detect data drift and generate an HTML monitoring report.

import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_PATH = "data/processed/train_ready.csv"
LOG_PATH = "artifacts/logs/predictions.jsonl"
OUTPUT_PATH = "artifacts/reports/evidently_report.html"


def load_reference_data(path: str = REFERENCE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_current_data(path: str = LOG_PATH) -> pd.DataFrame:
    log_path = Path(path)

    if not log_path.exists():
        raise FileNotFoundError(
            "Prediction log file not found. Run some API predictions first."
        )

    rows = []

    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError("Prediction log file is empty.")

    return pd.DataFrame(rows)


def align_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_columns = [
        col for col in reference_df.columns if col in current_df.columns
    ]

    reference_df = reference_df[common_columns].copy()
    current_df = current_df[common_columns].copy()

    return reference_df, current_df


def build_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
):
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    return snapshot


def save_report(report, path: str = OUTPUT_PATH) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    report.save_html(str(output))


def main() -> None:
    print("Loading reference data...")
    reference_df = load_reference_data()

    print("Loading prediction logs...")
    current_df = load_current_data()

    print("Aligning columns...")
    reference_df, current_df = align_columns(reference_df, current_df)

    print("Generating Evidently report...")
    report = build_report(reference_df, current_df)

    save_report(report)

    print(f"Report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()