# Handles prediction logging for the fraud detection API.

# This module writes every prediction request and model response to a JSONL log file,
# which can later be used for monitoring, dashboards, and drift analysis.

import json
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = "artifacts/logs/predictions.jsonl"


def log_prediction(payload: dict, prediction_result: dict, path: str = LOG_PATH) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
        **prediction_result,
    }

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_record) + "\n")