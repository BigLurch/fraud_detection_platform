# Handles prediction logging for the fraud detection API.

# This module writes every prediction request and model response to a JSONL log file,
# which can later be used for monitoring, dashboards, and drift analysis.

import json
import random
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = "artifacts/logs/predictions.jsonl"


COUNTRY_CITY_COORDINATES = {
    "Sweden": [
        {"city": "Stockholm", "lat": 59.3293, "lon": 18.0686},
        {"city": "Gothenburg", "lat": 57.7089, "lon": 11.9746},
        {"city": "Malmö", "lat": 55.6050, "lon": 13.0038},
        {"city": "Uppsala", "lat": 59.8586, "lon": 17.6389},
    ],
    "Norway": [
        {"city": "Oslo", "lat": 59.9139, "lon": 10.7522},
        {"city": "Bergen", "lat": 60.3913, "lon": 5.3221},
    ],
    "Denmark": [
        {"city": "Copenhagen", "lat": 55.6761, "lon": 12.5683},
        {"city": "Aarhus", "lat": 56.1629, "lon": 10.2039},
    ],
    "Finland": [
        {"city": "Helsinki", "lat": 60.1699, "lon": 24.9384},
        {"city": "Tampere", "lat": 61.4978, "lon": 23.7610},
    ],
    "Germany": [
        {"city": "Berlin", "lat": 52.5200, "lon": 13.4050},
        {"city": "Hamburg", "lat": 53.5511, "lon": 9.9937},
        {"city": "Munich", "lat": 48.1351, "lon": 11.5820},
    ],
    "Netherlands": [
        {"city": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
        {"city": "Rotterdam", "lat": 51.9244, "lon": 4.4777},
    ],
    "Poland": [
        {"city": "Warsaw", "lat": 52.2297, "lon": 21.0122},
        {"city": "Krakow", "lat": 50.0647, "lon": 19.9450},
        {"city": "Gdansk", "lat": 54.3520, "lon": 18.6466},
    ],
    "Romania": [
        {"city": "Bucharest", "lat": 44.4268, "lon": 26.1025},
        {"city": "Cluj-Napoca", "lat": 46.7712, "lon": 23.6236},
    ],
    "Nigeria": [
        {"city": "Lagos", "lat": 6.5244, "lon": 3.3792},
        {"city": "Abuja", "lat": 9.0765, "lon": 7.3986},
        {"city": "Kano", "lat": 12.0022, "lon": 8.5920},
    ],
    "Turkey": [
        {"city": "Istanbul", "lat": 41.0082, "lon": 28.9784},
        {"city": "Ankara", "lat": 39.9334, "lon": 32.8597},
        {"city": "Izmir", "lat": 38.4237, "lon": 27.1428},
    ],
}


def get_coordinates(country: str) -> dict:
    cities = COUNTRY_CITY_COORDINATES.get(
        country,
        COUNTRY_CITY_COORDINATES["Sweden"],
    )

    city = random.choice(cities)

    return {
        "city": city["city"],
        "lat": round(city["lat"] + random.uniform(-0.08, 0.08), 4),
        "lon": round(city["lon"] + random.uniform(-0.08, 0.08), 4),
    }


def log_prediction(payload: dict, prediction_result: dict, path: str = LOG_PATH) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    geolocation = get_coordinates(payload.get("country", "Sweden"))
    source = payload.get("source", "manual")

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "is_manual": source == "manual",
        **payload,
        **geolocation,
        **prediction_result,
    }

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_record) + "\n")