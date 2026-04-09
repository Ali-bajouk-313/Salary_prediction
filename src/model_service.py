from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH
from src.schemas import PredictionInput


@lru_cache
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache
def load_metadata() -> dict:
    return json.loads(Path(METADATA_PATH).read_text(encoding='utf-8'))


@lru_cache
def canonical_job_titles() -> set[str]:
    metadata = load_metadata()
    return {title.lower() for title in metadata['categorical_options']['job_title']}


@lru_cache
def title_fallback_map() -> dict[str, str]:
    metadata = load_metadata()
    return {title.lower(): title for title in metadata['categorical_options']['job_title']}


def normalize_payload(payload: PredictionInput) -> dict:
    data = payload.model_dump()
    title = data['job_title'].lower()
    if title not in canonical_job_titles():
        fallback = 'Data Scientist'
        data['job_title'] = title_fallback_map().get(title, fallback)
    return data


def predict_salary(payload: PredictionInput) -> float:
    model = load_model()
    normalized = normalize_payload(payload)
    frame = pd.DataFrame([normalized])
    prediction = model.predict(frame)[0]
    return round(float(prediction), 2)