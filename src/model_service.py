from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH
from src.feature_engineering import map_region, normalize_job_title, seniority_score
from src.schemas import PredictionInput


@lru_cache
def load_model():
    model = joblib.load(MODEL_PATH)
    regressor = getattr(model, 'named_steps', {}).get('regressor')
    if regressor is not None and hasattr(regressor, 'n_jobs'):
        regressor.set_params(n_jobs=1)
    return model


@lru_cache
def load_metadata() -> dict:
    return json.loads(Path(METADATA_PATH).read_text(encoding='utf-8'))


@lru_cache
def canonical_job_titles() -> set[str]:
    metadata = load_metadata()
    return set(metadata['categorical_options']['job_title'])


def normalize_payload(payload: PredictionInput) -> dict[str, Any]:
    data = payload.model_dump()

    data['experience_level'] = str(data['experience_level']).upper().strip()
    data['employment_type'] = str(data['employment_type']).upper().strip()
    data['employee_residence'] = str(data['employee_residence']).upper().strip()
    data['company_location'] = str(data['company_location']).upper().strip()
    data['company_size'] = str(data['company_size']).upper().strip()

    normalized_title = normalize_job_title(data['job_title'])
    data['job_title'] = normalized_title if normalized_title in canonical_job_titles() else 'other'
    data['seniority_score'] = seniority_score(data['experience_level'])
    data['company_region'] = map_region(data['company_location'])
    data['employee_region'] = map_region(data['employee_residence'])

    return data


def predict_salary(payload: PredictionInput) -> float:
    model = load_model()
    normalized = normalize_payload(payload)
    feature_columns = load_metadata().get('feature_columns')
    frame = pd.DataFrame([normalized], columns=feature_columns)
    prediction = model.predict(frame)[0]
    return round(float(prediction), 2)
