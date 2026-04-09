from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.config import RAW_DATA_PATH

TARGET = 'salary_in_usd'
DROP_COLUMNS = ['Unnamed: 0', 'salary', 'salary_currency']
FEATURE_COLUMNS = [
    'work_year',
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'remote_ratio',
    'company_location',
    'company_size',
]
CATEGORICAL_COLUMNS = [
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'company_location',
    'company_size',
]
NUMERIC_COLUMNS = ['work_year', 'remote_ratio']


@dataclass
class PreparedData:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    metadata: dict


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    csv_path = path or str(RAW_DATA_PATH)
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=[col for col in DROP_COLUMNS if col in cleaned.columns])
    cleaned = cleaned.dropna(subset=FEATURE_COLUMNS + [TARGET]).drop_duplicates().reset_index(drop=True)
    cleaned['remote_ratio'] = cleaned['remote_ratio'].clip(lower=0, upper=100)
    cleaned['work_year'] = cleaned['work_year'].astype(int)
    cleaned[TARGET] = cleaned[TARGET].astype(float)
    return cleaned


def build_metadata(df: pd.DataFrame) -> dict:
    categorical_options = {
        col: sorted(df[col].astype(str).unique().tolist())
        for col in CATEGORICAL_COLUMNS
    }
    numeric_ranges = {
        'work_year': {'min': int(df['work_year'].min()), 'max': int(df['work_year'].max())},
        'remote_ratio': {'allowed': sorted(df['remote_ratio'].astype(int).unique().tolist())},
    }
    sample_payload = {
        col: (categorical_options[col][0] if col in categorical_options else int(df[col].iloc[0]))
        for col in FEATURE_COLUMNS
    }
    return {
        'feature_columns': FEATURE_COLUMNS,
        'categorical_columns': CATEGORICAL_COLUMNS,
        'numeric_columns': NUMERIC_COLUMNS,
        'categorical_options': categorical_options,
        'numeric_ranges': numeric_ranges,
        'sample_payload': sample_payload,
        'training_rows': int(len(df)),
    }


def prepare_training_data(path: str | None = None) -> PreparedData:
    raw = load_raw_data(path)
    cleaned = clean_dataset(raw)
    metadata = build_metadata(cleaned)
    return PreparedData(
        frame=cleaned,
        features=cleaned[FEATURE_COLUMNS].copy(),
        target=cleaned[TARGET].copy(),
        metadata=metadata,
    )


if __name__ == '__main__':
    prepared = prepare_training_data()
    print(json.dumps(prepared.metadata, indent=2))
