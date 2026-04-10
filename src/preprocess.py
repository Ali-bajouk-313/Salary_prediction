from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.config import RAW_DATA_PATH
from src.feature_engineering import map_region, normalize_job_title, seniority_score

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
    'seniority_score',
    'company_region',
    'employee_region',
]
CATEGORICAL_COLUMNS = [
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'company_location',
    'company_size',
    'company_region',
    'employee_region',
]
NUMERIC_COLUMNS = ['work_year', 'remote_ratio', 'seniority_score']


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
    cleaned = cleaned.dropna(
        subset=[
            'work_year',
            'experience_level',
            'employment_type',
            'job_title',
            'employee_residence',
            'remote_ratio',
            'company_location',
            'company_size',
            TARGET,
        ]
    ).reset_index(drop=True)

    cleaned['work_year'] = cleaned['work_year'].astype(int)
    cleaned['experience_level'] = cleaned['experience_level'].astype(str).str.upper().str.strip()
    cleaned['employment_type'] = cleaned['employment_type'].astype(str).str.upper().str.strip()
    cleaned['job_title'] = cleaned['job_title'].map(normalize_job_title)
    cleaned['employee_residence'] = cleaned['employee_residence'].astype(str).str.upper().str.strip()
    cleaned['remote_ratio'] = cleaned['remote_ratio'].clip(lower=0, upper=100).astype(int)
    cleaned['company_location'] = cleaned['company_location'].astype(str).str.upper().str.strip()
    cleaned['company_size'] = cleaned['company_size'].astype(str).str.upper().str.strip()
    cleaned[TARGET] = cleaned[TARGET].astype(float)
    cleaned['seniority_score'] = cleaned['experience_level'].map(seniority_score).astype(int)
    cleaned['company_region'] = cleaned['company_location'].map(map_region)
    cleaned['employee_region'] = cleaned['employee_residence'].map(map_region)

    return cleaned.drop_duplicates().reset_index(drop=True)


def build_metadata(df: pd.DataFrame) -> dict:
    categorical_options = {col: sorted(df[col].astype(str).unique().tolist()) for col in CATEGORICAL_COLUMNS}
    numeric_ranges = {
        'work_year': {'min': int(df['work_year'].min()), 'max': int(df['work_year'].max())},
        'remote_ratio': {'allowed': sorted(df['remote_ratio'].astype(int).unique().tolist())},
        'seniority_score': {'min': int(df['seniority_score'].min()), 'max': int(df['seniority_score'].max())},
    }
    sample_payload = {
        'work_year': int(df['work_year'].iloc[0]),
        'experience_level': str(df['experience_level'].iloc[0]),
        'employment_type': str(df['employment_type'].iloc[0]),
        'job_title': str(df['job_title'].iloc[0]),
        'employee_residence': str(df['employee_residence'].iloc[0]),
        'remote_ratio': int(df['remote_ratio'].iloc[0]),
        'company_location': str(df['company_location'].iloc[0]),
        'company_size': str(df['company_size'].iloc[0]),
        'seniority_score': int(df['seniority_score'].iloc[0]),
        'company_region': str(df['company_region'].iloc[0]),
        'employee_region': str(df['employee_region'].iloc[0]),
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
