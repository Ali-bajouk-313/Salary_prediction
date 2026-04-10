from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / '.env')

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'ds_salaries.csv'
PROCESSED_DIR = DATA_DIR / 'processed'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'salary_model.joblib'
METADATA_PATH = ARTIFACTS_DIR / 'training_metadata.json'
METRICS_PATH = ARTIFACTS_DIR / 'metrics.json'
CHARTS_DIR = ARTIFACTS_DIR / 'charts'
LOCAL_HISTORY_PATH = ARTIFACTS_DIR / 'prediction_history.json'

SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_TIMEOUT_SECONDS = float(os.getenv('OLLAMA_TIMEOUT_SECONDS', '12'))
SUPABASE_TIMEOUT_SECONDS = float(os.getenv('SUPABASE_TIMEOUT_SECONDS', '3'))
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000')

for path in [PROCESSED_DIR, ARTIFACTS_DIR, CHARTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
