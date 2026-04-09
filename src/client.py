from __future__ import annotations

import json
import random
from pathlib import Path

import requests

from src.config import API_BASE_URL, METADATA_PATH


def load_metadata() -> dict:
    return json.loads(Path(METADATA_PATH).read_text(encoding='utf-8'))


def generate_payloads(limit: int = 12) -> list[dict]:
    metadata = load_metadata()
    options = metadata['categorical_options']
    ranges = metadata['numeric_ranges']
    payloads = []
    titles = options['job_title']
    random.seed(42)
    for _ in range(limit):
        payloads.append(
            {
                'work_year': random.randint(ranges['work_year']['min'], ranges['work_year']['max']),
                'experience_level': random.choice(options['experience_level']),
                'employment_type': random.choice(options['employment_type']),
                'job_title': random.choice(titles),
                'employee_residence': random.choice(options['employee_residence']),
                'remote_ratio': random.choice(ranges['remote_ratio']['allowed']),
                'company_location': random.choice(options['company_location']),
                'company_size': random.choice(options['company_size']),
            }
        )
    return payloads


def call_api(limit: int = 12) -> list[dict]:
    results = []
    for payload in generate_payloads(limit=limit):
        response = requests.get(f'{API_BASE_URL}/predict', params=payload, timeout=20)
        response.raise_for_status()
        results.append(response.json())
    return results


if __name__ == '__main__':
    try:
        print(json.dumps(call_api(), indent=2))
    except requests.RequestException as exc:
        raise SystemExit(f'API call failed: {exc}')
