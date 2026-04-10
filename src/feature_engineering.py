from __future__ import annotations

import re


def normalize_job_title(job_title: str) -> str:
    cleaned = re.sub(r'[^a-z0-9]+', ' ', str(job_title).lower()).strip()
    if not cleaned:
        return 'other'
    if 'research scientist' in cleaned:
        return 'research_scientist'
    if 'analytics engineer' in cleaned:
        return 'analytics_engineer'
    if 'architect' in cleaned:
        return 'architect'
    if 'analyst' in cleaned:
        return 'data_analyst'
    if 'data engineer' in cleaned or 'etl' in cleaned:
        return 'data_engineer'
    if 'machine learning' in cleaned and 'engineer' in cleaned:
        return 'ml_engineer'
    if cleaned == 'ml engineer' or cleaned.endswith(' ml engineer') or cleaned.startswith('ml engineer '):
        return 'ml_engineer'
    if 'software engineer' in cleaned:
        return 'software_engineer'
    if 'director' in cleaned or 'head of' in cleaned:
        return 'director'
    if 'manager' in cleaned:
        return 'manager'
    if 'research' in cleaned:
        return 'research_role'
    if 'scientist' in cleaned or cleaned == 'ai scientist':
        return 'data_scientist'
    if 'engineer' in cleaned:
        return 'engineer'
    return 'other'


def map_region(country_code: str) -> str:
    code = str(country_code).upper().strip()

    north_america = {'US', 'CA', 'MX'}
    europe = {
        'GB',
        'DE',
        'FR',
        'ES',
        'PT',
        'NL',
        'IE',
        'IT',
        'PL',
        'GR',
        'AT',
        'BE',
        'CH',
        'DK',
        'SE',
        'NO',
        'FI',
        'RO',
        'CZ',
        'HU',
    }
    asia = {'IN', 'JP', 'CN', 'SG', 'AE', 'SA', 'TR', 'IL', 'PK', 'ID', 'MY', 'TH', 'VN', 'KR', 'PH', 'HK'}
    south_america = {'BR', 'AR', 'CL', 'CO', 'PE'}
    africa = {'ZA', 'EG', 'NG', 'KE', 'MA', 'TN'}
    oceania = {'AU', 'NZ'}

    if code in north_america:
        return 'north_america'
    if code in europe:
        return 'europe'
    if code in asia:
        return 'asia'
    if code in south_america:
        return 'south_america'
    if code in africa:
        return 'africa'
    if code in oceania:
        return 'oceania'
    return 'other'


def seniority_score(experience_level: str) -> int:
    return {
        'EN': 1,
        'MI': 2,
        'SE': 3,
        'EX': 4,
    }.get(str(experience_level).upper().strip(), 0)
