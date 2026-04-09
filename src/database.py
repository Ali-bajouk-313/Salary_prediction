from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.config import SUPABASE_KEY, SUPABASE_URL


def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
    except ImportError:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def save_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    client = get_supabase_client()
    payload = {
        **record,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    if client is None:
        return {'saved': False, 'reason': 'Supabase is not configured', 'payload': payload}
    response = client.table('prediction_runs').insert(payload).execute()
    return {'saved': True, 'response': getattr(response, 'data', response)}


def fetch_prediction_history(limit: int = 50) -> list[dict[str, Any]]:
    client = get_supabase_client()
    if client is None:
        return []
    response = client.table('prediction_runs').select('*').order('created_at', desc=True).limit(limit).execute()
    return list(getattr(response, 'data', []) or [])
