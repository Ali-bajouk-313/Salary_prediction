from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import LOCAL_HISTORY_PATH, SUPABASE_KEY, SUPABASE_TIMEOUT_SECONDS, SUPABASE_URL

_LOCAL_HISTORY_LIMIT = 2000
_REMOTE_PAGE_SIZE = 250
_REBUILD_TIMEOUT_SECONDS = max(SUPABASE_TIMEOUT_SECONDS, 15.0)
_HISTORY_FIELDS = [
    'run_id',
    'created_at',
    'work_year',
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'remote_ratio',
    'company_location',
    'company_size',
    'predicted_salary_usd',
    'llm_analysis',
]
_REMOTE_HISTORY_SELECT = ','.join(_HISTORY_FIELDS)
_history_lock = threading.Lock()


def get_supabase_client(timeout_seconds: float | None = None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        from supabase.lib.client_options import ClientOptions
    except ImportError:
        return None
    client_timeout = timeout_seconds or SUPABASE_TIMEOUT_SECONDS
    try:
        return create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options=ClientOptions(
                postgrest_client_timeout=client_timeout,
                storage_client_timeout=client_timeout,
                function_client_timeout=client_timeout,
            ),
        )
    except Exception:
        return None


def _run_with_timeout(operation, timeout_seconds: float):
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, operation()))
        except Exception as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)

    if worker.is_alive():
        return False, TimeoutError(f'Operation timed out after {timeout_seconds} seconds')
    if result_queue.empty():
        return False, RuntimeError('Operation ended without a result')
    return result_queue.get()


def _history_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    snapshot = {field: record.get(field) for field in _HISTORY_FIELDS if field in record}
    if not snapshot.get('run_id') and record.get('created_at'):
        snapshot['run_id'] = str(record['created_at'])
    return snapshot


def _read_local_history() -> list[dict[str, Any]]:
    path = Path(LOCAL_HISTORY_PATH)
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(records, list):
        return []
    return [_history_snapshot(item) for item in records if isinstance(item, dict)]


def _write_local_history(records: list[dict[str, Any]]) -> None:
    path = Path(LOCAL_HISTORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding='utf-8')


def _sort_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[str, str]:
        created_at = str(item.get('created_at') or '')
        run_id = str(item.get('run_id') or '')
        return created_at, run_id

    return sorted(records, key=sort_key, reverse=True)


def _merge_history(*sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for records in sources:
        for item in records:
            snapshot = _history_snapshot(item)
            run_id = str(snapshot.get('run_id') or '')
            created_at = str(snapshot.get('created_at') or '')
            key = run_id or created_at or str(id(snapshot))
            existing = merged.get(key)
            if existing is None:
                merged[key] = snapshot
                continue
            existing_created = str(existing.get('created_at') or '')
            if created_at >= existing_created:
                merged[key] = snapshot
    return _sort_history(list(merged.values()))


def _save_local_prediction_record(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = _history_snapshot(payload)
    with _history_lock:
        records = _read_local_history()
        records = [item for item in records if item.get('run_id') != snapshot.get('run_id')]
        records.append(snapshot)
        records = _sort_history(records)[:_LOCAL_HISTORY_LIMIT]
        _write_local_history(records)
    return {'saved': True, 'path': str(LOCAL_HISTORY_PATH), 'count': len(records)}


def _fetch_remote_prediction_history_page(start: int, end: int, timeout_seconds: float) -> list[dict[str, Any]]:
    client = get_supabase_client(timeout_seconds)
    if client is None:
        return []
    ok, result = _run_with_timeout(
        lambda: client.table('prediction_runs')
        .select(_REMOTE_HISTORY_SELECT)
        .order('created_at', desc=True)
        .range(start, end)
        .execute(),
        timeout_seconds,
    )
    if not ok:
        return []
    return [_history_snapshot(item) for item in (getattr(result, 'data', []) or [])]


def _fetch_remote_prediction_history(limit: int | None, timeout_seconds: float) -> list[dict[str, Any]]:
    target = limit or _LOCAL_HISTORY_LIMIT
    collected: list[dict[str, Any]] = []
    start = 0

    while len(collected) < target:
        end = min(start + _REMOTE_PAGE_SIZE - 1, target - 1)
        page = _fetch_remote_prediction_history_page(start, end, timeout_seconds)
        if not page:
            break
        collected.extend(page)
        if len(page) < (end - start + 1):
            break
        start += len(page)

    return collected[:target]


def save_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        **record,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    local_result = _save_local_prediction_record(payload)

    client = get_supabase_client()
    if client is None:
        return {
            'saved': True,
            'local_saved': True,
            'remote_saved': False,
            'reason': 'Supabase is not configured; saved locally',
            'payload': _history_snapshot(payload),
        }

    ok, result = _run_with_timeout(
        lambda: client.table('prediction_runs').insert(payload).execute(),
        SUPABASE_TIMEOUT_SECONDS,
    )
    if ok:
        return {
            'saved': True,
            'local_saved': local_result['saved'],
            'remote_saved': True,
            'response': getattr(result, 'data', result),
            'payload': _history_snapshot(payload),
        }
    if isinstance(result, Exception):
        return {
            'saved': True,
            'local_saved': local_result['saved'],
            'remote_saved': False,
            'reason': f'Supabase save failed: {result}',
            'payload': _history_snapshot(payload),
        }
    return {
        'saved': True,
        'local_saved': local_result['saved'],
        'remote_saved': False,
        'reason': 'Supabase save failed',
        'payload': _history_snapshot(payload),
    }


def fetch_prediction_history(limit: int | None = 50) -> list[dict[str, Any]]:
    with _history_lock:
        local_history = _read_local_history()
    sorted_history = _sort_history(local_history)
    if limit is None:
        return sorted_history
    return sorted_history[:limit]


def rebuild_prediction_history(limit: int | None = None) -> dict[str, Any]:
    target = limit or _LOCAL_HISTORY_LIMIT
    with _history_lock:
        local_history = _read_local_history()
    client = get_supabase_client(_REBUILD_TIMEOUT_SECONDS)
    if client is None:
        merged = _sort_history(local_history)[:target]
        with _history_lock:
            _write_local_history(merged)
        return {
            'saved': True,
            'count': len(merged),
            'local_count': len(local_history),
            'remote_count': 0,
            'remote_available': False,
            'reason': 'Supabase is not configured or could not be reached',
            'path': str(LOCAL_HISTORY_PATH),
        }
    remote_history = _fetch_remote_prediction_history(target, _REBUILD_TIMEOUT_SECONDS)
    merged = _merge_history(local_history, remote_history)[:target]
    with _history_lock:
        _write_local_history(merged)
    return {
        'saved': True,
        'count': len(merged),
        'local_count': len(local_history),
        'remote_count': len(remote_history),
        'remote_available': True,
        'reason': 'No remote records were fetched during rebuild' if not remote_history else '',
        'path': str(LOCAL_HISTORY_PATH),
    }
