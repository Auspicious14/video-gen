"""
Simple in-memory job store.
For production: swap this out for Redis with rq or Celery.
The interface is identical so the routers don't change.
"""
import threading
from typing import Dict, Any

_lock  = threading.Lock()
_store: Dict[str, Dict[str, Any]] = {}

def create_job(job_id: str) -> None:
    with _lock:
        _store[job_id] = {
            "status":    "queued",
            "progress":  0,
            "video_url": None,
            "error":     None,
        }

def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id].update(kwargs)

def get_job(job_id: str) -> Dict[str, Any]:
    with _lock:
        return dict(_store.get(job_id, {}))

def delete_job(job_id: str) -> None:
    with _lock:
        _store.pop(job_id, None)