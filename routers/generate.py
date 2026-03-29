"""
/generate-video endpoint.

Design: async HTTP — client gets a job_id immediately, then polls /status.
The heavy inference runs in a background ThreadPoolExecutor (not asyncio,
because PyTorch is synchronous and blocks the event loop otherwise).

WHY NOT Celery/Redis?
  For a solo dev MVP, threading is enough. One T4 can process one video
  at a time. Celery adds operational complexity (broker, worker process,
  deployment). Add it in Phase 2 when you have paying users queueing.
"""

import uuid
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from models import GenerateRequest, GenerateResponse, StatusResponse
from config import FRAMES_DIR, OUTPUT_DIR
from store import create_job, update_job, get_job
from services.prompt_processor import expand_prompt
from services.generation_engine import run_animatediff, run_svd
from services.renderer import render_video, cleanup_job_frames

router  = APIRouter()
logger  = logging.getLogger(__name__)
_sem    = threading.Semaphore(1)  # one inference job at a time on single GPU


@router.post("/generate-video", response_model=GenerateResponse)
async def generate_video(req: GenerateRequest, bg: BackgroundTasks):
    """
    Accepts a generation request. Returns job_id immediately.
    Inference runs in the background.

    Example request:
    {
      "prompt": "a lone samurai standing in a misty bamboo forest at dawn",
      "duration_seconds": 4,
      "camera_motion": "zoom_in",
      "style": "cinematic",
      "seed": 42
    }

    Example response:
    {
      "job_id": "a1b2c3d4",
      "status": "queued",
      "message": "Job queued. Poll /status/a1b2c3d4 for progress."
    }
    """
    job_id = uuid.uuid4().hex[:8]
    create_job(job_id)
    bg.add_task(_run_pipeline, job_id, req)

    return GenerateResponse(
        job_id=job_id,
        status="queued",
        message=f"Job queued. Poll /status/{job_id} for progress.",
    )


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """Download the final MP4 once status == 'done'."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    video_path = OUTPUT_DIR / f"{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"video_{job_id}.mp4",
    )


# ── Pipeline orchestrator (runs in background thread) ────────────────────────

def _run_pipeline(job_id: str, req: GenerateRequest) -> None:
    """
    Full pipeline: prompt → frames → temporal → render → cleanup.
    Semaphore ensures only one job runs at a time on shared GPU.
    """
    acquired = _sem.acquire(blocking=False)
    if not acquired:
        update_job(job_id, status="queued",
                   error="GPU busy, please retry in a moment.")
        return

    frames_dir = FRAMES_DIR / job_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_job(job_id, status="processing", progress=2)

        # 1. Expand prompt
        expanded = expand_prompt(req)
        logger.info(f"[{job_id}] Expanded prompt: {expanded.full_positive[:80]}...")
        update_job(job_id, progress=5)

        # 2. Generate frames
        def progress_cb(pct: int):
            update_job(job_id, progress=pct)

        if req.use_svd and req.image_base64:
            frame_paths = run_svd(req, expanded, frames_dir, progress_cb)
        else:
            frame_paths = run_animatediff(req, expanded, frames_dir, progress_cb)

        # 3. Render to MP4
        update_job(job_id, progress=90)
        video_path = render_video(frames_dir, job_id, progress_cb)

        # 4. Update job — done
        video_url = f"/result/{job_id}"
        update_job(job_id, status="done", progress=100, video_url=video_url)
        logger.info(f"[{job_id}] Done. Output: {video_path}")

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline error: {e}")
        update_job(job_id, status="error", error=str(e))
    finally:
        cleanup_job_frames(frames_dir)
        _sem.release()