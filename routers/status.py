from fastapi import APIRouter, HTTPException
from models import StatusResponse
from store import get_job

router = APIRouter()

@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Poll this endpoint to track progress.

    Returns:
      status:    queued | processing | done | error
      progress:  0-100
      video_url: set only when status == 'done'
      error:     set only when status == 'error'

    Recommended polling interval: 2s during processing, stop on done/error.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        video_url=job.get("video_url"),
        error=job.get("error"),
    )