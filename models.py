from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class CameraMotion(str, Enum):
    static     = "static"
    pan_left   = "pan_left"
    pan_right  = "pan_right"
    zoom_in    = "zoom_in"
    zoom_out   = "zoom_out"
    tilt_up    = "tilt_up"
    tilt_down  = "tilt_down"

class StylePreset(str, Enum):
    cinematic  = "cinematic"
    anime      = "anime"
    realistic  = "realistic"
    watercolor = "watercolor"
    neon       = "neon"

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    negative_prompt: str = Field(
        default="blurry, low quality, watermark, text, cropped, worst quality"
    )
    duration_seconds: int = Field(default=3, ge=2, le=10)
    camera_motion: CameraMotion = CameraMotion.static
    style: StylePreset = StylePreset.cinematic
    seed: Optional[int] = None          # None = random, set for reproducibility
    use_svd: bool = False               # True = use img2video path (needs image)
    image_base64: Optional[str] = None # Base64 PNG/JPG for SVD or ControlNet

class GenerateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "error"]
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "error"]
    progress: int = 0     # 0-100
    video_url: Optional[str] = None
    error: Optional[str] = None

class ExpandedPrompt(BaseModel):
    scene: str
    subject: str
    style: str
    camera: str
    lighting: str
    negative: str
    full_positive: str    # assembled prompt ready for diffuser