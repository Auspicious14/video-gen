import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
FRAMES_DIR = BASE_DIR / "tmp" / "frames"
OUTPUT_DIR = BASE_DIR / "tmp" / "output"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config ─────────────────────────────────────────────────────────────
# AnimateDiff v3 motion adapter (≈ 1.7 GB, loads once at startup)
ANIMATEDIFF_MOTION_ADAPTER = os.getenv(
    "ANIMATEDIFF_MOTION_ADAPTER",
    "guoyww/animatediff-motion-adapter-v1-5-3"
)
# Base SD 1.5 checkpoint used with AnimateDiff
BASE_SD_MODEL = os.getenv(
    "BASE_SD_MODEL",
    "SG161222/Realistic_Vision_V5.1_noVAE"
)
# Stable Video Diffusion for img2video path
SVD_MODEL = os.getenv(
    "SVD_MODEL",
    "stabilityai/stable-video-diffusion-img2vid-xt"
)
# ControlNet (openpose or depth) for structure control
CONTROLNET_MODEL = os.getenv(
    "CONTROLNET_MODEL",
    "lllyasviel/sd-controlnet-depth"
)

# ── Runtime config ────────────────────────────────────────────────────────────
def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # MPS hits a 16 GB buffer cap with AnimateDiff temporal attention — use CPU
    return "cpu"

DEVICE      = os.getenv("DEVICE") or _detect_device()
TORCH_DTYPE = "float16" if DEVICE == "cuda" else "float32"
NUM_FRAMES      = int(os.getenv("NUM_FRAMES", "8"))    # 16 → 8, halves attention buffer
FPS             = int(os.getenv("FPS", "8"))
WIDTH           = int(os.getenv("WIDTH", "256"))        # 512 → 256, quarters memory
HEIGHT          = int(os.getenv("HEIGHT", "256"))
INFERENCE_STEPS = int(os.getenv("INFERENCE_STEPS", "15"))  # 25 → 15, faster on Mac
GUIDANCE_SCALE  = float(os.getenv("GUIDANCE_SCALE", "7.5"))

# ── Upscaler ─────────────────────────────────────────────────────────────────
ENABLE_UPSCALE  = os.getenv("ENABLE_UPSCALE", "false").lower() == "true"
UPSCALE_FACTOR  = int(os.getenv("UPSCALE_FACTOR", "2"))    # 2× or 4×

# ── Dev mode: skip heavy inference, return a test video ───────────────────────
DEV_MODE        = os.getenv("DEV_MODE", "false").lower() == "true"

# ── API keys (optional fallback paths) ───────────────────────────────────────
FAL_API_KEY     = os.getenv("FAL_API_KEY", "")   # fal.ai fallback
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "") # GPT-4 prompt expansion fallback