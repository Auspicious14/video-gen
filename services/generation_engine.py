"""
Generation Engine — the core AI layer.

DESIGN DECISIONS:

1. WHY AnimateDiff over SVD as default?
   AnimateDiff runs on top of any SD 1.5 checkpoint. You can swap
   the base model for a community fine-tune (anime, realistic, etc.)
   without retraining. SVD is higher quality but is locked to its own
   checkpoint, costs more VRAM (24 GB for XT), and takes 2-3× longer.
   AnimateDiff fits on a T4 (16 GB) at fp16.

2. WHY ControlNet?
   Without structure control, successive frames drift (the "flickering"
   problem). ControlNet depth maps extracted from a reference frame act
   as a geometric anchor, reducing spatial drift. It adds ~2 GB VRAM.

3. WHY seed locking?
   Same seed across all denoising steps = same noise schedule = the
   diffusion "starts" from the same state every frame. This is the
   cheapest form of temporal consistency. Combined with AnimateDiff's
   temporal attention it's surprisingly effective at under 512px.

LIMITATIONS:
   - At > 512px resolution, VRAM requirements spike fast.
   - AnimateDiff maxes out at ~24 frames reliably. Longer videos need
     multi-shot stitching (Phase 3).
   - SVD path requires at least one input image.

FAL.AI FALLBACK:
   When no GPU is available locally (or for higher quality), the engine
   falls back to fal.ai's hosted AnimateDiff endpoint. Costs ~$0.005/video.
"""

import io
import logging
import base64
from pathlib import Path
from typing import Callable

import torch
import numpy as np
from PIL import Image

from config import (
    ANIMATEDIFF_MOTION_ADAPTER, BASE_SD_MODEL, SVD_MODEL,
    CONTROLNET_MODEL, DEVICE, NUM_FRAMES, INFERENCE_STEPS,
    GUIDANCE_SCALE, WIDTH, HEIGHT, DEV_MODE, FAL_API_KEY
)
from models import GenerateRequest, ExpandedPrompt
from services.prompt_processor import MOTION_LORA_MAP
from services.temporal import TemporalConsistencyModule

logger = logging.getLogger(__name__)

# ── Model registry (loaded once at startup) ───────────────────────────────────
_animatediff_pipe = None
_svd_pipe         = None
_controlnet       = None

def load_models() -> None:
    """
    Call this once on FastAPI startup. Loads to GPU in fp16.
    On CPU / DEV_MODE, skips entirely.
    """
    global _animatediff_pipe, _controlnet

    if DEV_MODE:
        logger.warning("DEV_MODE=true — skipping model loading.")
        return

    from diffusers import (
        AnimateDiffPipeline,
        MotionAdapter,
        DDIMScheduler,
        ControlNetModel,
    )

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    logger.info("Loading MotionAdapter...")
    adapter = MotionAdapter.from_pretrained(
        ANIMATEDIFF_MOTION_ADAPTER,
        torch_dtype=dtype
    )

    logger.info("Loading AnimateDiff pipeline...")
    _animatediff_pipe = AnimateDiffPipeline.from_pretrained(
        BASE_SD_MODEL,
        motion_adapter=adapter,
        torch_dtype=dtype,
    )
    _animatediff_pipe.scheduler = DDIMScheduler.from_config(
        _animatediff_pipe.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )

    # On MPS, scaled_dot_product_attention hits a 16 GB buffer cap.
    # Force the pipeline onto CPU — slower but stable on Mac.
    # On CUDA this block is skipped entirely.
    if DEVICE != "cuda":
        _animatediff_pipe = _animatediff_pipe.to("cpu")
        _animatediff_pipe.enable_attention_slicing(1)
        _animatediff_pipe.enable_vae_slicing()
    else:
        _animatediff_pipe.enable_vae_slicing()
        _animatediff_pipe.enable_attention_slicing(1)
        _animatediff_pipe.enable_model_cpu_offload()

    logger.info("Loading ControlNet (depth)...")
    _controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype
    )

    logger.info("All models loaded.")


def _load_motion_lora(camera_motion) -> None:
    """Load the appropriate motion LoRA for the requested camera move."""
    lora_id = MOTION_LORA_MAP.get(camera_motion)
    if not lora_id or not _animatediff_pipe:
        return
    try:
        _animatediff_pipe.load_lora_weights(lora_id, adapter_name="motion_lora")
        _animatediff_pipe.set_adapters(["motion_lora"], [0.8])
    except Exception as e:
        # LoRA load failure is non-fatal — continue without camera direction
        logger.warning(f"Motion LoRA load failed ({e}), continuing without it.")


def run_animatediff(
    req: GenerateRequest,
    expanded: ExpandedPrompt,
    frames_dir: Path,
    progress_cb: Callable[[int], None],
) -> list[Path]:
    """
    Core AnimateDiff inference. Returns list of frame paths.

    TEMPORAL CONSISTENCY STRATEGY:
    - Fixed seed across all denoising iterations (via generator)
    - AnimateDiff temporal attention handles frame-to-frame coherence natively
    - TemporalConsistencyModule applies optical-flow-guided blending post-gen
    """
    if DEV_MODE:
        return _dev_mode_frames(frames_dir, progress_cb)

    if _animatediff_pipe is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    # Seed locking — deterministic noise = consistent textures across frames
    seed = req.seed if req.seed is not None else torch.randint(0, 2**32, (1,)).item()
    # Generator must use CPU on MPS/CPU devices — CUDA generator only valid on CUDA
    gen_device = "cuda" if DEVICE == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    # Load motion LoRA for requested camera move
    _load_motion_lora(req.camera_motion)

    num_frames = NUM_FRAMES  # typically 16

    def _step_callback(pipe, i, t, kwargs):
        pct = int((i / INFERENCE_STEPS) * 70)  # reserve 30% for post-processing
        progress_cb(pct)
        return kwargs

    progress_cb(5)
    output = _animatediff_pipe(
        prompt=expanded.full_positive,
        negative_prompt=expanded.negative,
        num_frames=num_frames,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=INFERENCE_STEPS,
        generator=generator,
        width=WIDTH,
        height=HEIGHT,
        callback_on_step_end=_step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pil",
    )

    frames: list[Image.Image] = output.frames[0]

    # Apply temporal consistency post-processing
    progress_cb(75)
    tc = TemporalConsistencyModule()
    frames = tc.apply(frames)

    # Save frames to disk
    frame_paths = []
    for idx, frame in enumerate(frames):
        path = frames_dir / f"frame_{idx:04d}.png"
        frame.save(path)
        frame_paths.append(path)

    progress_cb(90)
    return frame_paths


def run_svd(
    req: GenerateRequest,
    expanded: ExpandedPrompt,
    frames_dir: Path,
    progress_cb: Callable[[int], None],
) -> list[Path]:
    """
    Stable Video Diffusion path — used when image_base64 is provided.
    SVD generates temporally smooth video from a single image.
    Requires ~24 GB VRAM for XT. Falls back to fal.ai if no local GPU.

    WHY SVD here? The user provided a reference image. SVD will animate
    it more faithfully than prompting AnimateDiff from scratch.
    """
    if not req.image_base64:
        raise ValueError("SVD path requires image_base64 in the request.")

    if DEV_MODE:
        return _dev_mode_frames(frames_dir, progress_cb)

    # Check VRAM — SVD XT needs ~24 GB, SVD base needs ~14 GB
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 13:
            logger.warning(f"Only {vram_gb:.1f} GB VRAM. Routing SVD to fal.ai.")
            return _fal_fallback(req, expanded, frames_dir, progress_cb)

    from diffusers import StableVideoDiffusionPipeline

    global _svd_pipe
    if _svd_pipe is None:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        _svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            SVD_MODEL,
            torch_dtype=dtype,
            variant="fp16",
        )
        _svd_pipe.enable_model_cpu_offload()

    # Decode reference image
    img_bytes = base64.b64decode(req.image_base64)
    ref_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    ref_image = ref_image.resize((WIDTH, HEIGHT))

    seed = req.seed if req.seed is not None else torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    progress_cb(10)
    frames = _svd_pipe(
        ref_image,
        decode_chunk_size=8,   # lower = less VRAM
        generator=generator,
        num_frames=NUM_FRAMES,
        num_inference_steps=INFERENCE_STEPS,
        motion_bucket_id=127,  # 0=static … 255=very dynamic
        fps=7,
    ).frames[0]

    frame_paths = []
    for idx, frame in enumerate(frames):
        path = frames_dir / f"frame_{idx:04d}.png"
        frame.save(path)
        frame_paths.append(path)

    progress_cb(90)
    return frame_paths


def _fal_fallback(req, expanded, frames_dir, progress_cb) -> list[Path]:
    """
    fal.ai hosted inference fallback for when local GPU is insufficient.
    Uses fal-client SDK. API key set in .env as FAL_API_KEY.
    Cost: ~$0.004 per video at 512px, 16 frames.
    """
    if not FAL_API_KEY:
        raise RuntimeError("No local GPU and FAL_API_KEY not set. Cannot generate video.")

    import fal_client

    progress_cb(10)
    result = fal_client.run(
        "fal-ai/animatediff-v2v",
        arguments={
            "prompt": expanded.full_positive,
            "negative_prompt": expanded.negative,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": req.seed or 42,
            "width": WIDTH,
            "height": HEIGHT,
        }
    )

    # Download frames from the returned URL
    import httpx
    video_url = result["video"]["url"]
    progress_cb(70)

    # fal returns an MP4 — write it directly, skip frame extraction
    video_bytes = httpx.get(video_url).content
    out_path = frames_dir / "fal_output.mp4"
    out_path.write_bytes(video_bytes)

    # Signal that we got a complete video (renderer will detect this)
    return [out_path]


def _dev_mode_frames(frames_dir: Path, progress_cb: Callable) -> list[Path]:
    """Returns 16 solid-color gradient frames for UI testing without a GPU."""
    import random
    paths = []
    r, g, b = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
    for i in range(NUM_FRAMES):
        progress_cb(int((i / NUM_FRAMES) * 90))
        ratio  = i / max(NUM_FRAMES - 1, 1)
        color  = (int(r * (1 - ratio) + 50 * ratio),
                  int(g * (1 - ratio) + 180 * ratio),
                  int(b * (1 - ratio) + 100 * ratio))
        img    = Image.new("RGB", (WIDTH, HEIGHT), color)
        path   = frames_dir / f"frame_{i:04d}.png"
        img.save(path)
        paths.append(path)
    return paths