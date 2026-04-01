"""
Generation Engine.

Inference path priority:
  1. Local CUDA GPU  — full quality, fastest
  2. Local MPS       — Apple Silicon, ~15-30 min per video
  3. Local CPU       — last resort, ~50 min
  4. DEV_MODE        — instant color gradient, no model needed
"""

import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from config import (
    ANIMATEDIFF_MOTION_ADAPTER, BASE_SD_MODEL,
    NUM_FRAMES, INFERENCE_STEPS,
    GUIDANCE_SCALE, WIDTH, HEIGHT, DEV_MODE,
)
from models import GenerateRequest, ExpandedPrompt
from services.prompt_processor import MOTION_LORA_MAP
from services.temporal import TemporalConsistencyModule

logger = logging.getLogger(__name__)

_animatediff_pipe = None


# ── Device resolution ─────────────────────────────────────────────────────────

def _detect_best_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32   # float16 causes VAE artifacts on MPS
    return "cpu", torch.float32


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> None:
    global _animatediff_pipe

    if DEV_MODE:
        logger.warning("DEV_MODE=true — skipping model loading.")
        return

    infer_device, dtype = _detect_best_device()
    logger.info(f"Using device: {infer_device} | dtype: {dtype}")

    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.models.attention_processor import AttnProcessor2_0

    logger.info("Loading MotionAdapter...")
    adapter = MotionAdapter.from_pretrained(
        ANIMATEDIFF_MOTION_ADAPTER,
        torch_dtype=dtype,
    )

    logger.info(f"Loading base model: {BASE_SD_MODEL}...")
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

    if infer_device == "mps":
        _animatediff_pipe = _animatediff_pipe.to("mps")
        _animatediff_pipe.unet.set_attn_processor(AttnProcessor2_0())
        _animatediff_pipe.enable_vae_slicing()
        _animatediff_pipe.enable_attention_slicing(4)
    elif infer_device == "cuda":
        _animatediff_pipe.enable_vae_slicing()
        _animatediff_pipe.enable_attention_slicing(1)
        _animatediff_pipe.enable_model_cpu_offload()
    else:
        _animatediff_pipe = _animatediff_pipe.to("cpu")
        _animatediff_pipe.enable_vae_slicing()
        _animatediff_pipe.enable_attention_slicing(1)

    logger.info("All models loaded.")


# ── Motion LoRA ───────────────────────────────────────────────────────────────

def _load_motion_lora(camera_motion) -> None:
    lora_id = MOTION_LORA_MAP.get(camera_motion)
    if not lora_id or not _animatediff_pipe:
        return
    try:
        _animatediff_pipe.load_lora_weights(lora_id, adapter_name="motion_lora")
        _animatediff_pipe.set_adapters(["motion_lora"], [0.8])
    except Exception as e:
        logger.warning(f"Motion LoRA load failed ({e}), continuing without it.")


# ── Core inference ────────────────────────────────────────────────────────────

def run_animatediff(
    req: GenerateRequest,
    expanded: ExpandedPrompt,
    frames_dir: Path,
    progress_cb: Callable[[int], None],
) -> list[Path]:
    if DEV_MODE:
        return _dev_mode_frames(frames_dir, progress_cb)

    if _animatediff_pipe is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    seed = req.seed if req.seed is not None else torch.randint(0, 2**32, (1,)).item()
    actual_device = next(_animatediff_pipe.unet.parameters()).device.type
    gen_device = "cpu" if actual_device in ("mps", "cpu") else "cuda"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    _load_motion_lora(req.camera_motion)

    def _step_cb(pipe, i, t, kwargs):
        progress_cb(int((i / INFERENCE_STEPS) * 70))
        return kwargs

    progress_cb(5)
    output = _animatediff_pipe(
        prompt=expanded.full_positive,
        negative_prompt=expanded.negative,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=INFERENCE_STEPS,
        generator=generator,
        width=WIDTH,
        height=HEIGHT,
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pil",
    )

    frames: list[Image.Image] = output.frames[0]

    progress_cb(75)
    frames = TemporalConsistencyModule().apply(frames)

    frame_paths = []
    for idx, frame in enumerate(frames):
        path = frames_dir / f"frame_{idx:04d}.png"
        frame.save(path)
        frame_paths.append(path)

    progress_cb(90)
    return frame_paths


# ── Dev mode ──────────────────────────────────────────────────────────────────

def _dev_mode_frames(frames_dir: Path, progress_cb: Callable) -> list[Path]:
    import random
    r, g, b = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
    paths = []
    for i in range(NUM_FRAMES):
        progress_cb(int((i / NUM_FRAMES) * 90))
        ratio = i / max(NUM_FRAMES - 1, 1)
        color = (
            int(r * (1 - ratio) + 50 * ratio),
            int(g * (1 - ratio) + 180 * ratio),
            int(b * (1 - ratio) + 100 * ratio),
        )
        img  = Image.new("RGB", (WIDTH, HEIGHT), color)
        path = frames_dir / f"frame_{i:04d}.png"
        img.save(path)
        paths.append(path)
    return paths