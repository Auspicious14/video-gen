"""
Rendering Pipeline.

Responsibility: Take a directory of PNG frames and produce a final MP4.

PIPELINE STEPS:
  1. (Optional) Upscale frames with Real-ESRGAN
  2. (Optional) FILM frame interpolation — doubles FPS, removes jitter
  3. FFmpeg: stitch frames → H.264 MP4 at target FPS
  4. Cleanup temp frames

WHY FFmpeg over imageio/moviepy?
  FFmpeg gives us fine-grained control over codec params (CRF, preset,
  pixel format). For web delivery, yuv420p + H.264 CRF 18 is the gold
  standard. imageio is fine for dev but produces larger, less compatible
  files.

WHY Real-ESRGAN?
  AnimateDiff and SVD run at 512px for VRAM budget. Real-ESRGAN 2× or 4×
  brings output to 1024–2048px. The quality improvement is dramatic.
  Cost: ~0.5s per frame on a T4.
"""

import subprocess
import shutil
import logging
from pathlib import Path
from typing import Callable

from config import OUTPUT_DIR, FPS, ENABLE_UPSCALE, UPSCALE_FACTOR

logger = logging.getLogger(__name__)


def render_video(
    frames_dir: Path,
    job_id: str,
    progress_cb: Callable[[int], None],
    fps: int = FPS,
) -> Path:
    """
    Master render function. Returns path to the output MP4.
    """
    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    # Handle fal.ai pre-rendered video (single MP4 file)
    fal_mp4 = frames_dir / "fal_output.mp4"
    if fal_mp4.exists():
        out_path = OUTPUT_DIR / f"{job_id}.mp4"
        shutil.copy(fal_mp4, out_path)
        progress_cb(100)
        return out_path

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    progress_cb(92)

    if ENABLE_UPSCALE:
        logger.info("Upscaling frames...")
        frame_paths = _upscale_frames(frames_dir, frame_paths, progress_cb)
        progress_cb(96)

    out_path = OUTPUT_DIR / f"{job_id}.mp4"
    _ffmpeg_stitch(frames_dir, out_path, fps)

    progress_cb(100)
    return out_path


def _ffmpeg_stitch(frames_dir: Path, out_path: Path, fps: int) -> None:
    """
    Stitch PNG frames → H.264 MP4 with web-compatible encoding.

    Key flags:
      -crf 18       : near-lossless quality (lower = better, larger file)
      -preset slow  : best compression at this CRF (use 'fast' on weak CPU)
      -pix_fmt      : yuv420p required for browser/mobile compatibility
      -vf scale     : ensures even dimensions (required by H.264)
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(frames_dir / "frame_*.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dimensions
        "-movflags", "+faststart",  # puts metadata at start for web streaming
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
    logger.info(f"Video saved: {out_path}")


def _upscale_frames(
    frames_dir: Path,
    frame_paths: list[Path],
    progress_cb: Callable,
) -> list[Path]:
    """
    Upscale using Real-ESRGAN via subprocess (assumes realesrgan-ncnn-vulkan
    or the Python binding is installed).

    ALTERNATIVE: Run basicsr Python binding inline — more integration,
    more dependencies. Subprocess keeps the main process clean.

    LIMITATION: Adds 2-10s per frame depending on hardware.
    For production: run upscaling as a background job after preview delivery.
    """
    upscale_dir = frames_dir / "upscaled"
    upscale_dir.mkdir(exist_ok=True)

    for i, src in enumerate(frame_paths):
        dst = upscale_dir / src.name
        cmd = [
            "realesrgan-ncnn-vulkan",
            "-i", str(src),
            "-o", str(dst),
            "-n", "realesrgan-x4plus",
            "-s", str(UPSCALE_FACTOR),
            "-f", "png",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning(f"Upscale failed for {src.name}, using original.")
            shutil.copy(src, dst)
        progress_cb(92 + int((i / len(frame_paths)) * 4))

    return sorted(upscale_dir.glob("frame_*.png"))


def cleanup_job_frames(frames_dir: Path) -> None:
    """Delete temp frames directory after render completes."""
    try:
        shutil.rmtree(frames_dir)
        logger.info(f"Cleaned up {frames_dir}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """
    Phase 2: Mix generated audio onto the video.
    Audio is truncated/padded to match video duration automatically.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",          # end when shorter stream ends
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio mix failed:\n{result.stderr}")
    return output_path