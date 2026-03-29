"""
Temporal Consistency Module.

PROBLEM: Diffusion models generate each frame somewhat independently.
Even with AnimateDiff's temporal attention, you still get "flickering"
— subtle changes in texture, color, or edges between frames.

THREE TECHNIQUES (applied in order, cheapest first):

1. SEED LOCKING (happens in generation_engine.py)
   - Same random seed → same noise initialization → consistent textures
   - Cost: zero. Just set generator once and reuse.
   - Limitation: only works within a single pipeline call. Multi-shot
     sequences lose consistency across clips.

2. LATENT REUSE / KEYFRAME ANCHORING
   - Generate a "keyframe" image at full quality.
   - Feed its latent as a soft prior into neighboring frames.
   - AnimateDiff does a variant of this natively via temporal attention.
   - We add an explicit weighted blend here for extra control.

3. OPTICAL FLOW-GUIDED BLENDING (post-processing)
   - After all frames are generated, we compute optical flow between
     consecutive frames (using OpenCV Farneback).
   - Warp frame N toward frame N+1 and alpha-blend with the raw frame.
   - This averages out per-pixel flicker without smearing motion.
   - Cost: ~50ms per frame pair on CPU.

TRADEOFFS:
   - Technique 3 can slightly soften fast motion.
   - For high-motion prompts ("explosion", "waterfall"), reduce alpha to 0.2.
   - For slow/static prompts, 0.4 gives clean results.

ALTERNATIVE: EbSynth (per-frame style transfer guided by keyframes).
   Much better quality, but requires a separate binary and 10-20s/frame.
   Worth adding in Phase 2 for professional output.
"""

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

FLOW_BLEND_ALPHA = 0.35  # how much to blend warped frame into raw frame


class TemporalConsistencyModule:
    def apply(self, frames: list[Image.Image]) -> list[Image.Image]:
        """
        Apply temporal consistency to a list of PIL frames.
        Falls back to simpler blending if OpenCV is not installed.
        """
        if len(frames) <= 1:
            return frames

        if HAS_CV2:
            return self._flow_guided_blend(frames)
        else:
            return self._simple_blend(frames)

    def _flow_guided_blend(self, frames: list[Image.Image]) -> list[Image.Image]:
        """Optical flow warp + blend. Reduces flicker while preserving motion."""
        arr = [np.array(f, dtype=np.uint8) for f in frames]
        out = [arr[0].copy()]

        for i in range(1, len(arr)):
            prev_gray = cv2.cvtColor(arr[i - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(arr[i],     cv2.COLOR_RGB2GRAY)

            # Farneback optical flow: prev → curr
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            h, w = flow.shape[:2]
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)

            # Warp previous frame toward current position
            warped_prev = cv2.remap(
                arr[i - 1], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            # Blend: mostly current frame, a bit of warped-previous
            blended = cv2.addWeighted(
                arr[i],        1.0 - FLOW_BLEND_ALPHA,
                warped_prev,   FLOW_BLEND_ALPHA,
                0
            )
            out.append(blended)

        return [Image.fromarray(f) for f in out]

    def _simple_blend(self, frames: list[Image.Image]) -> list[Image.Image]:
        """
        Fallback: simple linear blend between consecutive frames.
        Less accurate than optical flow but needs no OpenCV.
        """
        arr = [np.array(f, dtype=np.float32) for f in frames]
        out = [frames[0]]

        for i in range(1, len(arr)):
            blended = (arr[i] * (1 - FLOW_BLEND_ALPHA / 2) +
                       arr[i - 1] * (FLOW_BLEND_ALPHA / 2))
            out.append(Image.fromarray(blended.clip(0, 255).astype(np.uint8)))

        return out


def anchor_keyframe(
    frames: list[Image.Image],
    keyframe_idx: int = 0,
    strength: float = 0.15,
) -> list[Image.Image]:
    """
    Color-grade all frames toward a reference keyframe.
    Corrects per-frame color drift by pulling mean/std toward the anchor.

    This is a cheap substitute for proper histogram matching.
    For Phase 2, use skimage.exposure.match_histograms for better results.
    """
    anchor = np.array(frames[keyframe_idx], dtype=np.float32)
    anchor_mean = anchor.mean(axis=(0, 1))
    anchor_std  = anchor.std(axis=(0, 1))

    result = []
    for frame in frames:
        arr = np.array(frame, dtype=np.float32)
        f_mean = arr.mean(axis=(0, 1))
        f_std  = arr.std(axis=(0, 1)) + 1e-6

        # Normalize to anchor statistics
        normalized = (arr - f_mean) / f_std * anchor_std + anchor_mean
        blended    = arr * (1 - strength) + normalized * strength
        result.append(Image.fromarray(blended.clip(0, 255).astype(np.uint8)))

    return result