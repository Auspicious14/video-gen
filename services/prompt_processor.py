"""
Prompt Processing Layer — Gemini 2.5 Flash edition.

Gemini handles the semantic understanding that the rule-based system couldn't:
  - Extracts dominant emotional tone ("melancholic", "tense", "euphoric")
  - Picks appropriate diffusion tokens for abstract concepts
  - Generates scene-specific negative prompts
  - Falls back to rule-based logic if the API call fails or is skipped

SDK: google-genai (pip install google-genai)
Model: gemini-2.5-flash-preview-05-20
"""

import os
import json
import logging
from google import genai
from google.genai import types

from models import GenerateRequest, ExpandedPrompt, StylePreset, CameraMotion

logger = logging.getLogger(__name__)

# ── Gemini client (singleton) ─────────────────────────────────────────────────
_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Static lookup tables (used as fallback + LoRA routing) ───────────────────
STYLE_SUFFIXES: dict[StylePreset, str] = {
    StylePreset.cinematic:  "cinematic lighting, anamorphic lens, film grain, 4k, dramatic shadows",
    StylePreset.anime:      "anime style, studio ghibli, soft cel shading, vibrant colors",
    StylePreset.realistic:  "photorealistic, 8k uhd, sharp focus, natural light, DSLR",
    StylePreset.watercolor: "watercolor painting, soft edges, pastel tones, loose brushwork",
    StylePreset.neon:       "neon glow, cyberpunk, dark background, vivid lights, synthwave",
}

CAMERA_SUFFIXES: dict[CameraMotion, str] = {
    CameraMotion.static:    "static camera, locked shot",
    CameraMotion.pan_left:  "smooth camera pan left",
    CameraMotion.pan_right: "smooth camera pan right",
    CameraMotion.zoom_in:   "slow zoom in, dolly push",
    CameraMotion.zoom_out:  "slow zoom out, pull back",
    CameraMotion.tilt_up:   "camera tilt upward, reveal shot",
    CameraMotion.tilt_down: "camera tilt downward",
}

MOTION_LORA_MAP: dict[CameraMotion, str | None] = {
    CameraMotion.static:    None,
    CameraMotion.pan_left:  "guoyww/animatediff-motion-lora-pan-left",
    CameraMotion.pan_right: "guoyww/animatediff-motion-lora-pan-right",
    CameraMotion.zoom_in:   "guoyww/animatediff-motion-lora-zoom-in",
    CameraMotion.zoom_out:  "guoyww/animatediff-motion-lora-zoom-out",
    CameraMotion.tilt_up:   "guoyww/animatediff-motion-lora-tilt-up",
    CameraMotion.tilt_down: "guoyww/animatediff-motion-lora-tilt-down",
}

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a prompt engineer specialised in Stable Diffusion video generation.
Given a user's natural-language video idea, extract structured generation parameters.

Rules:
- style_suffix: 8-14 highly effective SD tokens for the requested style. No sentences.
- camera_suffix: 4-8 SD tokens describing the camera move. No sentences.
- lighting: 6-10 SD tokens. Be specific (direction, colour temperature, quality).
- subject: the primary visual subject in 3-6 words.
- scene: one short phrase (e.g. "urban exterior at night", "coastal cliffside at dawn").
- emotional_tone: the dominant mood in 2-5 words (e.g. "melancholic and isolated").
- negative: comma-separated list of things to avoid. Include quality negatives AND
  scene-specific ones (e.g. for night scenes add "overexposed, bright daylight").
- full_positive: the final assembled prompt ready to send to the diffusion model.
  Order: (subject:1.4), [user prompt], [emotional_tone tokens], [style_suffix], [camera_suffix],
  [lighting], high detail, masterpiece, best quality.
  Wrap the subject in (subject:1.4) to boost its attention weight in the diffusion model.

Respond ONLY with valid JSON. No markdown fences, no preamble, no explanation.
Schema:
{
  "scene": string,
  "subject": string,
  "style_suffix": string,
  "camera_suffix": string,
  "lighting": string,
  "emotional_tone": string,
  "negative": string,
  "full_positive": string
}
"""

# ── Main entry point ──────────────────────────────────────────────────────────

def expand_prompt(req: GenerateRequest) -> ExpandedPrompt:
    """
    Expand a raw user prompt using Gemini 2.5 Flash.
    Falls back to rule-based expansion on any error.
    """
    style_s  = STYLE_SUFFIXES[req.style]
    camera_s = CAMERA_SUFFIXES[req.camera_motion]

    try:
        expanded = _gemini_expand(req, style_s, camera_s)
        logger.info(f"Gemini expansion OK — tone: {expanded.get('emotional_tone', '—')}")
        return _to_model(expanded, req)
    except Exception as e:
        logger.warning(f"Gemini expansion failed ({e}), falling back to rule-based.")
        return _rule_based_fallback(req, style_s, camera_s)


# ── Gemini call ───────────────────────────────────────────────────────────────

def _gemini_expand(req: GenerateRequest, style_s: str, camera_s: str) -> dict:
    """
    Single Gemini 2.5 Flash call with JSON response. ~600-900ms round trip.
    Uses thinking disabled (budgetTokens=0) — we need speed, not reasoning here.
    """
    client = _get_client()

    user_message = (
        f"User prompt: {req.prompt}\n"
        f"Style preset: {req.style.value}\n"
        f"Camera motion: {req.camera_motion.value}\n"
        f"Style suffix to use: {style_s}\n"
        f"Camera suffix to use: {camera_s}\n"
        f"Duration: {req.duration_seconds} seconds\n"
        f"Negative prompt base: {req.negative_prompt}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.4,        # low — we want consistent structured output
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0   # disable thinking for speed
            ),
        ),
    )

    raw = response.text.strip()

    # Strip accidental markdown fences if the model slips
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Model builder ─────────────────────────────────────────────────────────────

def _to_model(data: dict, req: GenerateRequest) -> ExpandedPrompt:
    return ExpandedPrompt(
        scene=data.get("scene", "general scene"),
        subject=data.get("subject", req.prompt[:40]),
        style=data.get("style_suffix", STYLE_SUFFIXES[req.style]),
        camera=data.get("camera_suffix", CAMERA_SUFFIXES[req.camera_motion]),
        lighting=data.get("lighting", "natural light"),
        negative=data.get("negative", req.negative_prompt),
        full_positive=data.get("full_positive", req.prompt),
    )


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_fallback(
    req: GenerateRequest,
    style_s: str,
    camera_s: str,
) -> ExpandedPrompt:
    lighting = _infer_lighting(req.prompt)
    full_positive = (
        f"{req.prompt}, {style_s}, {camera_s}, "
        f"{lighting}, high detail, masterpiece, best quality"
    )
    return ExpandedPrompt(
        scene=_parse_scene(req.prompt),
        subject=" ".join(req.prompt.split()[:4]),
        style=style_s,
        camera=camera_s,
        lighting=lighting,
        negative=req.negative_prompt,
        full_positive=full_positive,
    )


def _parse_scene(prompt: str) -> str:
    lower = prompt.lower()
    if any(w in lower for w in ["city", "street", "urban", "market", "bus stop"]):
        return "urban exterior"
    if any(w in lower for w in ["forest", "jungle", "nature", "tree"]):
        return "natural exterior"
    if any(w in lower for w in ["room", "office", "kitchen", "indoor"]):
        return "interior"
    if any(w in lower for w in ["ocean", "sea", "beach", "coast"]):
        return "coastal exterior"
    return "general scene"


def _infer_lighting(prompt: str) -> str:
    lower = prompt.lower()
    if any(w in lower for w in ["night", "dark", "midnight", "3am"]):
        return "moody night lighting, ambient glow, deep shadows"
    if any(w in lower for w in ["sunset", "golden hour", "dusk"]):
        return "golden hour lighting, warm amber tones"
    if any(w in lower for w in ["morning", "sunrise", "dawn"]):
        return "soft morning light, cool blue tones"
    return "natural daylight, soft diffused shadows"