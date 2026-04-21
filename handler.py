"""
Momentum AI Creator — RunPod Serverless FLUX handler
Text-to-image only. No face reference, no NSFW bypass, no uploaded-photo img2img.

Character consistency is achieved via deterministic seed + detailed text description
passed from the frontend. Optional: load a user-trained LoRA from the network volume
(trained on synthetic outputs only — never on uploaded photos of real people).

Input schema (all optional except prompt):
    prompt: str                      # required
    negative_prompt: str             # default: quality boilerplate
    seed: int | None                 # for character consistency
    steps: int                       # default 24
    guidance: float                  # default 3.5
    width: int                       # default 1024
    height: int                      # default 1024
    batch_size: int                  # default 1
    scheduler: str                   # euler | euler_ancestral | dpm++ | heun
    style_preset: str | None         # "cinematic" | "editorial" | "concept_art" | ...
    enhance_prompt: bool             # default True (adds photoreal realism tags)
    character_lora: str | None       # filename in /runpod-volume/loras/<name>.safetensors
    character_lora_weight: float     # default 0.85
    upscale_factor: float            # 1.0 | 2.0
    max_sequence_length: int         # default 256

Output:
    { status, images: [{image: b64_png, seed, index}], metadata: {...} }
"""

import os
# Kill flash-attn before any torch/transformers import attempts to register
# its custom ops — otherwise we crash on torch.library.infer_schema with
# PEP 604 union types.
os.environ.setdefault("DIFFUSERS_NO_FLASH_ATTN", "1")
os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import sys
# Block any flash_attn import attempted later in the stack.
class _BlockedFlashAttn:
    def __getattr__(self, name):
        raise ImportError("flash_attn intentionally disabled")
for _name in ("flash_attn", "flash_attn_3", "flash_attn_interface"):
    sys.modules[_name] = _BlockedFlashAttn()  # type: ignore

import time
import base64
import runpod
import torch
from io import BytesIO
from PIL import Image, PngImagePlugin

from diffusers import (
    FluxPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
)

# ─────────────────────────────────────────────────────────────────────────────
# Perf flags
# ─────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
LORA_DIR = "/runpod-volume/loras"  # Optional user-trained LoRAs (synthetic-data only)

SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "heun": HeunDiscreteScheduler,
}

# ─────────────────────────────────────────────────────────────────────────────
# Style presets (prefix / suffix pairs injected around the user prompt)
# ─────────────────────────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "cinematic": {
        "prefix": "cinematic still, ",
        "suffix": (
            ", shot on ARRI Alexa, anamorphic lens flare, teal and orange grade, "
            "shallow depth of field, dramatic key light, 35mm film grain, "
            "cinematic composition, ultra sharp, 8K"
        ),
    },
    "editorial": {
        "prefix": "high fashion editorial photograph, ",
        "suffix": (
            ", Vogue cover styling, studio lighting, beauty dish, "
            "Hasselblad medium format, Kodak Portra 400, pristine skin texture, "
            "magazine retouching, ultra sharp focus"
        ),
    },
    "concept_art": {
        "prefix": "concept art, ",
        "suffix": (
            ", matte painting, volumetric lighting, atmospheric perspective, "
            "ArtStation trending, detailed environment design, 8K"
        ),
    },
    "product": {
        "prefix": "product photography, ",
        "suffix": (
            ", seamless studio backdrop, softbox lighting, commercial campaign, "
            "Hasselblad H6D, tack sharp, ultra detailed materials, 8K"
        ),
    },
    "portrait": {
        "prefix": "portrait photograph, ",
        "suffix": (
            ", Sony A7R V, 85mm f/1.4 GM lens, Rembrandt lighting, "
            "natural skin texture with visible pores, catch light in eyes, "
            "Kodak Portra 400, shallow depth of field, ultra sharp, 8K"
        ),
    },
    "anime": {
        "prefix": "anime illustration, ",
        "suffix": (
            ", cel shaded, vibrant colors, clean line art, "
            "detailed background, studio-quality animation key frame"
        ),
    },
    "raw": {"prefix": "", "suffix": ""},
}

DEFAULT_NEGATIVE = (
    "blurry, distorted, low quality, bad anatomy, deformed hands, extra fingers, "
    "extra limbs, watermark, text, signature, jpeg artifacts, low resolution"
)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model load — pipeline loads on first job, not at import.
# This lets the worker report healthy immediately so RunPod doesn't kill it
# during the FLUX weight download / VRAM load (can take 2–5 min).
# ─────────────────────────────────────────────────────────────────────────────
pipe = None
_CURRENT_LORA: str | None = None


def get_pipe():
    """Load the FLUX pipeline on first request, then cache globally."""
    global pipe
    if pipe is not None:
        return pipe

    print("[load] Importing FLUX.1-dev (first request)", flush=True)
    t0 = time.time()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("[load][WARN] HF_TOKEN not set in environment", flush=True)

    try:
        p = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        print(f"[load] Weights loaded ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[load][FATAL] from_pretrained failed: {e}", flush=True)
        raise

    if torch.cuda.is_available():
        try:
            p = p.to("cuda")
            print("[load] Moved to CUDA", flush=True)
        except Exception as e:
            print(f"[load] Direct CUDA load failed, falling back to cpu_offload: {e}", flush=True)
            p.enable_model_cpu_offload()
        try:
            p.vae.enable_tiling()
            p.vae.enable_slicing()
        except Exception as e:
            print(f"[load] vae tiling/slicing skipped: {e}", flush=True)
    else:
        print("[load][WARN] CUDA not available — CPU inference will be extremely slow", flush=True)

    pipe = p
    print(f"[load] Pipeline ready ({time.time()-t0:.1f}s total)", flush=True)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(prompt: str, style_preset: str | None, enhance: bool) -> str:
    if not enhance:
        return prompt
    preset = STYLE_PRESETS.get(style_preset or "portrait", STYLE_PRESETS["portrait"])
    return f"{preset['prefix']}{prompt}{preset['suffix']}"


def set_scheduler(name: str) -> None:
    p = get_pipe()
    if name in SCHEDULERS:
        p.scheduler = SCHEDULERS[name].from_config(p.scheduler.config)


def load_character_lora(lora_name: str | None, weight: float) -> bool:
    """
    Load a user-trained LoRA from the network volume.
    Filename only — no arbitrary paths, no uploads, no URLs.
    LoRAs must be trained on synthetic (model-generated) data only.
    """
    global _CURRENT_LORA
    p = get_pipe()

    if lora_name is None:
        if _CURRENT_LORA is not None:
            try:
                p.unload_lora_weights()
                _CURRENT_LORA = None
                print("[lora] unloaded", flush=True)
            except Exception as e:
                print(f"[lora] unload failed: {e}", flush=True)
        return True

    # Sanitize — only allow a bare filename, no traversal
    safe_name = os.path.basename(lora_name)
    if not safe_name.endswith(".safetensors"):
        safe_name += ".safetensors"

    lora_path = os.path.join(LORA_DIR, safe_name)
    if not os.path.isfile(lora_path):
        print(f"[lora] not found: {lora_path}", flush=True)
        return False

    if _CURRENT_LORA == safe_name:
        try:
            p.set_adapters(["character"], adapter_weights=[weight])
            return True
        except Exception:
            pass

    try:
        if _CURRENT_LORA is not None:
            p.unload_lora_weights()
        p.load_lora_weights(lora_path, adapter_name="character")
        p.set_adapters(["character"], adapter_weights=[weight])
        _CURRENT_LORA = safe_name
        print(f"[lora] loaded {safe_name} @ {weight}", flush=True)
        return True
    except Exception as e:
        print(f"[lora] load failed: {e}", flush=True)
        _CURRENT_LORA = None
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────
def handler(job):
    job_input = job.get("input", {}) or {}

    prompt = job_input.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return {"status": "error", "message": "prompt is required"}

    negative_prompt = job_input.get("negative_prompt") or DEFAULT_NEGATIVE
    steps = int(job_input.get("steps", 24))
    guidance = float(job_input.get("guidance", 3.5))
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    batch_size = max(1, min(4, int(job_input.get("batch_size", 1))))
    scheduler = job_input.get("scheduler", "euler")
    style_preset = job_input.get("style_preset", "portrait")
    enhance_prompt = bool(job_input.get("enhance_prompt", True))
    seq_length = int(job_input.get("max_sequence_length", 256))
    upscale_factor = float(job_input.get("upscale_factor", 1.0))
    seed = job_input.get("seed")

    character_lora = job_input.get("character_lora")
    character_lora_weight = float(job_input.get("character_lora_weight", 0.85))

    # Clamp sizes to sane bounds
    width = max(512, min(1536, (width // 8) * 8))
    height = max(512, min(1536, (height // 8) * 8))
    steps = max(8, min(60, steps))

    final_prompt = build_prompt(prompt, style_preset, enhance_prompt)
    p = get_pipe()
    set_scheduler(scheduler)
    load_character_lora(character_lora, character_lora_weight)

    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    seed = int(seed)

    print(f"📸 {batch_size}x {width}x{height} steps={steps} cfg={guidance} seed={seed}")
    print(f"   prompt: {final_prompt[:140]}…")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    start = time.time()

    images_out = []
    for i in range(batch_size):
        current_seed = seed + i
        gen.manual_seed(current_seed)

        try:
            with torch.inference_mode():
                result = p(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    max_sequence_length=seq_length,
                    generator=gen,
                )
            image = result.images[0]

            if upscale_factor > 1.0:
                new_size = (int(width * upscale_factor), int(height * upscale_factor))
                image = image.resize(new_size, Image.LANCZOS)

            buffered = BytesIO()
            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt", final_prompt)
            meta.add_text("negative_prompt", negative_prompt)
            meta.add_text("seed", str(current_seed))
            meta.add_text("steps", str(steps))
            meta.add_text("guidance", str(guidance))
            meta.add_text("scheduler", scheduler)
            meta.add_text("model", MODEL_ID)
            if character_lora:
                meta.add_text("character_lora", os.path.basename(character_lora))
            image.save(buffered, format="PNG", pnginfo=meta, compress_level=6)

            images_out.append(
                {
                    "image": base64.b64encode(buffered.getvalue()).decode("utf-8"),
                    "seed": current_seed,
                    "index": i,
                }
            )
        except Exception as e:
            print(f"❌ generation error at index {i}: {e}")
            return {"status": "error", "message": str(e), "partial": images_out}

    elapsed = round(time.time() - start, 2)
    print(f"✅ done in {elapsed}s ({batch_size} image(s))")

    return {
        "status": "success",
        "images": images_out,
        "metadata": {
            "execution_time_seconds": elapsed,
            "batch_size": batch_size,
            "scheduler": scheduler,
            "steps": steps,
            "guidance": guidance,
            "dimensions": f"{width}x{height}",
            "upscale_factor": upscale_factor,
            "style_preset": style_preset,
            "enhance_prompt": enhance_prompt,
            "character_lora": os.path.basename(character_lora) if character_lora else None,
            "prompt_used": final_prompt,
            "base_seed": seed,
        },
    }


runpod.serverless.start({"handler": handler})
