"""RunPod Serverless handler for Qwen-Image-2512 at FP8.

THE CONTRACT IS THE POINT. The caller is ARIA's `model-lab/qwen-image.js`, which sends a finished prompt and
reads `images[0].data`. The field names here match that client exactly — a worker that returns a differently
shaped payload fails at the END of a paid generation, which is the most expensive place to fail.

FP8 IS APPLIED TO THE TRANSFORMER ONLY. That is where almost all the parameters are, and it is the part
whose footprint decides whether this fits on a 24GB card. The text encoder and VAE stay at bfloat16:
quantising them saves little and costs visible quality, which would turn this into a test of a quantisation
choice rather than of the model.

THE MODEL LOADS ONCE, at import, not per request. A serverless worker is reused for the whole of its idle
window, so per-request loading would pay the heaviest cost on every single call.
"""
import base64
import io
import os
import time
import traceback

import runpod
import torch
from diffusers import DiffusionPipeline

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2512")
MODEL_REVISION = os.environ.get("QWEN_MODEL_REVISION", "25468b98e3276ca6700de15c6628e51b7de54a26")

_pipe = None
_quantisation = "none"


def _load():
    """Load the pinned pipeline and quantise the transformer to float8."""
    global _pipe, _quantisation
    if _pipe is not None:
        return _pipe

    print(f"[worker] loading {MODEL_ID} @ {MODEL_REVISION}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        # The weights are baked into the image. Refusing the network at run time means a cold start can
        # never quietly resolve to something newer than what was pinned at build.
        local_files_only=True,
    )
    pipe = pipe.to("cuda")

    try:
        from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_

        quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
        _quantisation = "float8_dynamic"
        print("[worker] transformer quantised to float8")
    except Exception as exc:  # noqa: BLE001
        # Reported, never silent. A worker that fell back to bfloat16 without saying so would be a different
        # contestant from the one the comparison is named after, and every output would carry the wrong
        # label. Echoing it in the response is what makes that checkable from the caller.
        _quantisation = f"bfloat16 (float8 unavailable: {exc})"
        print(f"[worker] FP8 quantisation unavailable, running bfloat16: {exc}")

    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    return _pipe


def _round16(value, default):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    n = max(256, min(2048, n))
    return n - (n % 16)


def handler(job):
    started = time.time()
    try:
        payload = job.get("input") or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return {"error": "A prompt is required."}

        width = _round16(payload.get("width"), 1328)
        height = _round16(payload.get("height"), 1328)
        steps = int(payload.get("steps") or payload.get("num_inference_steps") or 30)
        cfg = float(payload.get("true_cfg_scale") or payload.get("cfg") or 4.0)
        seed = int(payload.get("seed") if payload.get("seed") is not None else 12)
        fmt = str(payload.get("output_format") or "png").lower()
        fmt = "JPEG" if fmt in ("jpg", "jpeg") else "PNG"

        pipe = _load()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=payload.get("negative_prompt") or "",
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Bytes, never a URL. The caller stores these permanently, and a signed link would expire long
        # before the comparison is judged.
        return {
            "images": [{"data": data, "mime_type": "image/png" if fmt == "PNG" else "image/jpeg"}],
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "quantisation": _quantisation,
            "latency_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
