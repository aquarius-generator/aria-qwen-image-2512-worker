"""RunPod Serverless handler for Qwen-Image-2512 at FP8.

THE CONTRACT IS THE POINT. The caller is ARIA's `model-lab/qwen-image.js`, which sends a finished prompt and
reads `images[0].data`. The field names here match that client exactly: a worker that returns a differently
shaped payload fails at the END of a paid generation, which is the most expensive place to fail.

WEIGHTS ARE FETCHED ON FIRST START, NOT BAKED INTO THE IMAGE. 57.7 GB does not download inside RunPod's
1800 second build ceiling, which is what killed the first build. `weights.ensure_weights()` puts them on a
network volume when one is attached, so the download happens once for the endpoint rather than once per
worker, and the location used is echoed on every response.

THE WARM-UP RUNS AT IMPORT, DURING WORKER INIT, NOT INSIDE THE FIRST JOB. The execution timeout bounds how
long one REQUEST may run; it is not a budget for bringing a worker up. Doing a 57.7 GB fetch inside the
first job spends that budget on a download and then kills the request that paid for it.

FP8 IS APPLIED TO THE TRANSFORMER ONLY. That is where almost all the parameters are. The text encoder and
VAE stay at bfloat16: quantising them saves little and costs visible quality, which would turn this into a
test of a quantisation choice rather than of the model.

MEMORY PLACEMENT IS NOT A CHANGE OF MODEL. A 20B transformer in bf16 is roughly 45 GB and does not fit a
24 GB card, so the pipeline runs with model CPU offload, falling back to sequential offload if that is
still too large. The model and the quantisation are untouched by that choice, and what was actually used is
reported in `offload` and `quantisation` rather than left to be inferred.
"""
import base64
import io
import os
import time
import traceback

import runpod
import torch
from diffusers import DiffusionPipeline

import weights

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2512")
MODEL_REVISION = os.environ.get("QWEN_MODEL_REVISION", "25468b98e3276ca6700de15c6628e51b7de54a26")

_pipe = None
_quantisation = "none"
_offload = "none"
_cache_kind = "unknown"
_fetch_seconds = 0.0
_load_seconds = 0.0
_init_error = None


def _load():
    """Fetch (once), load the pinned pipeline, quantise the transformer, and place it in memory."""
    global _pipe, _quantisation, _offload, _cache_kind, _fetch_seconds, _load_seconds
    if _pipe is not None:
        return _pipe

    local_path, _cache_kind, _fetch_seconds = weights.ensure_weights()
    print(f"[worker] weights ready at {local_path} ({_cache_kind}) in {_fetch_seconds}s", flush=True)

    started = time.time()
    # Loaded from the resolved local snapshot, so this can never quietly resolve to a newer revision.
    # diffusers reads model_index.json and constructs QwenImagePipeline from it.
    pipe = DiffusionPipeline.from_pretrained(local_path, torch_dtype=torch.bfloat16)

    try:
        from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_

        quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
        _quantisation = "float8_dynamic"
        print("[worker] transformer quantised to float8", flush=True)
    except Exception as exc:  # noqa: BLE001
        # Reported, never silent. A worker that fell back to bfloat16 without saying so would be a different
        # contestant from the one the comparison is named after, and every output would carry the wrong
        # label. Echoing it is what makes that checkable from the caller.
        _quantisation = f"bfloat16 (float8 unavailable: {type(exc).__name__}: {exc})"
        print(f"[worker] FP8 quantisation unavailable, running bfloat16: {exc}", flush=True)

    # The transformer alone is far larger than a 24 GB card, so it is never resident in full. Offload moves
    # submodules on and off the GPU around each forward pass.
    try:
        pipe.enable_model_cpu_offload()
        _offload = "model_cpu_offload"
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] model cpu offload unavailable ({exc}); using sequential offload", flush=True)
        pipe.enable_sequential_cpu_offload()
        _offload = "sequential_cpu_offload"

    try:
        pipe.enable_attention_slicing()
    except Exception:  # noqa: BLE001
        pass

    pipe.set_progress_bar_config(disable=True)
    _load_seconds = round(time.time() - started, 1)
    print(f"[worker] pipeline ready in {_load_seconds}s ({_quantisation}, {_offload})", flush=True)
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

        cold = _pipe is None
        pipe = _load()
        generator = torch.Generator(device="cpu").manual_seed(seed)

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
            "offload": _offload,
            "weights_cache": _cache_kind,
            "cold_start": cold,
            "weights_fetch_seconds": _fetch_seconds,
            "pipeline_load_seconds": _load_seconds,
            "latency_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


# WARM AT IMPORT. RunPod imports this module while the worker is initialising and before it is handed a
# job, which is exactly where a long one-off fetch belongs. A failure here is printed in full and left for
# the first request to surface as a plain error, rather than being swallowed into a worker that looks
# healthy and fails every job it touches.
try:
    _load()
except Exception as _exc:  # noqa: BLE001
    _init_error = f"{type(_exc).__name__}: {_exc}"
    print(f"[worker] INIT FAILED: {_init_error}", flush=True)
    traceback.print_exc()

runpod.serverless.start({"handler": handler})
