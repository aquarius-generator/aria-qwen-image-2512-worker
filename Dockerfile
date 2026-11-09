# Qwen-Image-2512 (Apache-2.0) at FP8, as a RunPod Serverless worker.
#
# WHAT CHANGED AND WHY, because this is the whole fix:
# The first build failed on RunPod's 1800 second ceiling. The log went quiet for about 27 minutes after pip
# finished, which is exactly what baking the weights in looks like. Measured afterwards against the Hugging
# Face API: the pinned revision is **57.7 GB across 30 files**. There is no build tuning that fits that
# inside 30 minutes, and it would not have fit on the 60 GB container disk next to the image in any case.
#
# So the image is now CODE AND DEPENDENCIES ONLY. The weights are fetched once on first start into a cache
# that every later start reuses. That turns a 57.7 GB cost from something paid on every build into something
# paid once per cache.
#
# THE PINS STAY. A tag or a branch can be re-pointed upstream, and a blind comparison whose contestant
# changed between the owner's first sitting and her second proves nothing. The revision below is an
# immutable commit, it is echoed on every response, and it is the only thing the handler will load.

FROM runpod/base:0.6.3-cuda12.4.1

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DIFFUSERS_VERBOSITY=error

# The immutable weights reference. Named here, in the fetch step and in the handler, and echoed on every
# response, so the four can never silently disagree about which model answered.
ENV QWEN_MODEL_ID="Qwen/Qwen-Image-2512" \
    QWEN_MODEL_REVISION="25468b98e3276ca6700de15c6628e51b7de54a26"

# Preferred cache, used when a network volume is attached. RunPod mounts a network volume at
# /runpod-volume. `weights.py` falls back to the container disk when that path is not there, so the image
# works with or without a volume and says which one it used.
ENV QWEN_VOLUME_CACHE="/runpod-volume/hf" \
    QWEN_FALLBACK_CACHE="/app/hf-cache"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# NOTE: there is deliberately NO `RUN python3 weights.py` here. That line is what made the first build fail,
# and putting it back would reintroduce a 57.7 GB download inside a 1800 second budget.
COPY weights.py /app/weights.py
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
