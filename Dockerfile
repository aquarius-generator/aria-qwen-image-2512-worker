# Qwen-Image-2512 (Apache-2.0) at FP8, as a RunPod Serverless worker.
#
# TWO DELIBERATE CHOICES, both about making the thing reproducible:
#
# 1. THE WEIGHTS ARE BAKED IN AT BUILD TIME, pinned to an immutable commit. Downloading tens of gigabytes on
#    every cold start would make the first request of each idle period slow AND expensive, because a cold
#    start on a scale-to-zero endpoint is billed as GPU-seconds just like useful work. Baking them in also
#    means the model cannot change underneath a comparison that is already being judged.
#
# 2. A VERSION-PINNED BASE AND PINNED PIP PACKAGES. A worker whose dependency set drifts is a contestant
#    that drifts, and the whole point of this is a fair comparison between two fixed things.

FROM runpod/base:0.6.3-cuda12.4.1

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DIFFUSERS_VERBOSITY=error

# The immutable weights reference. Named here, in the download step and in the handler, and echoed on every
# response, so the four can never silently disagree about which model answered.
ENV QWEN_MODEL_ID="Qwen/Qwen-Image-2512" \
    QWEN_MODEL_REVISION="25468b98e3276ca6700de15c6628e51b7de54a26"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Pre-fetch the pinned weights into the image. The handler then runs with local_files_only, so a cold start
# cannot quietly reach the network and resolve to something newer than what was pinned here.
COPY download_weights.py /app/download_weights.py
RUN python3 /app/download_weights.py

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
