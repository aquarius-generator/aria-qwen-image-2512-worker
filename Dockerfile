# Qwen-Image-2512 (Apache-2.0) at FP8, as a RunPod Serverless worker.
#
# THE BUILD BUDGET IS 1800 SECONDS AND THIS FILE HAS LOST TO IT TWICE, for two different reasons:
#
#   Build 1 baked the weights in. Measured against the Hugging Face API, the pinned revision is 57.7 GB
#   across 30 files. No build tuning fits that inside 30 minutes, and it would not have fitted on the 60 GB
#   container disk next to the image either. The weights now load on FIRST START into a cache that later
#   starts reuse, so that cost is paid once per cache instead of once per build.
#
#   Build 3 used ranged pip pins. The resolver backtracked across a tree containing torch and spent the
#   whole budget searching, with no weights involved at all. Every version in requirements.txt is now exact.
#
# So the image is CODE AND DEPENDENCIES ONLY, and the dependency install is arranged to be boring: wheels
# only, no resolution search, and torch in a layer of its own so a change to any other pin does not drag
# the largest download in the set along with it.
#
# THE PINS STAY. A tag or a branch can be re-pointed upstream, and a blind comparison whose contestant
# changed between the owner's first sitting and her second proves nothing. The revision below is an
# immutable commit, it is echoed on every response, and it is the only thing the handler will load.

FROM runpod/base:0.6.3-cuda12.4.1

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
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

# Recorded in the build log on purpose: diffusers 0.42 requires Python 3.11 or newer, and a base image that
# quietly ships something older would otherwise fail much later and much less clearly.
RUN python3 --version

# torch first, alone. It is the largest part of the install by a wide margin, and keeping it in its own
# layer means editing any other pin does not re-fetch it.
RUN python3 -m pip install --no-cache-dir --only-binary=:all: \
        torch==2.6.0 torchvision==0.21.0

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --only-binary=:all: -r /app/requirements.txt

# NOTE: there is deliberately NO weights fetch here. That is what made build 1 fail, and putting it back
# would reintroduce a 57.7 GB download inside a 1800 second budget.
COPY weights.py /app/weights.py
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
