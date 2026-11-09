# Qwen-Image-2512 (Apache-2.0) at FP8, as a RunPod Serverless worker.
#
# THIS FILE HAS LOST TO THE 1800 SECOND BUILD BUDGET TWICE AND TO A SILENT ERROR ONCE. Each cause was
# different, and each one is now closed:
#
#   Build 1, TIMEOUT: the weights were baked in. Measured against the Hugging Face API, the pinned revision
#   is 57.7 GB across 30 files. No build tuning fits that inside 30 minutes, and it would not have fitted
#   on the 60 GB container disk next to the image either. Weights now load on FIRST START into a cache that
#   later starts reuse, so the cost is paid once per cache rather than once per build.
#
#   Build 3, TIMEOUT: ranged pip pins. The resolver backtracked across a tree containing torch and spent
#   the whole budget searching, with no weights involved at all. Every version is now exact.
#
#   Build 4, FAILURE IN 9 MINUTES: the base image shipped Python 3.10 and diffusers 0.42.1 declares
#   requires_python >=3.11.0. The API returns only a docker buildx exit status and the real log is
#   console-only, so this was found by reading the base image's own config blob out of the registry rather
#   than from any log. Hence the version assert below: the next wrong base costs seconds, not nine minutes.
#
# THE PINS STAY. A tag or a branch can be re-pointed upstream, and a blind comparison whose contestant
# changed between the owner's first sitting and her second proves nothing. The revision below is an
# immutable commit, it is echoed on every response, and it is the only thing the handler will load.

# Ubuntu 24.04 base, which carries Python 3.12. torch stays on 2.6.0 regardless of this image's CUDA
# version: torch wheels bundle their own CUDA runtime, so a cu124 wheel runs fine here, and moving torch
# would drag torchao along with it for no gain.
FROM runpod/base:2.1.0-cuda1281-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
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

# FIRST STEP ON PURPOSE. This is the exact thing that killed build 4, and asserting it here turns a nine
# minute mystery into a ten second message that names the requirement.
RUN python3 -c "import sys; print('python', sys.version); assert sys.version_info >= (3, 11), 'diffusers 0.42 requires Python 3.11 or newer; this base has ' + sys.version"

# torch first, alone. It is the largest part of the install by a wide margin, and keeping it in its own
# layer means editing any other pin does not re-fetch it.
RUN python3 -m pip install --no-cache-dir --prefer-binary \
        torch==2.6.0 torchvision==0.21.0

# --prefer-binary rather than --only-binary=:all:. The strict form fails the entire build if any transitive
# dependency ships only an sdist, and with the build log console-only there would be no way to see which.
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Fails the build here, cheaply, if the set that was just installed does not actually import. Cheaper than
# discovering it from a worker that starts and then fails every job, which is what build 2 did.
RUN python3 -c "import torch, diffusers, transformers; from diffusers import DiffusionPipeline; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)"

# NOTE: there is deliberately NO weights fetch here. That is what made build 1 fail, and putting it back
# would reintroduce a 57.7 GB download inside a 1800 second budget.
COPY weights.py /app/weights.py
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
