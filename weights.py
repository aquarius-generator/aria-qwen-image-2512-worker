"""Fetch the pinned Qwen-Image-2512 weights on FIRST START, into a cache later starts reuse.

This used to run at build time and that is what broke the build: 57.7 GB across 30 files does not download
inside RunPod's 1800 second build ceiling, and it would not have fitted on the 60 GB container disk beside
the image either.

TWO THINGS THIS FILE IS CAREFUL ABOUT:

1. WHERE THE CACHE LIVES IS DECIDED AT RUN TIME, NOT ASSUMED. A network volume mounts at /runpod-volume.
   If it is there and writable, the weights go there and survive every worker the endpoint ever starts. If
   it is not, they go to the container disk, which survives only as long as that one worker. Those two
   cases have very different cold-start costs, so the choice is returned to the caller and echoed on every
   response rather than being invisible.

2. A HALF-FINISHED DOWNLOAD MUST NOT LOOK LIKE A FINISHED ONE. `snapshot_download` is resumable and
   verifies by hash, so a worker killed mid-fetch resumes rather than serving a truncated model. The retry
   loop exists for the network, not for corruption.
"""
import os
import time

from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2512")
REVISION = os.environ.get("QWEN_MODEL_REVISION", "25468b98e3276ca6700de15c6628e51b7de54a26")
VOLUME_CACHE = os.environ.get("QWEN_VOLUME_CACHE", "/runpod-volume/hf")
FALLBACK_CACHE = os.environ.get("QWEN_FALLBACK_CACHE", "/app/hf-cache")


def choose_cache():
    """Return (path, kind). Prefers the network volume; falls back to the container disk."""
    mount = os.path.dirname(VOLUME_CACHE.rstrip("/")) or "/runpod-volume"
    if os.path.isdir(mount) and os.access(mount, os.W_OK):
        os.makedirs(VOLUME_CACHE, exist_ok=True)
        return VOLUME_CACHE, "network-volume"
    os.makedirs(FALLBACK_CACHE, exist_ok=True)
    return FALLBACK_CACHE, "container-disk"


def ensure_weights(attempts=3):
    """Download the pinned revision if it is not cached yet. Returns (local_path, cache_kind, seconds)."""
    cache, kind = choose_cache()
    started = time.time()
    last = None
    for attempt in range(attempts):
        try:
            path = snapshot_download(
                repo_id=MODEL_ID,
                revision=REVISION,              # an immutable commit, never a branch or a tag
                cache_dir=cache,
                ignore_patterns=["*.md", ".gitattributes"],
                max_workers=8,
            )
            return path, kind, round(time.time() - started, 1)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[weights] attempt {attempt + 1} of {attempts} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {MODEL_ID}@{REVISION} into {cache}: {last}")


if __name__ == "__main__":
    # Runnable by hand for a cache warm-up. Deliberately NOT invoked from the Dockerfile.
    p, k, s = ensure_weights()
    print(f"[weights] {p} ({k}) in {s}s")
