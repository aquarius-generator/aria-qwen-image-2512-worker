"""Pre-fetch the pinned Qwen-Image-2512 weights into the image at build time.

Why at build time rather than on first use: a cold start on a scale-to-zero endpoint is billed as
GPU-seconds, so downloading tens of gigabytes on the first request of every idle period is both slow and
expensive. Baking the weights in also fixes WHICH weights — a comparison whose model changed halfway
through is not a comparison.
"""
import os

from huggingface_hub import snapshot_download

MODEL_ID = os.environ["QWEN_MODEL_ID"]
REVISION = os.environ["QWEN_MODEL_REVISION"]

if __name__ == "__main__":
    print(f"[build] fetching {MODEL_ID} @ {REVISION}")
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,          # an immutable commit, never a branch or a tag
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"[build] weights at {path}")
