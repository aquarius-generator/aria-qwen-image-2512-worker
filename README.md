# aria-qwen-image-2512-worker

A RunPod Serverless worker running **Qwen-Image-2512** (Apache-2.0) at **FP8**, for ARIA's `model-lab`
blind bake-off.

## Why this repository exists

There is no publicly pullable container image for a Qwen-Image-2512 FP8 serverless worker. Measured against
the real registries on 2026-11-09, six GHCR references and two Docker Hub spellings all answered `401`, and
GHCR returns `401` for a private repository and a missing one alike — so the only sound conclusion is *not
publicly pullable*.

RunPod can build a worker from a GitHub repository instead of pulling an image, but only from a repository
**connected to the RunPod account**: the deploy screen is populated with repos the account owns, forked, or
owns through an organisation. A third-party repo is never in that list. So the fix is to own one, which is
what this is.

**The one image that *is* publicly pullable is a different model** — `ghcr.io/protech1/qwen-image-runpod-serverless`
runs Qwen-Image-2.1 at GGUF. Using it would have been quicker and would have made the bake-off a test of
something nobody asked about, so it is recorded and not used.

## Status — read this before trusting it

**This worker has never been built and never been run.** It was authored on a machine with no Docker and no
RunPod credential, so nothing here is verified beyond the two things that could be checked:

- the weights reference resolves on Hugging Face (`Qwen/Qwen-Image-2512`, Apache-2.0, revision `25468b98…`);
- the handler's input and output field names match exactly what `model-lab/qwen-image.js` sends and parses.

The first build will need the usual shakedown. Saying so here is cheaper than discovering it from a bake-off
that silently produced nothing.

## Pinning

The weights are pinned to an immutable commit, not to `main`:

```
Qwen/Qwen-Image-2512 @ 25468b98e3276ca6700de15c6628e51b7de54a26
```

A tag or a branch can be re-pointed upstream. A blind comparison whose contestant changed between the
owner's first sitting and her second proves nothing, so the revision is fixed at build time, baked into the
image, and reported back on every response as `model_revision`.

## Deploying

1. In the RunPod console: **Settings → GitHub Integration**, authorise, and grant access to this repository.
2. **Serverless → New Endpoint → Import Git Repository**, choose `aria-qwen-image-2512-worker`.
3. Cost guard, matching the rest of the lab: **min workers 0, max workers 1, idle timeout 5s**, a GPU with
   **24GB or more** (RTX 4090 class), container disk **60GB**.
4. Put the resulting endpoint id into the ARIA service's Railway variables as `MODEL_LAB_IMAGE_ENDPOINT`.

ARIA reads that variable ahead of its own state file, so a console-created endpoint needs nothing rebuilt on
its side — the next deploy picks it up and fills the ten image comparisons by itself.

## Contract

**Input** (`{"input": { … }}`)

| field | default | notes |
|---|---|---|
| `prompt` | required | |
| `negative_prompt` | `""` | |
| `width` / `height` | 1328 | rounded to a multiple of 16 |
| `steps` | 30 | |
| `true_cfg_scale` | 4.0 | `cfg` accepted as an alias |
| `seed` | 12 | fixed by default so a re-run reproduces what was judged |
| `output_format` | `png` | `png` or `jpeg` |

**Output**

```json
{
  "images": [{ "data": "<base64>", "mime_type": "image/png" }],
  "seed": 12,
  "model": "Qwen/Qwen-Image-2512",
  "model_revision": "25468b98e3276ca6700de15c6628e51b7de54a26",
  "quantisation": "float8_dynamic",
  "latency_seconds": 12.3
}
```

Base64 bytes, never a URL. The caller stores the bytes permanently, and a signed link would expire long
before the comparison is judged.

## Licence

Worker code: MIT. Model weights: Apache-2.0 per the upstream model card — review it before production use.
