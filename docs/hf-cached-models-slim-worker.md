# Model-free MiniMax H3 worker

This refactor preserves the validated `multimodal-4` application runtime while moving its
eight immutable model payloads out of the Docker image. It does not modify or replace the
`multimodal-4` image.

## Pinned releases

Application rollback identity:

- image: `multimodal-4`
- digest: `sha256:a8f93afb6759c91bdb99ee0474aa34fb2fcf26135d5a21e1d5a44ff6cc62fb2f`
- source: `1f9035fb0c397509e3b20ed63b5336507ad98114`
- build: `33289814525-1`
- ComfyUI: `dec5d9450a5290bcf63430409ea41018e67f41c3`

Model identity:

- repository: `CDitfort/privora-minimax-h3-models`
- revision: `ecb69a4211d74b5798398021003bccde02d63757`
- manifest: `multimodal-4-hf-cache-v1`
- payload: `69,309,544,079` bytes

The worker never resolves `refs/main`, scans for a convenient snapshot, or downloads a
missing model. It derives only this directory:

`/runpod-volume/huggingface-cache/hub/models--CDitfort--privora-minimax-h3-models/snapshots/ecb69a4211d74b5798398021003bccde02d63757/`

## Runtime configuration

Required values (the Dockerfile also supplies these pinned defaults):

```text
H3_MODEL_REPO=CDitfort/privora-minimax-h3-models
H3_MODEL_REVISION=ecb69a4211d74b5798398021003bccde02d63757
H3_MODEL_MANIFEST_VERSION=multimodal-4-hf-cache-v1
```

Operator-only path override for tests or a future RunPod cache-layout change:

```text
H3_MODEL_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
```

The image forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_HUB_DISABLE_TELEMETRY=1`. No client field can supply a repository, revision, cache
path, download URL, or filesystem path.

## Exact symlink map

At startup `bootstrap.py` validates all eight source files and exact sizes before it
creates any link. Existing regular files or links to a different target make startup
fail; exact existing links are accepted for warm-worker idempotency.

| ComfyUI relative destination | Pinned snapshot relative source |
| --- | --- |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | `vae/minimax_h3_video_vae_fp16.safetensors` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | `vae/minimax_h3_audio_vae_fp32.safetensors` |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` |

The legacy Pixaroma alias
`diffusion_models/h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors` points to the first
ComfyUI destination. It is an alias, not a ninth payload.

## Private GHCR publication

`.github/workflows/build-slim-ghcr.yml` is manual-only. Its definition lives on `main` so
GitHub exposes `workflow_dispatch`, but it checks out and verifies an immutable commit from
`hf-cached-models-slim`; it never builds the legacy `main` Dockerfile. The workflow:

1. runs `python -m unittest discover -s tests -p 'test_*.py'`;
2. runs all six existing Node test files with `node --test`;
3. creates a small Docker context through `scripts/stage_slim_image.py`;
4. rejects every exact manifest weight name and any alternate large model-like file;
5. builds `Dockerfile.slim` remotely for `linux/amd64`;
6. pushes an immutable canary tag and full source-commit tag to
   `ghcr.io/chrisditfort/privora-h3-runpod-worker`;
7. verifies the package remains private and reads OCI metadata without pulling layers;
8. uploads only `slim-image-release.json`, never image or model bytes, as an artifact.

Authentication is the source repository's ephemeral `GITHUB_TOKEN` with
`packages: write`. The package namespace matches the source repository owner, avoiding a
cross-account credential. Tags `latest`, `multimodal-4`, `code`, and `staging-*` are
reserved and refused.

## Later RunPod test endpoint (do not apply during this phase)

Create a **new** endpoint, leave the existing endpoint and `multimodal-4` untouched, then:

1. Set its container image to the verified private GHCR image by immutable digest, not a
   mutable tag.
2. Set RunPod's Model field to `CDitfort/privora-minimax-h3-models`.
3. Set `H3_MODEL_REPO`, `H3_MODEL_REVISION`, and `H3_MODEL_MANIFEST_VERSION` to the exact
   values above.
4. Set `H3_IMAGE_DIGEST` to the verified OCI digest. The source commit, repository, tag,
   and build ID are already embedded in the image's trusted build-identity file.
5. Start with zero production traffic and validate cache activation before spending GPU
   time on an inference canary.

The official latent upscaler is intentionally absent until source, license and checksum
are independently validated. User LoRAs remain a separate future R2 flow: Safetensors,
server-issued opaque IDs/keys, SHA-256 verification, family gating, per-job cleanup, and
no arbitrary client paths or URLs.
