# MiniMax H3 FL2VA — RunPod Serverless (Blackwell / sm_120)

A RunPod Serverless worker that runs MiniMax H3 FL2VA **text-to-video** and **first-frame
image-to-video** with native audio on an **NVIDIA RTX PRO 6000 Blackwell Server Edition**,
with all ~42 GB of model weights baked into the image so workers never download weights at
startup.

Final image:

```
ghcr.io/chrisditfort/minimax-h3-blackwell-serverless:latest
```

## What's inside

Inherited unchanged from `ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-blackwell`:

| Component | Version |
|---|---|
| PyTorch | `2.10.0+cu130` |
| CUDA | 13 |
| Compute capability | `sm_120` (12, 0) |
| SageAttention | `sageattn3` 1.0.0 |
| ComfyUI | pinned at `dec5d945` (native MiniMax H3 support) |
| ComfyUI-Pixaroma | pinned at `433bbedc` |

Baked model weights (~42.5 GB total, appended as four separate layers):

| File | Path in image | Size |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | 20.97 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` | 15.69 GB |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | 5.21 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | 0.61 GB |

REF2VA is deliberately **not** included. FL2VA already provides first-frame image
conditioning — `MiniMaxH3ImageToVideo` takes optional `first_frame` / `last_frame` inputs —
so image-to-video needs no extra weights. Excluding REF2VA saves a further ~21 GB.

### Model paths, and why they are where they are

ComfyUI lives at **`/opt/comfyui-baked`**, not `/workspace/runpod-slim/ComfyUI`. The base
image only copies it into `/workspace` at *Pod* runtime, via the entrypoint this image
replaces. Running it in place avoids duplicating a ~550 MB tree and keeps the worker
immune to anything RunPod might mount over `/workspace`.

The FL2VA diffusion model is reachable under **two** names:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors` — the real file, matching the bare
  filename the existing Cloudflare workflow uses.
- `h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors` — a relative symlink to the same
  bytes, matching the bundled Pixaroma workflows.

ComfyUI walks model directories with `os.walk(followlinks=True)` and `get_full_path()`
accepts symlinks, so both spellings resolve and nothing is stored twice.

### Startup behaviour

The base image's `ENTRYPOINT` (`/entrypoint.sh`) is a Pod stack: driver preflight →
download ~60 GB of weights → SageAttention bootstrap → `/start.sh`, which starts SSH,
JupyterLab, FileBrowser and a ComfyUI bound to `0.0.0.0`.

This image **replaces `ENTRYPOINT` entirely** (overriding `CMD` alone would not work — the
base sets `ENTRYPOINT`, so a `CMD` becomes its arguments). The handler is PID 1, starts one
private ComfyUI on `127.0.0.1:8188`, and exposes no ports. No model download, SSH, Jupyter
or FileBrowser ever runs.

Cold start is therefore: pull cached image → start handler → start ComfyUI → load baked
models → serve. The first uncached pull is large; that is expected and only happens once
per host.

## Build

The image is built in two stages by `.github/workflows/build.yml`.

1. **`build-code`** builds the slim runtime image from the `Dockerfile` and pushes
   `ghcr.io/chrisditfort/minimax-h3-blackwell-serverless:code`.
2. **`append-models`** streams each model from Hugging Face into a tar layer, verifies it,
   appends it to the remote image with `crane`, deletes it, and moves to the next. The last
   append is tagged `:latest`.

Model weights are never downloaded inside a BuildKit build. Each model is streamed straight
from Hugging Face into its layer tarball — because the exact byte size is known up front,
the tar header is written before the bytes arrive, so peak runner disk is ~1× the largest
model (21 GB) instead of 2×. Every layer is verified against the exact size **and** SHA-256
that the base image's own downloader uses before it is appended, and any mismatch fails the
build loudly.

### Trigger a build

```bash
# From the GitHub UI: Actions -> "Build Blackwell Serverless Image" -> Run workflow
gh workflow run build.yml

# Code image only, skipping the ~42 GB model append (fast, for handler changes):
gh workflow run build.yml -f skip_models=true

gh run watch
```

Pushes to `main` build automatically. A full build takes roughly 45–75 minutes, dominated
by the 42 GB transfer.

## Deploy on RunPod

Create a Serverless endpoint with:

| Setting | Value |
|---|---|
| Container image | `ghcr.io/chrisditfort/minimax-h3-blackwell-serverless:latest` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU count | 1 |
| Min workers | 0 |
| Max workers | 1 (initially) |
| Scaler | Queue Delay |
| Queue delay | 1 second (for testing) |
| Idle timeout | 60 seconds |
| Execution timeout | 600 seconds or higher |
| Container disk | ≥ 60 GB |
| Ports | none |
| Start command | *blank — use the image `CMD`* |

The image is public on GHCR; no registry credentials are needed.

### Verifying Blackwell on first boot

The worker logs its GPU stack before accepting jobs. Expect:

```
[handler] torch.__version__      = 2.10.0+cu130
[handler] torch.version.cuda     = 13.0
[handler] torch.cuda.get_device_name(0)       = NVIDIA RTX PRO 6000 Blackwell Server Edition
[handler] torch.cuda.get_device_capability(0) = (12, 0)
[handler] COMFY_SAGE_ATTENTION3 = 1
[handler] sageattn3 present, version 1.0.0
[handler] SageAttention3 enabled: GPU capability (12, 0) matches the wheel (12.0).
```

and from ComfyUI itself:

```
Using SageAttention3
```

A capability below `(12, 0)` or a non-CUDA-13 torch is logged as a `WARNING` — it means the
wrong GPU was scheduled, not that the image is wrong.

### If a non-Blackwell GPU gets scheduled

The `sageattn3` wheel in the base image is compiled for one compute capability (`sm_120`).
It *imports* fine on any GPU — the mismatch only surfaces at generation time, as
`no kernel image is available for execution on the device` on **every** attention call,
which floods the log and silently falls back to PyTorch attention for the whole run.

So the worker checks the capability it was actually scheduled and turns SageAttention3 off
up front when it does not match, letting ComfyUI choose its own backend cleanly:

```
[handler] WARNING: SageAttention3 was built for compute capability 12.0 but this worker
          was scheduled a 9.0 GPU. Disabling it - ...
[handler]   This is a GPU scheduling problem: pin the RunPod endpoint to an
          RTX PRO 6000 Blackwell to get SageAttention3. Generation still works without it.
```

Seeing this means the endpoint is not pinned to Blackwell. Generation still succeeds, just
slower. Fix it in the endpoint's GPU selection rather than in the image.

## Measuring cold-start performance

Every job ends with one structured line, so a run can be compared against another without
reading the surrounding ComfyUI chatter:

```
[perf] proc=657169cad581 cold_process=true job_in_proc=1 proc_age=0.0s comfy_boot=8.4s        comfy_wait=0.1s submit=0.0s pre_sampling=13.2s first_step=5.1s sampling=47.1s        steps=20/20 decode=8.8s output=1.3s total=71.4s per_step=2.36s status=ok
[perf] nodes KSampler=47.1s MiniMaxH3VideoVAEDecode=7.2s MiniMaxH3TextEncode=3.9s
```

| Field | Meaning |
|---|---|
| `proc` | Identifies the worker **process**. The same value on two jobs means the process was reused; a new value means that job paid a full cold start. On a scale-to-zero endpoint this is the only way to tell whether FlashBoot actually restored a worker |
| `cold_process` | `true` on the first job a process serves |
| `job_in_proc` | How many jobs this process has served, including this one |
| `proc_age` | Seconds between process start and this job starting |
| `comfy_boot` | How long ComfyUI itself took to answer HTTP |
| `comfy_wait` | How much of that this job actually waited for. `~0.0s` means ComfyUI was already up and its boot was **not** on this job's clock |
| `submit` | Time to POST the prompt to ComfyUI |
| `pre_sampling` | Prompt queued → first sampler frame: model staging, text encode, graph setup. **This is the cold-start cost the optimisation work targets** |
| `first_step` | First sampler frame → first completed step. Carries the model-initialisation stall, so it is not mistaken for per-step cost |
| `sampling` | First → last sampler frame (includes `first_step`) |
| `steps` | Steps observed / total |
| `per_step` | `sampling` ÷ steps observed |
| `decode` | Last sampler frame → workflow finished: VAE decode, audio, mux, save |
| `output` | Collecting, encoding and delivering artefacts |
| `total` | Whole handler invocation |
| `status` | `ok`, `bad_request`, `no_output`, `workflow_error` or `exception`. Emitted for failures too — a job that times out is exactly the one whose breakdown you want |

A phase whose boundary never arrived reports `n/a` rather than failing the job.

### FlashBoot preload (opt-in, off by default)

`pre_sampling` is ~16s on a cold Blackwell worker, and the `[perf] nodes` line shows where
it goes: the loaders finish in ~0.6s, then `MiniMaxH3ImageToVideo` spends ~7s staging the
text encoder and `SamplerCustomAdvanced` ~9s staging the DiT. **Executing a loader does not
put weights on the GPU** - `model_management.load_models_gpu()` does, and only when a node
actually uses the model. So a loader-only warmup would save nothing.

`H3_FLASHBOOT_PRELOAD=1` therefore runs the smallest graph that reaches both of those
calls, after ComfyUI is ready and before `runpod.serverless.start()`:

```
UNETLoader ─┬─────────────────────────► BasicGuider ─┐
CLIPLoader ─┼─► MiniMaxH3ImageToVideo ───────────────┼─► SamplerCustomAdvanced ─► PreviewAny
VAELoader  ─┘   (32x32, 5 frames)      BasicScheduler┘   (1 step)
VAELoader (audio) ───────────────────────────────────────────────────────────────► PreviewAny
```

No `VAEDecode`, no `CreateVideo`, no `SaveVideo` - nothing is decoded and nothing is written
to disk. `PreviewAny` is there because ComfyUI **rejects a prompt with no `OUTPUT_NODE`**
("Prompt has no outputs"), so a graph of loaders alone would never execute at all.

Why the real job then gets it for free: ComfyUI keys its output cache on `class_type` +
inputs and deliberately excludes node id, so the real workflow's loaders - whose inputs are
byte-identical to the preload's - hit the cache and receive **the same `ModelPatcher`
objects**. `load_models_gpu()` finds those already in `current_loaded_models`
(`LoadedModel.__eq__` compares by identity) and skips staging. Models survive the preload
prompt because `unload_all_models()` only runs on OOM or under `--disable-smart-memory`.

**Not preloaded:** VRAM staging for the two VAEs. There is no way to stage a VAE without
running a decode, and decode is the expensive part. Their *loaders* run, so those cache
entries are warm, but the ~10.8s decode phase is untouched.

If the preload fails or times out, it logs and the worker starts normally - the first
request just loads models the usual way. On timeout ComfyUI is sent `/interrupt` so the
first real job is not queued behind an abandoned synthetic prompt.

#### Measuring it

```
# A: control
H3_FLASHBOOT_PRELOAD=0

# B: preload
H3_FLASHBOOT_PRELOAD=1
```

Startup gains a line:

```
[perf] preload proc=… status=ok total=15.9s loaded=te,dit,video_vae,… preload_te=7.1s preload_dit=8.4s
[perf] startup proc=… to_serverless_ready=24.8s comfy_boot=7.1s preload_enabled=true preload_total=15.9s
```

Compare the **first job's `pre_sampling`** against the ~16.0s control. `sampling` and
`per_step` must stay ~47.0s / ~2.35s; if they move, something other than residency changed.

The blunt check for whether it worked: ComfyUI logs `Requested to load <name>` **only when a
model is not already resident**. After a successful preload the first real job should show
*fewer* `Requested to load MiniMaxH3TEModel_` / `MiniMaxH3` lines than the control.

#### The honest caveat

Within one process this mechanism is sound and verified in ComfyUI's source. Whether
**FlashBoot** preserves it across a scale-to-zero is the open question: it snapshots the
worker process, but CUDA context and VRAM residency are not ordinary process memory, and a
restore may well come back with an empty GPU. If it does not survive, the preload merely
moves ~16s from the first request into startup - both are billed, so that is a wash, not a
win. Judge it on `pre_sampling`, not on the fact that the preload ran.

### A/B testing `--highvram`

The 96 GB Blackwell parts hold the entire ~41 GB working set with room to spare, but ComfyUI
still selects `NORMAL_VRAM` and streams weights through pinned host RAM. Whether pinning
them in VRAM helps is measurable without rebuilding: set `COMFY_EXTRA_ARGS` on the endpoint
and compare runs of the **same image**.

```
# baseline
COMFY_EXTRA_ARGS=            # or unset

# candidate
COMFY_EXTRA_ARGS=--highvram
```

Confirm the flag actually landed — startup logs it explicitly:

```
[handler] ComfyUI effective args: COMFY_EXTRA_ARGS='--highvram' -> extra=['--highvram']
[INFO] Set vram state to: HIGH_VRAM
```

Then compare **`pre_sampling`** and **`first_step`** between the two, on jobs where
`cold_process=true` in both. `sampling` and `per_step` should be unchanged; if they move,
something other than residency changed and the comparison is not clean.

## Experimental two-GPU execution

This branch adds an opt-in path that splits **one** generation across **two** GPUs by
sequence-parallelising the H3 transformer (Ulysses: shard the packed token sequence, switch
to head-sharding inside attention with an all-to-all, switch back) and distributing the
video VAE's temporal chunks. It is off by default — `H3_GPU_MODE=single` applies no patches
to ComfyUI at all, so the image behaves exactly like its predecessor until it is asked not
to.

```
H3_GPU_MODE=dual        # plus 2 GPUs per worker on the endpoint
```

A worker asked for `dual` that cannot deliver it **refuses to start** rather than quietly
serving single-GPU results from an endpoint that believes it is measuring two. Every rank
verifies sharded attention against unsharded attention on the real GPUs before it serves.

Full design, log formats, failure modes and what is deliberately still serial:
[`docs/dual-gpu-experiment.md`](docs/dual-gpu-experiment.md). Build it with the
**"Build Experimental Dual-GPU Image"** workflow, which publishes to a new immutable tag
and never touches `:latest`, `:code` or `staging-*`.

## Cloudflare Worker API (worker.js)

`worker.js` is the public API in front of both RunPod endpoints. It owns the contract and
builds the ComfyUI graph; the RunPod handler stays a thin executor that runs whatever
workflow it is handed. Callers never see ComfyUI node ids.

Run its tests with `node --test worker.test.js` (no dependencies).

### Normalized request

```json
{
  "backend": "h3-blackwell",
  "mode": "text_to_video",
  "prompt": "A cinematic ocean scene",
  "quality": "standard",
  "duration": 5,
  "aspect_ratio": "16:9",
  "seed": 51
}
```

The raw shape (`width`/`height`/`frames`/`steps`/`seed`) still works unchanged, and
`/status/...` and `/cancel/...` are untouched.

### Quality tiers

Dimensions are not invented here. They come from the pinned ComfyUI build's own
`adapt_canvas()` in `comfy_extras/nodes_minimax_h3.py`: short edge to the tier's value,
total area capped at 768x1344, each axis rounded to 32.

| Quality | Short edge | Steps | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 |
|---|---|---|---|---|---|---|---|
| `fast` | 576 | 14 | 1024x576 | 576x1024 | 576x576 | 768x576 | 576x768 |
| `standard` | 576 | 20 | 1024x576 | 576x1024 | 576x576 | 768x576 | 576x768 |
| `hd` | 768 | 20 | 1344x768 | 768x1344 | 768x768 | 1024x768 | 768x1024 |

`hd` is the canvas the H3 nodes themselves default to - a validated 768-class
configuration, not an upscale or a second-stage pipeline.

`GET /capabilities` returns this table live, so clients need not hardcode it.

### Duration

H3 runs at 24 fps and only accepts frame counts where `frames % 17 == 5`
(`align_frame_count()` in the same file). `duration` is converted in one helper:

```
duration: 5  ->  round(5 * 24) = 120  ->  snapped up to 124 frames (5.1667s)
```

Supplying an off-grid `frames` value is a 400 that names the nearest legal counts.

### Precedence

Explicit raw values beat the quality preset. A raw value that *contradicts* a normalized
one is a 400 rather than a silent guess:

| Combination | Result |
|---|---|
| `quality` + `steps` | `steps` wins |
| `quality`/`aspect_ratio` + `width`/`height` | `width`/`height` win |
| `aspect_ratio` + `width`/`height` that disagree | **400** |
| `duration` + `frames` that disagree | **400** |
| `width` without `height` | **400** |

### Structured prompt fields

`camera`, `shot`, `lighting`, `style`, `motion` and `audio_prompt` are appended as labelled
lines after the user's prompt, which is never rewritten. Absent fields add nothing.

```
A woman standing near the ocean.

Camera: slow dolly forward.
Lighting: warm sunset.
```

### Modes

| Mode | Status |
|---|---|
| `text_to_video` | available |
| `first_frame_to_video` | available (`first_frame.url`, https) |
| `last_frame_to_video` | **501** - handler stages only one image |
| `first_last_frame_to_video` | **501** - handler stages only one image |
| `reference` (Ref2VA) | **501** - model not in the image |
| `regenerate_2k` | **501** - no second-stage model |

An unavailable mode returns 501 with a reason. It never silently degrades to
text-to-video, which would return a video that quietly ignored the caller's keyframes.

### Ref2VA status

`MiniMaxH3ReferenceToVideo` **is present** in the pinned ComfyUI build, so no ComfyUI
change is needed. What is missing is the weight file:

| Item | Value |
|---|---|
| File | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Size | **20,970,379,616 bytes (20.97 GB)** |
| Image impact | 43.2 GB -> ~64 GB compressed, roughly +50% |
| Cold start | first pull on an uncached worker grows proportionally |
| VRAM | ~20 GB more resident if held alongside FL2VA; 96 GB is sufficient |

The node also needs an `audio_vae` input that FL2VA does not use, plus Autogrow inputs
(`ref_image_0..8`, `ref_video_0..2`, `ref_video_audio_0..2`, `ref_audio_0..2`), and the
prompt must reference `<Picture i>` / `<Video k>` / `<Audio j>` tags. The API shape is in
place; the weights are a deliberate, separately-costed decision.

### 2K status

Nothing installed and nothing measured. `regenerate_2k` exists only as a reserved mode
name so the routing layer does not need reshaping later.

## Confidential Generation

Optional per-request privacy mode using **hybrid encryption**. The inference worker receives
an encryption-only public key, generates a fresh random file key per video, encrypts with
AES-256-GCM, and wraps that file key to the public key. Persistent storage receives
ciphertext only; the browser unwraps and decrypts for playback.

The property this buys:

> A complete copy of the RunPod job input, the RunPod job output, Cloudflare application
> state and R2 storage is **not enough** to decrypt an artefact without the user's
> passphrase or an unlocked private key.

Standard generation is unchanged. A request with no `privacyMode` behaves exactly as before.

Full design, trust boundaries and threat model: **[docs/confidential-generation.md](docs/confidential-generation.md)**.
Product wording: **[docs/privacy-statement.md](docs/privacy-statement.md)**.

### The five keys

Named explicitly throughout the code, because conflating any two is how these systems go
wrong.

| Name | What | Lives where |
|---|---|---|
| `passphrase` | typed by the user | browser only |
| `keyEncryptionKey` | 256-bit, derived from the passphrase | browser only |
| `privateKey` | RSA-3072, unwraps file keys | browser; stored **encrypted** |
| `publicKey` | RSA-3072, wraps file keys | sent with the request; not secret |
| `fileEncryptionKey` | 256-bit random, one per artefact | inference worker; persisted **wrapped only** |

### Request

```json
{
  "backend": "h3-blackwell",
  "prompt": "A cinematic ocean scene",
  "quality": "standard",
  "duration": 5,

  "privacyMode": "confidential",
  "encryption": {
    "version": 2,
    "algorithm": "AES-256-GCM",
    "keyWrapAlgorithm": "RSA-OAEP-256",
    "publicKey": "<base64url SPKI, RSA >= 3072 bits>",
    "keyId": "<optional; must match the key's own fingerprint>"
  }
}
```

`keyId` is derived from the public key (truncated SHA-256 of the SPKI), so it can never name
a key it does not belong to. Supplying a disagreeing one is a 400.

**Rejected outright**, at any depth and regardless of case or separators:
`passphrase`, `password`, `privateKey`, `privateEncryptionKey`, `encryptedPrivateKey`,
`keyEncryptionKey`, `kek`, `fileEncryptionKey`, `fek`, `decryptionKey`, `derivedKey`,
`aesKey`, `symmetricKey`, `secretKey`. Nothing downstream reads them — the point is that a
frontend which starts sending one should fail loudly rather than silently reintroduce a
weakness.

### Status

```jsonc
{
  "id": "…", "status": "COMPLETED", "privacyMode": "confidential",
  "artifact": {
    "url": "/jobs/…/artifact",
    "encrypted": true,
    "cryptoVersion": 2,
    "videoAlgorithm": "AES-256-GCM",
    "keyWrapAlgorithm": "RSA-OAEP-256",
    "keyId": "…",
    "contentType": "application/octet-stream",
    "originalContentType": "video/mp4"
  },
  "video": { "url": "/jobs/…/video", "…": "unchanged, for existing clients" }
}
```

Never returned: a passphrase, a KEK, a private key, or a plaintext file key. The wrapped file
key is not in the status response either — it lives inside the container's authenticated
header, where it cannot be swapped.

### Browser

`client/confidential-generation.js` is a zero-dependency ES module; `client/demo.html` is a
minimal working UI.

```js
import { createKeyBundle, unlockKeyBundle, generate, fetchAndDecrypt, attachToVideo }
  from "./client/confidential-generation.js";

// Once per user. The private key is encrypted with a passphrase-derived key.
const bundle = await createKeyBundle(passphrase);      // persist this; it holds no usable secret

// Per generation. Only the public half is sent.
const job = await generate(API, { prompt, quality: "standard" }, { bundle });

// Playback.
const unlocked = await unlockKeyBundle(bundle, passphrase);
const { blob } = await fetchAndDecrypt(API, job.jobId, unlocked);
attachToVideo(document.querySelector("video"), blob);
```

### Crypto versions

| | v1 (symmetric) | v2 (hybrid) |
|---|---|---|
| Request carries | a 256-bit AES key | an RSA public key |
| File key origin | derived from the passphrase | random, generated in the worker |
| Readable | **yes** | yes |
| Creatable | **no** — 400 naming the replacement | yes |

Existing v1 artefacts stay decryptable and no data migration is needed. New v1 jobs are
refused: a compatibility path would preserve the exact property v2 removes, so there is no
flag to re-enable it.

### Fail closed

| Situation | Outcome |
|---|---|
| A symmetric `encryption.key`, or `version: 1` | 400 naming the v2 replacement |
| Any field that could decrypt the result | 400, at any depth |
| Public key not SPKI, wrong type, or under 3072 bits | 400 |
| `keyId` disagreeing with the public key | 400 |
| Unsupported algorithm / key-wrap algorithm / version | 400 |
| `standard` **with** an `encryption` block | 400 |
| No `JOB_TOKEN_SECRET` | 503; the job is never submitted |
| FEK generation, wrapping, encryption or upload fails | job fails; plaintext deleted; nothing stored |
| A plaintext MP4 uploaded for a confidential job | **422; nothing written to R2** |
| A **v1** container uploaded for a **v2** job | **422; nothing written to R2** |
| A v2 container with no valid `kw` block | **422; nothing written to R2** |
| Container header naming a different generation | **422; nothing written to R2** |

Enforced without trusting the uploader: the job's privacy mode **and crypto version** are
signed into its output token, so the upload endpoint reads both from an HMAC rather than from
the request or a storage lookup.

### Cost

The asymmetric step applies only to a 32-byte key, so it is invisible next to inference.

| Operation | Median |
|---|---|
| RSA-OAEP-3072 wrap (worker) | 0.05 ms |
| Encrypt a 2 MB / 5 s clip, end to end | 13.6 ms |
| RSA-3072 keygen (browser, once per user) | 129 ms |
| Unlock the private key (PBKDF2-dominated) | 236 ms |
| Unwrap the file key (browser) | 1.65 ms |
| Decrypt 2 MB (browser) | 2 ms |

### Recovery

There is none. If the passphrase is lost and no bundle was exported, the encrypted private
key cannot be unlocked, the wrapped file key cannot be unwrapped, and the video is
unrecoverable. No master key, no admin override, no escrow. That is the property, not a
defect.

### This is not zero-knowledge

The model needs the plaintext prompt and produces plaintext frames; both exist inside the
inference container while the job runs, as does the file key for the moment it is in use.
See [docs/privacy-statement.md](docs/privacy-statement.md) for the terminology to use and
the terminology to avoid.

## Realtime progress and R2 output

The browser never talks to RunPod. Cloudflare owns the public API, the storage and the
realtime channel; RunPod owns inference and nothing else.

```
Browser ──WebSocket──> Worker ──> JobChannel (Durable Object)
   │                     │                ▲
   │  POST /generate     │ R2 binding     │ HTTP progress + output
   ▼                     ▼                │
Worker ──job payload──> RunPod handler ───┘
                            │
                            └──> ComfyUI (internal websocket)
```

### Why RunPod holds no Cloudflare credentials

Each job is issued three short-lived HMAC tokens binding **job id + purpose + expiry**:
`progress`, `output-upload` and `asset-download`. A progress token cannot upload a video;
an output token for one job cannot write to another; all of them expire. The signing key
(`JOB_TOKEN_SECRET`) never leaves Cloudflare, and R2 is reached only through the Worker's
`H3_OUTPUTS` binding - there is no R2 API key anywhere in the image.

### Internal routes (RunPod -> Cloudflare)

| Route | Purpose token | Notes |
|---|---|---|
| `POST /internal/jobs/:id/progress` | `progress` | Normalised phase events; raw ComfyUI text is never forwarded |
| `PUT /internal/jobs/:id/output` | `output-upload` | Raw `video/mp4` body streamed straight into R2 |
| `GET /internal/jobs/:id/assets/:assetId` | `asset-download` | Keyframe fetch |

### Public routes

| Route | Notes |
|---|---|
| `POST /jobs/:id/assets` | Upload a keyframe. PNG/JPEG/WebP, 32 MB cap |
| `GET /jobs/:id/artifact` | Streams from R2. Range supported, so browsers can seek |
| `GET /jobs/:id/video` | Alias of the above, kept for existing callers |
| `DELETE /jobs/:id/artifact` | Removes the artefact. Alias: `/video` |
| `DELETE /jobs/:id` | Removes the artefact **and** that generation's input keyframes |
| `GET /ws/jobs/:id` | WebSocket. Current state on connect, then live events |

`artifact` is the mode-neutral name: a confidential artefact is not a playable video until
the browser has decrypted it, and the API should not say otherwise. `/video` still works
and still serves the right bytes in both modes.

### WebSocket protocol

```jsonc
{"type": "state",    "jobId": "...", "phase": "sampling", "step": 5, "steps": 20}
{"type": "progress", "jobId": "...", "phase": "sampling", "step": 7, "steps": 20, "percent": 35}
{"type": "progress", "jobId": "...", "phase": "decoding", "percent": 90}
{"type": "completed","jobId": "...", "video": {"url": "/jobs/.../video"}}
{"type": "failed",   "jobId": "...", "error": {"code": "...", "message": "..."}}
{"type": "cancelled","jobId": "..."}
```

Phases: `queued`, `starting_worker`, `comfy_ready`, `loading_models`, `sampling`,
`decoding`, `uploading`, `completed`, `failed`, `cancelled`. Step counters appear only
during `sampling`.

### R2 object naming

```
inputs/{jobId}/{assetId}.{png|jpg|webp}
outputs/{jobId}/video.mp4        standard
outputs/{jobId}/artifact.enc     confidential (ciphertext)
```

Two names, one prefix. A confidential artefact is not an MP4 and is not called one: anyone
looking at the bucket during an incident should be able to tell ciphertext from video at a
glance, and nothing should be able to hand `artifact.enc` to a `<video>` tag by mistake.
The prefix stays `outputs/` because the lifecycle rules below are prefix-scoped - moving it
would silently drop encrypted artefacts out of retention. Neither name contains the prompt.

Keys are chosen by the Worker, never by RunPod: the handler uploads to a route that
already knows where the object belongs, so a compromised or buggy worker cannot write
anywhere else in the bucket. A client-supplied `r2_key` is reduced to its asset id for the
same reason. There is no user namespace because this system has no user identity;
`outputKey()` and `inputKey()` are the only two places that would change if one is added.

**Lifecycle:** nothing is deleted automatically. Outputs are small (a 5s clip is ~1.5-3 MB)
so this is not urgent, but an R2 lifecycle rule expiring `inputs/` after a few days and
`outputs/` after whatever the product promises is the obvious next step. Do not add
aggressive cleanup that could race a job still writing.

### Deleting an output

```
DELETE /jobs/{jobId}/artifact     the artefact only (alias: /video)
DELETE /jobs/{jobId}              the artefact plus that generation's input keyframes
```

```json
{ "id": "08ad4ace-d847-46b4-a1d6-a919d7e3a0c9", "deleted": true, "scope": "output", "removed": 1 }
```

`removed` counts objects that actually existed. Both artefact names are always named for
deletion whether or not the listing mentioned them, so a listing that is momentarily behind
cannot leave ciphertext in the bucket - but naming a key is not evidence an object was
there, so those do not inflate the count.

**One code path for both privacy modes.** A confidential artefact is deleted by exactly the
same function as a standard one, because "we forgot to handle deletion for the encrypted
kind" is precisely the bug a second code path invites.

Keys are derived from the job id inside the Worker, so a caller cannot name an arbitrary
object - and there is deliberately **no** generic "delete this key" route anywhere in the
API. A malformed job id is a 400 before R2 is touched at all.

Idempotent: R2's delete does not fail on a missing key and the job is marked deleted either
way, so a retry after a dropped connection returns the same answer rather than an error.

Afterwards `/status/...?jobId=...` reports the output as gone instead of handing back a URL
that would 404, while the generation itself stays `COMPLETED` - deleting an artefact is not
a retrospective failure of the job that produced it:

```jsonc
// before
"video": { "url": "/jobs/.../video", "key": "outputs/.../video.mp4", "size": 2196233, "deleted": false }
// after
"video": { "deleted": true }
```

`GET /jobs/{jobId}/video` then returns 404, including for Range requests, and the WebSocket
`completed` event carries `{"deleted": true}` with no URL.

**Access control is unchanged:** possession of the job id is the credential, exactly as it
already is for `/status/{jobId}`. This endpoint is shaped so that a future API Runtime layer
can call it internally *after* verifying user ownership - the ownership check belongs there,
not here.

### Retention

Nothing expires by default; the bucket only carries R2's stock multipart-abort rule.
`scripts/set_r2_lifecycle.sh` applies a prefix-scoped policy - dry-run by default, `--apply`
to write it:

| Prefix | Suggested | Why |
|---|---|---|
| `inputs/` | 7 days | Keyframes are only needed while the job runs, which is under two minutes |
| `outputs/` | 30 days | Long enough that a bookmarked link is unlikely to surprise anyone; ~6 GB for 100 clips/day |

These are suggestions, not a decision made for you: retention length is user-visible and
belongs to the product. Edit the two constants at the top of the script before applying.

Lifecycle expiry and `DELETE /jobs/:id/video` are deliberately separate - the endpoint is
for deliberate removal, the policy is for outputs nobody came back for. Note that an object
expired by lifecycle leaves the Durable Object still reporting a URL until that job's state
is next written; `GET .../video` correctly 404s, but the status response can lag. Shortening
retention below the DO's own lifetime would make that more visible.

### Behind Cloudflare Access

If the Worker is behind Access - it is, on `minimax-h3-backend` - RunPod needs an Access
**service token** as well as its job token. They are different credentials doing different
jobs: Access decides whether the request reaches the Worker at all, the job token decides
what it may do once there.

Set on the RunPod endpoint (any of these spellings; the first pair is Cloudflare's own):

```
CLOUDFLARE_ACCESS_CLIENT_ID / CLOUDFLARE_ACCESS_CLIENT_SECRET
CF_ACCESS_CLIENT_ID         / CF_ACCESS_CLIENT_SECRET
CLOUDFLARE_ACCESS_KEY_ID    / CLOUDFLARE_SECRET_ACCESS_KEY
```

The last pair is accepted because it is easy to reach for, but those names conventionally
mean R2's *S3-API* credentials, which are a different credential and will not pass Access.
An Access service token's client id ends in `.access`.

### Troubleshooting the callback path

The worker states its own view at boot, which is the fastest way to rule out the
environment:

```
[handler] Cloudflare Access service token: configured (client id 6f1a2b...c.access, ends in '.access')
[handler] Cloudflare Access service token: NOT configured. ...
```

Then the `[perf]` line at the end of each job:

| Field | Meaning |
|---|---|
| `progress_callbacks=21/21` | All callbacks delivered |
| `progress_callbacks=0/21` | Every callback rejected - look for the warning naming Access |
| `output_upload`, `output_bytes` | Present only when the R2 path ran |

Two traps worth knowing, both of which have bitten this code:

* **A 302 is not a failure to `requests`.** `Response.ok` is `status_code < 400`, so
  Access's login redirect reads as success, and `requests` follows redirects by default -
  re-issuing the PUT as a GET without the body. Both are now explicitly rejected, and the
  upload additionally requires the Worker's own JSON acknowledgement, because anything in
  front of the Worker can answer 200 with HTML while storing nothing.
* **Env vars are injected at container start.** A worker container created before the
  variables were saved keeps the old environment no matter what the endpoint config says.
  The same `workerId` across runs is the tell. Releasing a new endpoint version forces
  fresh workers.

* **A RunPod secret reference that never resolved.** Endpoint values may be written as
  `{{ RUNPOD_SECRET_<name> }}`. If no account-level secret of that name exists under
  RunPod Settings -> Secrets, RunPod passes the literal braces through rather than
  failing. The variable then looks set, the handler sends a template string as a
  credential, and Cloudflare Access rejects it - identical, from inside the container, to
  a real token being refused. This is what actually broke the first working deployment,
  and it survived several rounds of checking the Access policy, the paths and the
  variable names, all of which were correct. The handler now treats such a value as
  unconfigured and names it in the error.

**R2 is the ground truth.** A status response can only report what the handler believed;
the object either exists or it does not:

```
npx wrangler r2 object get minimax-h3-private-output/outputs/{jobId}/video.mp4 --remote --file out.mp4
```

## Request schema

```jsonc
{
  "input": {
    "workflow":      { /* ComfyUI API-format workflow — required */ },

    // All three below are optional. Omit them all for text-to-video.
    "image_url":     "https://example.com/frame.png",  // first-frame image (HTTPS only)
    "image_base64":  "<base64>",                       // alternative to image_url
    "image_node_id": "15"                              // only if auto-detection is ambiguous
  }
}
```

`image_url` and `image_base64` are mutually exclusive — supplying both is rejected. Supplying
neither runs the existing text-to-video path unchanged.

### How the image reaches the workflow

The handler picks the target node in this order, and **fails rather than guessing**:

1. `image_node_id`, if given — validated to exist and to accept an `image` input.
2. Otherwise, the image loader that feeds `MiniMaxH3ImageToVideo.first_frame`, found by
   walking the workflow's links backwards. The bundled Pixaroma workflow wires the loader
   through an intermediate resize node (`PixaromaLoadImageMini → PixaromaLongestSide →
   first_frame`), so a direct-edge check would miss it.
3. Otherwise, the workflow's only image loader, if there is exactly one.
4. Otherwise, an error listing the candidates and asking for `image_node_id`.

Recognised loaders: `LoadImage`, `LoadImageMask`, `LoadImageOutput`, `PixaromaLoadImage`,
`PixaromaLoadImageMini` — all of which use the same `image` input field.

### Image input safety

| Control | Behaviour |
|---|---|
| Scheme | HTTPS only (`H3_ALLOW_INSECURE_IMAGE_URL=1` opts into http for internal testing) |
| SSRF | Rejects loopback, private, link-local, reserved, multicast and unspecified addresses — including `169.254.169.254`. Redirects are followed manually so **every hop** is re-validated |
| Size | Capped at `H3_MAX_IMAGE_BYTES` (32 MB), enforced while streaming, so a false `Content-Length` cannot get past it |
| Pixels | Capped at `H3_MAX_IMAGE_PIXELS` (64 MP) against decompression bombs |
| Format | PNG, JPEG, WebP only — determined from magic bytes **and** confirmed by Pillow, so forged headers are rejected |
| Filenames | Always generated (`h3-input-<uuid>.<ext>`). The remote/user filename is never used, ruling out path traversal and collisions |
| Execution | Written mode `0644`; uploaded content is never executed |
| Cleanup | Removed in a `finally` block, so failed and rejected jobs cannot orphan files. Cleanup only ever deletes `h3-input-*` files inside the input directory, so shared inputs and model files are never touched |

The base64 payload is never logged.

## Test

### Direct RunPod test — text-to-video

`/run` is asynchronous and returns a job id immediately.

```bash
curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"workflow":{ ...ComfyUI API-format workflow... }}}'
```

A complete, ready-to-run workflow is in
[`examples/fl2va-text-to-video.json`](examples/fl2va-text-to-video.json). To submit it:

```bash
python -c "import json; json.dump({'input':{'workflow':json.load(open('examples/fl2va-text-to-video.json'))}}, open('/tmp/job.json','w'))"

curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/job.json
```

Response:

```json
{"id": "abc-123-...", "status": "IN_QUEUE"}
```

### Direct RunPod test — image-to-video

Uses [`examples/fl2va-image-to-video.json`](examples/fl2va-image-to-video.json), which adds a
`LoadImage` node wired into `MiniMaxH3ImageToVideo.first_frame`. Its `image` value is a
placeholder — the handler overwrites it with the staged upload.

```bash
python -c "import json; json.dump({'input':{'workflow':json.load(open('examples/fl2va-image-to-video.json')),'image_url':'https://example.com/frame.png'}}, open('/tmp/job-i2v.json','w'))"

curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/job-i2v.json
```

Or as a one-liner with an inline URL:

```bash
curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"input\":{\"workflow\":$(cat examples/fl2va-image-to-video.json),\"image_url\":\"https://example.com/frame.png\"}}"
```

With base64 instead of a URL:

```bash
python -c "
import base64, json
image = base64.b64encode(open('frame.png','rb').read()).decode()
json.dump({'input': {'workflow': json.load(open('examples/fl2va-image-to-video.json')),
                     'image_base64': image}}, open('/tmp/job-b64.json','w'))
"
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/job-b64.json
```

In the worker log you should see:

```
[handler] Fetching input image from image_url
[handler] Accepted png input image (412334 bytes, 1024x576)
[handler] Image node auto-detected via MiniMaxH3ImageToVideo.first_frame: #15
[handler] Staged input image as h3-input-9f2c….png and wired it into node #15
[handler] Job …: queueing image-to-video workflow with 15 nodes
...
[handler] Removed staged input image h3-input-9f2c….png
```

### Automated tests

Everything runs without a GPU, ComfyUI, a RunPod account or a network. The Python tests stub
`runpod` and `websocket`; the JavaScript tests drive the Worker's real `fetch` handler
against stub R2 and Durable Object bindings, and the browser module against Node's WebCrypto
- the same API the browser exposes.

```bash
python -m pytest tests -q     # 250 tests
node --test                   # 210 tests: Worker + browser client
```

**Handler and image input** — text-to-video is a no-op; both image inputs rejected;
PNG/JPEG/WebP accepted; `data:` URI prefix; non-image and forged-magic-byte payloads
rejected; oversized input rejected with no temp file left; SSRF guards (loopback, private,
link-local, `169.254.169.254`, non-HTTPS schemes); detection through an intermediate node
with a decoy loader present; ambiguous workflows requiring `image_node_id`; cleanup removing
only its own files; upload truthfulness (a 302 to an Access login page is not a success);
FlashBoot preload; phase timing.

**Confidential Generation** (`tests/test_confidential_v2.py`,
`tests/test_confidential_artifacts.py`, `worker.confidential.test.js`,
`client/confidential-generation.test.js`).

The security invariant is tested from both languages: hand an attacker the RunPod job input,
the job output, the stored bytes, the object metadata, the job state, the status response,
the public key, the wrapped file key and the container header; derive a candidate key from
every one of them; prove none decrypts. Then decrypt with the private key, which never
crossed the boundary.

Around it: the public key wraps and cannot decrypt; the private key unwraps; a wrong private
key fails; a modified wrapped key fails safely; **a file key wrapped to an attacker's own
public key, swapped into the header, breaks the video's authentication**; sixteen encryptions
of one generation id yield sixteen distinct file keys, nonces and ciphertexts; the plaintext
file key appears in neither the container nor the metadata; a correct passphrase unlocks the
private key and a wrong one fails; editing a bundle's public key, key id or KDF parameters
breaks the unlock; plaintext deleted after success, after wrapping failure, after encryption
failure and after upload failure; `H3_KEEP_OUTPUTS` unable to preserve confidential
plaintext; every validation and forbidden-field rejection; all four 422 refusals; ciphertext
response headers; status never returning a decryption secret; deletion of both modes through
one path; and v1 artefacts still parsing and decrypting.

**Redaction** (`tests/test_redaction.py`, and the log assertions in the Worker suite) — keys,
passphrases, `Authorization` and Cloudflare Access credentials removed; `keyId` and R2 object
keys preserved; and an assertion that the redaction lists in `worker.js` and `artifacts.py`
are identical, because they had already drifted once.

**Cross-language conformance** (`tests/vectors/confidential_container.json`) — two vectors.
The v1 vector pins one container's exact bytes: one passphrase, through one KDF, produces the
key that opens it, and Python must reproduce those bytes byte for byte. The v2 vector pins a
hybrid container produced by Python from nothing but a public key, which `worker.js` must
parse and the browser must open with the matching private key. Its ciphertext is deliberately
not byte-pinned — RSA-OAEP is randomised, and a fixed seed is something no sane API exposes.
A change to the layout, the canonical header serialization, the AAD binding or the RSA-OAEP
parameters fails at least one of the three suites.

### Status

```bash
curl \
  "https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer RUNPOD_API_KEY"
```

On success:

```json
{
  "id": "abc-123-...",
  "status": "COMPLETED",
  "output": {
    "prompt_id": "...",
    "images": [
      {
        "filename": "MiniMaxH3_00001.mp4",
        "subfolder": "",
        "type": "output",
        "data": "<base64 mp4>"
      }
    ]
  }
}
```

To save the MP4:

```bash
curl -s "https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
| python -c "import sys,json,base64; d=json.load(sys.stdin); open('out.mp4','wb').write(base64.b64decode(d['output']['images'][0]['data']))"
```

### Cancel

```bash
curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/cancel/JOB_ID" \
  -H "Authorization: Bearer RUNPOD_API_KEY"
```

RunPod cancels at the platform level and signals the worker. The handler installs SIGTERM/
SIGINT handlers that terminate the ComfyUI child process cleanly, so a cancelled job does
not leave a GPU-resident ComfyUI behind.

## Output storage

Output delivery is pluggable via `OutputStore` in `handler.py`, selected with
`H3_OUTPUT_MODE`.

### `base64` (default)

Returns the MP4 inline as base64 under `output.images[]`, matching the existing Cloudflare
Worker and Vince's `worker-comfyui` behaviour. Guarded by `H3_MAX_BASE64_BYTES` (default
180 MB) so an oversized video fails with a clear message rather than an opaque payload
error.

### `r2` — encrypted upload to Cloudflare R2 (server-held key)

> **Not Confidential Generation.** These are two different schemes and the difference is the
> whole point. This mode encrypts with a **server-held** key-encryption key
> (`H3_KEY_WRAP_KEY`) and uploads directly from the worker to R2 with baked-in R2
> credentials; the platform can decrypt what it stores.
> [Confidential Generation](#confidential-generation) encrypts with a **client-derived** key
> that the platform never holds, and travels the normal Worker upload path.
>
> They are not alternatives to each other: this is at-rest encryption under a key you
> manage, that one is user-controlled encryption. Confidential Generation is what a request
> asks for per job; this is a deployment-wide setting for the direct-to-R2 path, and is
> unused when the Cloudflare Worker supplies an `output` block (which it always does).

Set `H3_OUTPUT_MODE=r2` to encrypt every video **inside the worker** and upload only
ciphertext. Plaintext MP4s never reach R2.

Per video:

1. A fresh AES-256 data key (DEK) and a fresh 96-bit nonce are drawn from `secrets` (CSPRNG).
2. The MP4 is encrypted with **AES-256-GCM**, streamed in 4 MB chunks so memory stays flat.
3. Only the ciphertext is uploaded to R2.
4. The DEK is wrapped with a long-lived key-encryption key (KEK) using AES-256-GCM, with the
   R2 object key as additional authenticated data so a wrapped key cannot be replayed onto a
   different object.
5. The plaintext MP4 and the temporary ciphertext file are deleted from the worker.

**The raw data key is never returned and never logged.** Only the wrapped form is. If
`H3_KEY_WRAP_KEY` is not configured the worker refuses to start, rather than falling back to
handing out raw keys.

The GCM tag is returned separately; the R2 object is pure ciphertext
(`tag_included_in_ciphertext: false`).

Response entry:

```json
{
  "filename": "MiniMaxH3_00001.mp4",
  "type": "output",
  "storage": "r2",
  "bucket": "my-bucket",
  "r2_key": "videos/9f2c.../MiniMaxH3_00001.mp4",
  "ciphertext_bytes": 8123456,
  "ciphertext_sha256": "…",
  "encryption": {
    "algorithm": "AES-256-GCM",
    "version": "1",
    "iv": "<base64 96-bit nonce>",
    "tag": "<base64 128-bit GCM tag>",
    "tag_included_in_ciphertext": false,
    "wrap_algorithm": "AES-256-GCM",
    "wrapped_key": "<base64>",
    "wrapped_key_iv": "<base64>",
    "wrapped_key_aad": "r2_key",
    "key_id": "default"
  }
}
```

Configuration:

| Variable | Required | Purpose |
|---|---|---|
| `H3_OUTPUT_MODE=r2` | yes | Enable encrypted R2 upload |
| `H3_KEY_WRAP_KEY` | yes | Base64 32-byte AES-256 KEK used to wrap per-video keys |
| `H3_KEY_WRAP_KEY_ID` | no | Key id echoed back for rotation (default `default`) |
| `R2_BUCKET` | yes | Destination bucket |
| `R2_ACCOUNT_ID` | yes* | Cloudflare account id (*or set `R2_ENDPOINT`) |
| `R2_ENDPOINT` | no | Overrides the derived endpoint |
| `R2_ACCESS_KEY_ID` | yes | R2 access key |
| `R2_SECRET_ACCESS_KEY` | yes | R2 secret key |
| `R2_PREFIX` | no | Key prefix, e.g. `videos` |

Generate a KEK:

```bash
python -c "import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

Store it as a RunPod **secret**, not a plain environment variable. Note that
`H3_KEY_WRAP_KEY` is a placeholder for your key-management design: to use an external KMS
(Cloudflare Keyless, AWS KMS, Vault), replace `EncryptedR2Store._wrap_key` with a call to
that service — the rest of the handler is unaffected.

Decrypting a returned object (with the KEK):

```python
import base64, boto3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

meta = entry["encryption"]
dek = AESGCM(kek).decrypt(
    base64.b64decode(meta["wrapped_key_iv"]),
    base64.b64decode(meta["wrapped_key"]),
    entry["r2_key"].encode(),          # AAD binds the key to the object
)
ciphertext = s3.get_object(Bucket=entry["bucket"], Key=entry["r2_key"])["Body"].read()
mp4 = AESGCM(dek).decrypt(
    base64.b64decode(meta["iv"]),
    ciphertext + base64.b64decode(meta["tag"]),   # GCM tag is returned separately
    None,
)
```

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `COMFY_DIR` | `/opt/comfyui-baked` | ComfyUI root |
| `COMFY_PORT` | `8188` | Internal ComfyUI port (localhost only) |
| `COMFY_OUTPUT_DIR` | `/tmp/comfy-output` | Where ComfyUI writes results |
| `COMFY_STARTUP_TIMEOUT` | `600` | Seconds to wait for ComfyUI readiness |
| `COMFY_EXTRA_ARGS` | *(empty)* | Extra ComfyUI CLI arguments, shell-quoted (e.g. `--highvram`). Logged verbatim at startup |
| `H3_PERF_NODES` | `1` | `0` suppresses the per-node `[perf] nodes` line |
| `H3_FLASHBOOT_PRELOAD` | `0` | `1` stages the text encoder and DiT onto the GPU before the serverless loop starts |
| `H3_FLASHBOOT_PRELOAD_TIMEOUT` | `60` | Seconds before the preload is abandoned and ComfyUI interrupted |
| `H3_FLASHBOOT_PRELOAD_WIDTH` / `_HEIGHT` / `_LENGTH` | `32` / `32` / `5` | Synthetic graph size; the node schema's minimums |
| `H3_FLASHBOOT_PRELOAD_UNET` / `_CLIP` / `_VIDEO_VAE` / `_AUDIO_VAE` | baked model filenames | Must match the real workflow's loader inputs exactly |
| `H3_PERF_NODES_TOP` | `6` | How many nodes to show on that line |
| `COMFY_SAGE_ATTENTION3` | `1` (from base) | `0` disables SageAttention3 if it destabilises |
| `H3_SAGE_AUTODETECT` | `1` | Auto-disable SageAttention3 when the scheduled GPU is not the capability the wheel was built for. `0` forces `COMFY_SAGE_ATTENTION3` through unchecked |
| `SAGE_SUPPORTED_CC` | `12.0` (from base) | Compute capability the `sageattn3` wheel was compiled for |
| `H3_JOB_TIMEOUT` | `3000` | Max seconds for one workflow |
| `H3_WS_RECV_TIMEOUT` | `30` | Websocket recv timeout (timeouts are normal) |
| `H3_EAGER_START` | `1` | Start ComfyUI during cold start, not the first job |
| `H3_OUTPUT_MODE` | `base64` | `base64` or `r2` |
| `COMFY_INPUT_DIR` | `/workspace/runpod-slim/ComfyUI/input` | Where input images are staged |
| `H3_MAX_IMAGE_BYTES` | `33554432` (32 MB) | Max input image size |
| `H3_MAX_IMAGE_PIXELS` | `64000000` | Max input image pixels (bomb guard) |
| `H3_IMAGE_TIMEOUT` | `30` | `image_url` download timeout, seconds |
| `H3_ALLOW_INSECURE_IMAGE_URL` | `0` | `1` permits http image URLs (testing only) |
| `H3_OVERWRITE_PLAINTEXT` | `1` | Overwrite a plaintext artefact with random bytes before unlinking it. Worth something on ext4, close to nothing on overlayfs or an SSD with wear levelling - see docs/confidential-generation.md §7. `0` skips it |

SageAttention3 is controlled by an environment variable rather than removed, so it can be
turned off without rebuilding if it proves unstable under Serverless.

## Workflow contract

The workflow is passed through to ComfyUI unchanged. The Cloudflare Worker's H3 workflow
uses `UNETLoader`, `CLIPLoader`, `VAELoader`, `MiniMaxH3ImageToVideo`, `RandomNoise`,
`BasicGuider`, `KSamplerSelect`, `BasicScheduler`, `SamplerCustomAdvanced`, `VAEDecode`,
`VAEDecodeAudio`, `CreateVideo` and `SaveVideo`, with sampler `res_multistep`, scheduler
`simple` and 24 fps. All of these are present in the pinned ComfyUI build.

`MiniMaxH3ImageToVideo` takes required `clip, vae, prompt, width, height, length` with
`first_frame` / `last_frame` **optional** (it has no `model` input — the MODEL from
`UNETLoader` goes to `BasicGuider` and `BasicScheduler`). Those optional inputs are what let
one model serve both modes: omit them for text-to-video, wire a loader into `first_frame`
for image-to-video.

`SaveVideo` emits `ui.PreviewVideo`, whose dict form is `{"images": [...]}` — so saved MP4s
arrive under the `images` key, which is exactly what the Cloudflare Worker already reads.
The handler also collects `gifs`, `videos` and `audio` for robustness.

`length` must sit on the model's `17k + 5` grid (124 ≈ 5 s at 24 fps; the trained range is
roughly 124–362).

## Repository layout

```
Dockerfile                      Serverless image: replaces the Pod entrypoint
handler.py                      RunPod handler, ComfyUI supervision, output stores
artifacts.py                    Model-independent privacy layer: privacy modes, the
                                container format, public-key wrapping, plaintext hygiene
worker.js                       Cloudflare Worker: public API, workflow builder, R2, DO
wrangler.toml                   Worker bindings, mirrored from the live deployment
models.tsv                      Model URLs, exact sizes and SHA-256 checksums
client/confidential-generation.js   Browser key pair, passphrase-encrypted private key,
                                local unwrapping and decryption
client/demo.html                Minimal Confidential Generation UI
docs/confidential-generation.md Architecture, trust boundaries, threat model, key lifecycle
docs/privacy-statement.md       Product wording, and the claims never to make
scripts/build_model_layer.py    Streams a model from HF straight into a Docker layer
scripts/make_container_vector.py    Regenerates the cross-language conformance vector
scripts/set_r2_lifecycle.sh     Applies the prefix-scoped R2 retention policy
examples/fl2va-text-to-video.json    Ready-to-run FL2VA text-to-video workflow
examples/fl2va-image-to-video.json  Same, plus a LoadImage feeding first_frame
tests/                          Python suites; tests/vectors/ holds the format vectors
worker.*.test.js                Worker suites, driven through the real fetch handler
.github/workflows/build.yml     Two-stage build: code image, then four model layers
```
