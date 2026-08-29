# PrivoraVideo RunPod Integration Inputs

**Purpose:** exact downstream contract for the agent updating the RunPod-facing Cloudflare Worker, API Runtime, and PrivoraVideo frontend.

**Snapshot date:** 2026-08-29 (Australia/Sydney)

**Scope:** Gates 3–12 and further paid GPU testing are paused. This document is based on source inspection, local contract tests, and read-only inspection of the live Cloudflare deployment. The RunPod image hardening described here is authorized; this document does not authorize Cloudflare, API Runtime, frontend, or paid GPU changes.

## Executive result

The rebuilt RunPod handler already has the required ownership boundary:

    Frontend
      -> API Runtime
      -> RunPod-facing Cloudflare Worker
      -> RunPod queue envelope
      -> canonical PrivoraVideo request
      -> handler.py
      -> privora/request.py validation
      -> privora/prompt.py + privora/references.py compilation
      -> privora/models.py routing
      -> privora/workflows.py graph construction
      -> ComfyUI

The canonical branch is selected when <code>job["input"]</code> has no non-empty <code>workflow</code> object. The handler then calls <code>build_privora_job(job_input, generation_id)</code>. The immediate downstream change is therefore:

    CURRENT
    {"input":{"workflow":{...},"privacy":...,"progress":...,"output":...}}

    REQUIRED
    {"input":{"mode":"create","prompt":"...", ...canonical fields...,
              "privacy":...,"progress":...,"output":...}}

Cloudflare and API Runtime must stop calling or recreating <code>buildWorkflowForSettings</code>, canvas arithmetic, frame alignment, H3 prompt tags, model/checkpoint selection, Turbo selection, step selection, sampler settings, or reference-fidelity mapping for the canonical path.

The following remain downstream orchestration responsibilities:

- authenticate the caller;
- select the RunPod endpoint;
- validate the public envelope sufficiently to reject obviously malformed/unsafe input;
- create the Cloudflare job ID;
- mint job-scoped callback and asset tokens;
- make reference bytes retrievable under the privacy rules in this document;
- submit, poll, cancel, persist product status, and expose the stored artifact;
- forward semantic product fields without translating them into ComfyUI concepts.

## Source-of-truth order

1. Rebuilt RunPod source in this worktree: <code>handler.py</code>, <code>privora/</code>, <code>artifacts.py</code>, <code>Dockerfile</code>, <code>models.tsv</code>, and tests.
2. Read-only live Cloudflare deployment inspection for the binding set and currently deployed routing behavior.
3. Committed <code>worker.js</code> and <code>wrangler.toml</code> for the legacy public contract.
4. RunPod’s queue API only for transport behavior not implemented inside this repository.

Official transport references used for item 4:

- [RunPod: Send requests](https://docs.runpod.io/serverless/endpoints/send-requests)
- [RunPod: Endpoint operations](https://docs.runpod.io/serverless/endpoints/operation-reference)
- [RunPod: Job states](https://docs.runpod.io/serverless/endpoints/job-states)

The active Cloudflare deployment is newer than committed <code>worker.js</code>. Its active version is <code>3fa6e25a-a325-4d94-9d6a-31cf7af1f981</code>, created <code>2026-08-29T04:18:58Z</code>, tag <code>privora-privacy-audit-v1</code>. Read-only inspection confirmed that it still constructs a ComfyUI <code>workflow</code> before calling RunPod. Its recent additions sanitize downstream errors and expose status progress; they do not implement the canonical Privora request.

## Hard migration rules

1. For new requests, never put <code>workflow</code> in <code>input</code>.
2. Do not send <code>width</code>, <code>height</code>, <code>frames</code>, or <code>steps</code> in the canonical schema.
3. Do not construct <code>&lt;Picture n&gt;</code>, <code>&lt;Video n&gt;</code>, or <code>&lt;Audio n&gt;</code> tags downstream.
4. Do not send LoRA filenames, checkpoint filenames, sampler names, or scheduler names downstream.
5. Treat the rebuilt worker’s capability result as authoritative for the image that answered it.
6. Keep <code>backend</code> outside the RunPod <code>input</code>; it selects a Cloudflare routing target and is not a Privora request field.
7. Treat <code>progress</code>, <code>output</code>, reference <code>token</code> values, and any signed asset URLs as trusted Cloudflare enrichment. Do not accept caller-selected callback URLs or tokens.
8. A RunPod outer status of <code>COMPLETED</code> is not sufficient: if <code>output.error</code> is present, the product result is failed.

# 1. RunPod endpoint connection

## Endpoint identifiers

The current Cloudflare backend map is exact:

| Cloudflare backend | Endpoint variable | Live value | API-key variable |
| --- | --- | --- | --- |
| <code>h3</code> | <code>RUNPOD_ENDPOINT_ID</code> | <code>xa6b4vs5gdva3r</code> | <code>RUNPOD_API_KEY</code> |
| <code>h3-blackwell</code> | <code>RUNPOD_BLACKWELL_ENDPOINT_ID</code> | <code>d0p4f4hgxyqsy2</code> | <code>RUNPOD_BLACKWELL_API_KEY</code>, falling back to <code>RUNPOD_API_KEY</code> |

The hardened image workflow’s new immutable registry tag is <code>multimodal-3</code> in <code>.github/workflows/build-dual-gpu.yml</code>. That tag is not an endpoint ID and must not be placed in a RunPod API URL. Repository source does not encode the RunPod dashboard display name. Do not assume the older <code>multimodal-2</code> digest contains these fixes.

Backend aliases currently accepted by Cloudflare are:

- <code>h3</code>: <code>h3</code>, <code>default</code>, <code>original</code>, <code>ada</code>, <code>h3-ada</code>, <code>48gb-pro</code>, <code>minimax-h3</code>
- <code>h3-blackwell</code>: <code>blackwell</code>, <code>h3-blackwell</code>, <code>h3_blackwell</code>, <code>minimax-h3-blackwell</code>

The canonical Privora migration target is the rebuilt Blackwell/multimodal image. Endpoint selection remains a Cloudflare/API Runtime concern; model-family selection inside that endpoint does not.

## API URL format

The committed and active Worker use these operations:

| Operation | Method and exact URL |
| --- | --- |
| Asynchronous submit | <code>POST https://api.runpod.ai/v2/{endpointId}/run</code> |
| Status | <code>GET https://api.runpod.ai/v2/{endpointId}/status/{runpodJobId}</code> |
| Cancel | <code>POST https://api.runpod.ai/v2/{endpointId}/cancel/{runpodJobId}</code> |

RunPod also defines:

| Operation | Method and exact URL | Repository use |
| --- | --- | --- |
| Synchronous submit | <code>POST https://api.runpod.ai/v2/{endpointId}/runsync</code> | Not currently called by <code>worker.js</code> |
| Queue/worker health | <code>GET https://api.runpod.ai/v2/{endpointId}/health</code> | Not currently called by <code>worker.js</code> |

The RunPod health operation is not the Privora capability operation. Privora capabilities are a normal handler request with <code>mode: "capabilities"</code>, described in section 4.

The current Worker’s outer queue policy is literal code, not an environment binding:

    "policy": {
      "executionTimeout": 600000,
      "ttl": 1800000
    }

No source in this repository consumes <code>RUNPOD_ENDPOINT_URL</code>, a polling-interval environment variable, a status-URL environment variable, or a callback-URL environment variable.

## Authentication

Every Cloudflare → RunPod operation uses:

    Authorization: Bearer <value of the selected RunPod API-key secret>

Submission also uses:

    Content-Type: application/json

Do not expose either RunPod API key to API Runtime clients or the frontend.

RunPod → Cloudflare callback and asset requests use two independent credentials when Cloudflare Access is enabled:

    CF-Access-Client-Id: <Access service-token client id>
    CF-Access-Client-Secret: <Access service-token secret>
    Authorization: Bearer <job-scoped HMAC token minted by Cloudflare>

The Access token decides whether the request reaches the Worker. The job token authorizes exactly one job and one purpose after it reaches the Worker.

# 2. Complete environment variable and binding inventory

Classification in this document means:

- **EXISTING:** confirmed in the live binding set or committed image defaults.
- **NEWLY REQUIRED:** required on the named system for the canonical integration if it is not already present there.
- **OPTIONAL:** code has a default or the feature is optional.
- **LEGACY:** accepted for compatibility or for the direct-R2 fallback, but not the target architecture.

No secret values are included.

## Cloudflare Worker

| Exact name | Kind | Secret? | Required? | Class | Purpose and exact consumer |
| --- | --- | ---: | ---: | --- | --- |
| <code>RUNPOD_ENDPOINT_ID</code> | plain variable | no | for <code>h3</code> | EXISTING | Endpoint selected by <code>BACKENDS.h3</code> in <code>worker.js</code>. |
| <code>RUNPOD_BLACKWELL_ENDPOINT_ID</code> | plain variable | no | for <code>h3-blackwell</code> | EXISTING | Endpoint selected by <code>BACKENDS["h3-blackwell"]</code>. This is the current rebuilt-image route. |
| <code>RUNPOD_API_KEY</code> | Worker secret | yes | yes | EXISTING | RunPod bearer credential for <code>h3</code>; also the fallback for Blackwell. |
| <code>RUNPOD_BLACKWELL_API_KEY</code> | Worker secret | yes | no if fallback is intended; otherwise yes | EXISTING | Preferred RunPod bearer credential for <code>h3-blackwell</code>. |
| <code>JOB_TOKEN_SECRET</code> | Worker secret | yes | yes for callbacks/R2; mandatory for Confidential | EXISTING | HMAC signing/verification for <code>progress</code>, <code>output-upload</code>, and <code>asset-download</code> job tokens. |
| <code>JOB_TOKEN_TTL_SECONDS</code> | plain variable | no | no | EXISTING | Token lifetime; default and live value <code>3600</code>. |
| <code>PUBLIC_BASE_URL</code> | plain variable | no | no | OPTIONAL | Overrides the origin used to create internal callback and asset URLs. |
| <code>H3_OUTPUTS</code> | R2 binding | n/a | yes for callback storage | EXISTING | Bucket binding read by <code>requireBucket()</code>. Live bucket: <code>minimax-h3-private-output</code>. |
| <code>JOB_CHANNEL</code> | Durable Object binding | n/a | yes for realtime/product progress | EXISTING | One <code>JobChannel</code> per Cloudflare job ID. |

Live read-only <code>wrangler secret list</code> confirmed exactly three Worker secrets: <code>JOB_TOKEN_SECRET</code>, <code>RUNPOD_API_KEY</code>, and <code>RUNPOD_BLACKWELL_API_KEY</code>. Live version inspection confirmed the bindings above match <code>wrangler.toml</code>.

The Cloudflare account identifier is <code>899c4111fb42559764b1bd1118d8cf79</code>. It is not read by <code>worker.js</code> and is not required in a request; Wrangler currently infers it from authentication.

Object-key contracts are Worker-owned:

| Object | Exact key |
| --- | --- |
| Current input image | <code>inputs/{cloudflareJobId}/{assetId}.{png|jpg|webp}</code> |
| Standard output | <code>outputs/{cloudflareJobId}/video.mp4</code> |
| Confidential output | <code>outputs/{cloudflareJobId}/artifact.enc</code> |

RunPod never chooses an R2 key.

## Cloudflare Access and API Runtime

The HTTP header names accepted by the Access-protected Worker are exact:

| Exact name | Kind | Secret? | Required? | Class | Where consumed |
| --- | --- | ---: | ---: | --- | --- |
| <code>CF-Access-Client-Id</code> | HTTP header | sensitive identifier | yes when Access protects the route | EXISTING | Cloudflare Access, before <code>worker.js</code>. |
| <code>CF-Access-Client-Secret</code> | HTTP header | yes | yes when Access protects the route | EXISTING | Cloudflare Access, before <code>worker.js</code>. |

The local protected-endpoint configuration uses those exact hyphenated header names. The API Runtime source itself is not present in this worktree, so this repository cannot prove how that separate application names its stored secrets. If its secret names are <code>CF_ACCESS_CLIENT_ID</code> and <code>CF_ACCESS_CLIENT_SECRET</code>, it must translate them to the two headers above.

Existing project information identifies these API Runtime names, so they should not be renamed during this migration:

| Exact name | System | Secret? | Required? | Class | Purpose |
| --- | --- | ---: | ---: | --- | --- |
| <code>CF_ACCESS_CLIENT_ID</code> | API Runtime | sensitive identifier | when Worker is Access-protected | EXISTING (reported; API Runtime source is external to this worktree) | Value sent in HTTP header <code>CF-Access-Client-Id</code>. |
| <code>CF_ACCESS_CLIENT_SECRET</code> | API Runtime | yes | when Worker is Access-protected | EXISTING (reported; API Runtime source is external to this worktree) | Value sent in HTTP header <code>CF-Access-Client-Secret</code>. |

Those same underscore names are also the preferred environment names consumed by the rebuilt RunPod handler for its outbound callbacks:

| Exact name | System | Secret? | Required? | Class | Purpose and where consumed |
| --- | --- | ---: | ---: | --- | --- |
| <code>CF_ACCESS_CLIENT_ID</code> | RunPod endpoint environment | sensitive identifier | when callback host is Access-protected | NEWLY REQUIRED on RunPod if absent | <code>handler.py::cloudflare_access_headers()</code> emits <code>CF-Access-Client-Id</code>. |
| <code>CF_ACCESS_CLIENT_SECRET</code> | RunPod endpoint environment | yes | when callback host is Access-protected | NEWLY REQUIRED on RunPod if absent | Emits <code>CF-Access-Client-Secret</code>. |
| <code>CLOUDFLARE_ACCESS_CLIENT_ID</code> | RunPod endpoint environment | sensitive identifier | no | OPTIONAL alias | Second-precedence alias for <code>CF_ACCESS_CLIENT_ID</code>. |
| <code>CLOUDFLARE_ACCESS_CLIENT_SECRET</code> | RunPod endpoint environment | yes | no | OPTIONAL alias | Second-precedence alias for <code>CF_ACCESS_CLIENT_SECRET</code>. |
| <code>CLOUDFLARE_ACCESS_KEY_ID</code> | RunPod endpoint environment | sensitive identifier | no | LEGACY alias | Third-precedence alias. The name is ambiguous with R2 S3 credentials; do not use for new configuration. |
| <code>CLOUDFLARE_SECRET_ACCESS_KEY</code> | RunPod endpoint environment | yes | no | LEGACY alias | Third-precedence alias; same ambiguity. |

Do not confuse a Cloudflare Access service token with R2’s S3 API access key.

## Rebuilt RunPod handler: integration and media limits

| Exact name | Secret? | Required? | Class/default | Purpose and exact consumer |
| --- | ---: | ---: | --- | --- |
| <code>COMFY_DIR</code> | no | image default | EXISTING, <code>/opt/comfyui-baked</code> | ComfyUI root and installed-model inventory. |
| <code>COMFY_PORT</code> | no | no | EXISTING, <code>8188</code> | Private local ComfyUI HTTP/WebSocket port. |
| <code>COMFY_INPUT_DIR</code> | no | image default | EXISTING | Per-job plaintext reference staging root. |
| <code>COMFY_OUTPUT_DIR</code> | no | image default | EXISTING | ComfyUI output root. |
| <code>COMFY_TEMP_DIR</code> | no | image default | EXISTING | ComfyUI temporary output root. |
| <code>COMFY_STARTUP_TIMEOUT</code> | no | no | OPTIONAL, <code>600</code> seconds | Cold-start ceiling. |
| <code>H3_JOB_TIMEOUT</code> | no | no | OPTIONAL, <code>3000</code> seconds | End-to-end ComfyUI execution deadline in the handler. |
| <code>H3_WS_RECV_TIMEOUT</code> | no | no | OPTIONAL, <code>30</code> seconds | ComfyUI WebSocket receive interval. |
| <code>H3_OUTPUT_MODE</code> | no | no with <code>output.url</code> | EXISTING, Docker default <code>base64</code> | Process-wide fallback store. A job-scoped <code>output.url</code> takes precedence and is the target architecture. |
| <code>H3_OUTPUT_UPLOAD_TIMEOUT</code> | no | no | OPTIONAL, <code>120</code> seconds | Default timeout for binary output PUT; per-request <code>output.timeout</code> can override it. |
| <code>H3_PROGRESS_TIMEOUT</code> | no | no | OPTIONAL, <code>3</code> seconds | Timeout for each progress callback. |
| <code>H3_PROGRESS_MIN_INTERVAL</code> | no | no | OPTIONAL, <code>0.4</code> seconds | Minimum interval between non-final sampler-step callbacks. |
| <code>H3_MAX_IMAGE_BYTES</code> | no | no | OPTIONAL, <code>33554432</code> | Image/keyframe/reference-image byte ceiling. |
| <code>H3_MAX_IMAGE_PIXELS</code> | no | no | OPTIONAL, <code>64000000</code> | Decoded image pixel ceiling. |
| <code>H3_IMAGE_TIMEOUT</code> | no | no | OPTIONAL, <code>30</code> seconds | Reference URL request timeout. |
| <code>H3_ALLOW_INSECURE_IMAGE_URL</code> | no | no | OPTIONAL, disabled | Allows HTTP only when exactly <code>1</code>; testing escape hatch, not production configuration. |
| <code>H3_MAX_REF_VIDEO_BYTES</code> | no | no | OPTIONAL, <code>268435456</code> | Reference-video byte ceiling. |
| <code>H3_MAX_REF_AUDIO_BYTES</code> | no | no | OPTIONAL, <code>67108864</code> | Standalone/nested reference-audio byte ceiling. |
| <code>H3_MEDIA_PROBE_TIMEOUT</code> | no | no | OPTIONAL, <code>20</code> seconds | <code>ffprobe</code> deadline per video/audio reference. |
| <code>H3_MAX_BASE64_BYTES</code> | no | no | OPTIONAL, <code>188743680</code> | Legacy inline-output ceiling. Not a reference limit. |
| <code>H3_KEEP_OUTPUTS</code> | no | no | OPTIONAL, <code>0</code> | Debug retention for standard local outputs. Ignored for Confidential plaintext cleanup. |
| <code>H3_OVERWRITE_PLAINTEXT</code> | no | no | OPTIONAL, <code>1</code> | Best-effort overwrite before deleting Confidential plaintext. |
| <code>COMFYUI_H3_COMMIT</code> | no | image default | EXISTING, Docker default <code>dec5d945</code> | Reported as <code>capabilities.worker.comfyuiCommit</code>. |
| <code>H3_BUILD_SOURCE_COMMIT</code> | no | image default | NEW, build-injected | Immutable source commit reported as <code>capabilities.worker.build.sourceCommit</code>. Do not set per request. |
| <code>H3_BUILD_IMAGE_TAG</code> | no | image default | NEW, build-injected | Published image tag reported as <code>capabilities.worker.build.imageTag</code>. |
| <code>H3_BUILD_ID</code> | no | image default | NEW, build-injected | CI run/attempt identity reported as <code>capabilities.worker.build.buildId</code>. |
| <code>H3_GPU_MODE</code> | no | endpoint operation | EXISTING, Docker default <code>single</code> | <code>single</code> or <code>dual</code>; reported as <code>capabilities.worker.gpuMode</code>. It is not a request field. |

Boot/performance variables such as <code>COMFY_EXTRA_ARGS</code>, <code>H3_EAGER_START</code>, <code>H3_FLASHBOOT_PRELOAD*</code>, <code>H3_SAGE_AUTODETECT</code>, <code>H3_PERF_NODES*</code>, <code>H3_SP_*</code>, and <code>NCCL_*</code> are endpoint/image operations, not downstream integration inputs. API Runtime and frontend must not set or infer them per request.

## Direct-R2 output fallback: legacy/optional, not the target

These are consumed only when the handler falls back to <code>H3_OUTPUT_MODE=r2</code>. The target callback architecture uses Cloudflare’s <code>H3_OUTPUTS</code> binding instead.

| Exact name | Secret? | Required in fallback? | Class | Purpose |
| --- | ---: | ---: | --- | --- |
| <code>R2_BUCKET</code> | no | yes | LEGACY | S3 bucket name. |
| <code>R2_PREFIX</code> | no | no | LEGACY | Optional output key prefix. |
| <code>R2_ACCOUNT_ID</code> | no | if <code>R2_ENDPOINT</code> absent | LEGACY | Derives the S3 endpoint. |
| <code>R2_ENDPOINT</code> | no | if account ID absent | LEGACY | Explicit S3 endpoint. |
| <code>R2_REGION</code> | no | no | LEGACY, <code>auto</code> | S3 region. |
| <code>R2_ACCESS_KEY_ID</code> | yes | yes | LEGACY | R2 S3 credential. |
| <code>R2_SECRET_ACCESS_KEY</code> | yes | yes | LEGACY | R2 S3 credential. |
| <code>H3_KEY_WRAP_KEY</code> | yes | yes | LEGACY | Base64 32-byte AES KEK for the old server-held wrapping design. Do not use for canonical Confidential v2. |
| <code>H3_KEY_WRAP_KEY_ID</code> | no | no | LEGACY, <code>default</code> | Identifier for that legacy KEK. |

## Callback and routing values that are payload fields, not environment variables

| Exact field | Created by | Consumer |
| --- | --- | --- |
| <code>progress.url</code>, <code>progress.token</code>, <code>progress.jobId</code> | Cloudflare | <code>handler.py::ProgressReporter</code> |
| <code>output.url</code>, <code>output.token</code>, <code>output.jobId</code>, optional <code>output.timeout</code> | Cloudflare | <code>handler.py::WorkerUploadStore</code> |
| Reference <code>url</code> and optional <code>token</code> | Cloudflare/API Runtime media transport | RunPod reference staging |

There is no implemented global callback URL, status URL, polling interval, or feature-flag variable for enabling the canonical parser. The absence of <code>workflow</code> selects it.

# 3. Exact RunPod request envelope

## Transport envelope

For asynchronous submission, Cloudflare must POST this outer shape to RunPod:

    {
      "input": {
        "...": "canonical PrivoraVideo request plus trusted orchestration fields"
      },
      "policy": {
        "executionTimeout": 600000,
        "ttl": 1800000
      }
    }

The value of <code>input</code> is passed to <code>handler.py</code> as <code>job["input"]</code>. Do not add another wrapper such as <code>request</code>, <code>payload</code>, or <code>privora</code>.

## Minimal canonical request inside the envelope

This is a complete normal text-to-video request:

    {
      "input": {
        "mode": "create",
        "prompt": "A red fox runs through wet ferns at dawn.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "generationMode": "quality",
        "privacy": {
          "mode": "standard"
        }
      },
      "policy": {
        "executionTimeout": 600000,
        "ttl": 1800000
      }
    }

With the existing Cloudflare callback/storage architecture, the actual trusted request should be:

    {
      "input": {
        "mode": "create",
        "prompt": "A red fox runs through wet ferns at dawn.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "generationMode": "quality",
        "privacy": {
          "mode": "standard"
        },
        "progress": {
          "url": "https://<worker-host>/internal/jobs/<cloudflareJobId>/progress",
          "token": "<job-scoped progress token>",
          "jobId": "<cloudflareJobId>"
        },
        "output": {
          "url": "https://<worker-host>/internal/jobs/<cloudflareJobId>/output",
          "token": "<job-scoped output-upload token>",
          "jobId": "<cloudflareJobId>"
        }
      },
      "policy": {
        "executionTimeout": 600000,
        "ttl": 1800000
      }
    }

The angle-bracketed values above are runtime values, not literal strings. Their formats are defined by the existing Worker:

- <code>cloudflareJobId</code> is <code>crypto.randomUUID()</code>.
- <code>progress.url</code> is <code>{base}/internal/jobs/{jobId}/progress</code>.
- <code>output.url</code> is <code>{base}/internal/jobs/{jobId}/output</code>.
- <code>base</code> is <code>PUBLIC_BASE_URL</code> when set, otherwise the public request origin.
- Tokens are HMAC-signed by <code>JOB_TOKEN_SECRET</code>, scoped to the job ID, purpose, and expiry.

## Field ownership at the Cloudflare boundary

| Fields | Origin | Rule |
| --- | --- | --- |
| <code>mode</code>, <code>prompt</code>, <code>quality</code>, <code>aspectRatio</code>, <code>duration</code>, <code>seed</code>, <code>generationMode</code>, <code>camera</code>, <code>style</code>, <code>referenceFidelity</code>, keyframes/references, public-key encryption fields | Frontend via API Runtime | Validate and forward semantically. Do not compile. |
| <code>backend</code> or <code>model</code> | API Runtime/Cloudflare routing | Use to select an endpoint, then remove from RunPod <code>input</code>. |
| <code>progress</code>, <code>output</code> | Cloudflare only | Replace any caller-supplied values. These are authenticated internal destinations. |
| Reference <code>token</code> and job-scoped reference <code>url</code> | Trusted media transport layer | Replace any client token. A caller may supply media bytes, but may not choose an internal authorization token. |
| <code>policy</code> | Cloudflare | RunPod transport field, outside <code>input</code>. |
| <code>workflow</code> | Nobody on canonical path | Must be absent. |

# 4. Capabilities request and response

## Exact handler request

    {
      "input": {
        "mode": "capabilities"
      }
    }

The handler returns before starting ComfyUI, loading a model, or staging media:

    {
      "capabilities": {
        "...": "document below"
      }
    }

When queried through RunPod <code>/run</code>, that handler result appears under the queue response's <code>output</code> after polling completes:

    {
      "id": "<runpodJobId>",
      "status": "COMPLETED",
      "output": {
        "capabilities": {
          "...": "document below"
        }
      }
    }

The repository does not currently call <code>/runsync</code>. A downstream agent may choose it for this cheap probe only after confirming its timeout/cold-start behavior; that choice does not change the inner request or handler response.

## Exact capability document for a complete hardened model inventory

Dynamic values are explicitly labelled. Every other value below is generated by the rebuilt source:

    {
      "modes": {
        "create": {
          "family": "fl2va",
          "description": "Text to video with native audio.",
          "available": true
        },
        "animate": {
          "family": "fl2va",
          "description": "Text to video anchored to a first and/or last keyframe.",
          "available": true
        },
        "references": {
          "family": "ref2va",
          "description": "Reference-guided generation from images, videos and audio.",
          "available": true
        },
        "remix": {
          "family": "ref2va",
          "description": "Reference-guided regeneration using a source video. This is not deterministic video editing: H3 regenerates the clip guided by the references rather than modifying the source frames, so unreferenced detail will change.",
          "available": true
        }
      },
      "quality": {
        "draft": {
          "steps": 20,
          "dimensions": {
            "16:9": "512x288",
            "9:16": "288x512",
            "1:1": "384x384",
            "4:3": "448x320",
            "3:4": "320x448",
            "21:9": "576x256"
          }
        },
        "standard": {
          "steps": 20,
          "dimensions": {
            "16:9": "1024x576",
            "9:16": "576x1024",
            "1:1": "768x768",
            "4:3": "896x672",
            "3:4": "672x896",
            "21:9": "1152x512"
          }
        },
        "hd": {
          "steps": 20,
          "dimensions": {
            "16:9": "1248x704",
            "9:16": "704x1248",
            "1:1": "768x768",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "21:9": "1440x640"
          }
        },
        "ultra": {
          "steps": 20,
          "dimensions": {
            "16:9": "1344x768",
            "9:16": "768x1344",
            "1:1": "768x768",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "21:9": "1536x672"
          }
        }
      },
      "aspectRatios": [
        "16:9",
        "1:1",
        "21:9",
        "3:4",
        "4:3",
        "9:16"
      ],
      "duration": {
        "fps": 24,
        "frameGrid": "frames % 17 == 5",
        "minSeconds": 0.2083,
        "maxSeconds": 149.6667,
        "trainedRangeSeconds": [
          5.1667,
          15.0833
        ]
      },
      "references": {
        "maxImages": 9,
        "maxVideos": 3,
        "maxAudio": 3,
        "maxVideoSoundtracks": 3,
        "maxTotal": 12,
        "modelMaxTotal": 18,
        "videoSeconds": [
          2.0,
          15.0
        ],
        "audioMaxSeconds": 15.0,
        "roles": {
          "image": [
            "character",
            "clothing",
            "composition",
            "environment",
            "face",
            "general",
            "identity",
            "object",
            "pose",
            "product",
            "style"
          ],
          "video": [
            "action",
            "body_performance",
            "camera_motion",
            "general",
            "motion",
            "scene_structure",
            "source",
            "timing",
            "visual_style"
          ],
          "audio": [
            "ambience",
            "dialogue",
            "general",
            "music",
            "rhythm",
            "sound_effect",
            "voice"
          ]
        },
        "fidelity": [
          "high",
          "standard"
        ]
      },
      "maxSeed": 18446744073709551615,
      "models": {
        "fl2va": true,
        "ref2va": true
      },
      "generationModes": {
        "quality": true,
        "turbo": true,
        "turboFast": true
      },
      "turbo": {
        "fl2va": {
          "8step": true,
          "4step": true
        },
        "ref2va": {
          "4step": true
        }
      },
      "byFamily": {
        "fl2va": [
          "quality",
          "turbo",
          "turboFast"
        ],
        "ref2va": [
          "quality",
          "turboFast"
        ]
      },
      "worker": {
        "processId": "<dynamic process UUID>",
        "gpuMode": "<single, dual, or unresolved>",
        "comfyuiCommit": "dec5d945",
        "build": {
          "sourceCommit": "<H3_BUILD_SOURCE_COMMIT>",
          "imageTag": "<H3_BUILD_IMAGE_TAG>",
          "buildId": "<H3_BUILD_ID>"
        }
      }
    }

Use these fields as follows:

- <code>modes.{mode}.available</code> is authoritative for a product mode.
- <code>models.ref2va</code> is authoritative for the checkpoint's presence.
- <code>byFamily</code> is authoritative for which generation modes a family can execute.
- <code>turbo</code> is authoritative for actual distilled configurations.
- <code>generationModes</code> is only a global "available in at least one family" summary. It must not be used alone to offer 8-step Turbo for Ref2VA.
- <code>quality.*.dimensions</code>, <code>duration</code>, and <code>references</code> are the source-defined product limits.
- <code>worker.processId</code> identifies a warm worker process, not a job or an image.
- <code>worker.gpuMode</code> reports the resolved endpoint execution mode.
- <code>worker.build.sourceCommit</code>, <code>worker.build.imageTag</code>, and <code>worker.build.buildId</code> identify the source and immutable release build that answered the probe. They are injected by the image build and are not request fields.

The capability response does not expose a Docker digest, RunPod endpoint version, LoRA filename, checkpoint filename, model checksum, or model repository revision. It does expose the source commit, image tag, and build-run identity. A digest cannot reliably be discovered from inside the container and is therefore recorded in this handoff after publication rather than fabricated at runtime.

The build manifest pins:

| Files | Revision |
| --- | --- |
| Base FL2VA, Ref2VA, Qwen text encoder, video VAE, audio VAE | <code>eb8a16107c595128b3a578f82d2ce2f75920c355</code> |
| FL2VA 4-step/8-step and Ref2VA 4-step LoRAs | <code>4cc1d817b6184899b41293954329f576cb5ae86b</code> |

Those revisions remain source/build facts rather than runtime capability fields. Downstream should correlate the runtime <code>worker.build</code> object with the published tag/digest recorded in the release identity section of this handoff; <code>processId</code> remains only a warm-process identifier.

## Current public capability defect

The active Cloudflare <code>GET /capabilities</code> is a static legacy document. It reports the old quality table, omits <code>21:9</code>, and reports reference generation unavailable because it says the Ref2VA checkpoint is excluded. That result is stale for the complete <code>multimodal-2</code> inventory and for the hardened <code>multimodal-3</code> target.

The downstream migration must make runtime worker capabilities authoritative. Cloudflare may cache a successful probe, but it must not overwrite <code>modes</code>, <code>models</code>, <code>byFamily</code>, <code>turbo</code>, reference limits, canvas dimensions, or duration values with its old tables.

# 5. Create request

## Copyable normal request

    {
      "input": {
        "mode": "create",
        "prompt": "A cinematic tracking shot of a red fox running through wet ferns at dawn.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "generationMode": "quality",
        "seed": 51,
        "privacy": {
          "mode": "standard"
        }
      }
    }

Omit <code>seed</code> for a worker-generated seed. Add trusted <code>progress</code> and <code>output</code> blocks in Cloudflare as shown in section 3.

## Canonical product fields

| Exact field | JSON type | Required? | Allowed/default | Validation | Returned equivalent |
| --- | --- | ---: | --- | --- | --- |
| <code>mode</code> | string | no | <code>create</code> default; <code>create</code>, <code>animate</code>, <code>references</code>, <code>remix</code> | Trimmed and lowercased. | <code>output.generation.mode</code> |
| <code>prompt</code> | string | yes | non-empty | Trimmed; non-string or empty is <code>MISSING_PROMPT</code>. User-supplied H3-looking tags are neutralized. | Not returned. |
| <code>quality</code> | string | no | <code>standard</code> default; <code>draft</code>, <code>standard</code>, <code>hd</code>, <code>ultra</code> | Exact after trim/lowercase. | <code>output.generation.quality</code> plus actual dimensions |
| <code>aspectRatio</code> | string | no | <code>16:9</code> default; <code>16:9</code>, <code>9:16</code>, <code>1:1</code>, <code>4:3</code>, <code>3:4</code>, <code>21:9</code> | Semantic ratio only. | <code>output.generation.aspectRatio</code> plus actual dimensions |
| <code>duration</code> | number | no | <code>5</code> seconds | Must be positive and not Boolean; worker converts and aligns frames. | <code>output.generation.durationSeconds</code>, <code>frames</code>, <code>fps</code>, optional adjustment flags |
| <code>seed</code> | integer | no | worker-generated | Boolean rejected; range <code>0..18446744073709551615</code> inclusive. Omitted generation uses <code>random.randrange(0, MAX_SEED)</code>, so its random upper bound is exclusive even though an explicit maximum is accepted. | <code>output.generation.seed</code> |
| <code>generationMode</code> | string | no | <code>quality</code> default; canonical values <code>quality</code>, <code>turbo</code>, <code>turboFast</code> | Availability validated against the selected family and installed files. Parser also accepts <code>turbo_fast</code> and <code>turbo-fast</code>, but downstream should emit the canonical camel-case value. | <code>output.generation.generationMode</code>, <code>steps</code>, <code>acceleration</code>, optional <code>accelerationNote</code> |
| <code>firstFrame</code> | object | conditional | image reference | Required with <code>lastFrame</code> and/or alone for <code>animate</code>; rejected by <code>create</code>, <code>references</code>, and <code>remix</code>. Explicit non-image types are rejected. | <code>output.generation.keyframes.first</code> |
| <code>lastFrame</code> | object | conditional | image reference | Same mode/type rules as <code>firstFrame</code>. | <code>output.generation.keyframes.last</code> |
| <code>references</code> | array of objects | conditional | default empty | Required non-empty for <code>references</code>; <code>remix</code> specifically requires a <code>video</code> with role <code>source</code>; forbidden non-empty for <code>create</code>/<code>animate</code>. | Counts only in <code>output.generation.references</code> |
| <code>referenceFidelity</code> | string | no | <code>standard</code>; <code>standard</code> or <code>high</code> | Maps internally to node values; downstream must not send node values. | <code>output.generation.references.fidelity</code> when references exist |
| <code>camera</code> | object | no | fields below | Recognized enum values compile to prompt prose; unknown values currently contribute nothing. Downstream should require an object. | Not returned. |
| <code>style</code> | object | no | fields below | Same behavior. | Not returned. |
| <code>privacy</code> | object | no | <code>{"mode":"standard"}</code> by default | Implemented modes are <code>standard</code>, <code>confidential</code>; <code>private</code>/<code>ephemeral</code> are declared but rejected. | <code>output.privacyMode</code> |
| <code>encryption</code> | object | only for Confidential | section 9 | Must be absent for standard; required for confidential. | Public protection metadata only |
| <code>output</code> | object | required for target callback storage and Confidential | trusted internal block | Cloudflare-owned. | Produces metadata-only <code>images[]</code> and <code>video</code> |
| <code>progress</code> | object | no | trusted internal block | Cloudflare-owned. | Callback stream, not generation metadata |

The canonical parser is strict. Its explicit top-level allowlist is exactly <code>mode</code>, <code>prompt</code>, <code>quality</code>, <code>aspectRatio</code>, <code>duration</code>, <code>seed</code>, <code>generationMode</code>, <code>firstFrame</code>, <code>lastFrame</code>, <code>references</code>, <code>referenceFidelity</code>, <code>camera</code>, <code>style</code>, <code>privacy</code>, <code>encryption</code>, <code>progress</code>, and <code>output</code>. Unknown keys return <code>UNKNOWN_FIELD</code>, so <code>generatonMode</code> cannot silently fall back to Quality. The legacy width/height parser and raw-workflow route remain permissive during cutover.

## Camera object

| Field | Exact accepted values |
| --- | --- |
| <code>shot</code> | <code>extreme_wide</code>, <code>wide</code>, <code>medium_wide</code>, <code>medium</code>, <code>medium_close</code>, <code>close</code>, <code>extreme_close</code> |
| <code>movement</code> | <code>static</code>, <code>pan</code>, <code>tilt</code>, <code>dolly</code>, <code>truck</code>, <code>orbit</code>, <code>crane</code>, <code>handheld</code>, <code>zoom</code>, <code>push_in</code>, <code>pull_out</code> |
| <code>strength</code> | <code>subtle</code>, <code>moderate</code>, <code>strong</code> |
| <code>speed</code> | <code>slow</code>, <code>medium</code>, <code>fast</code> |

<code>strength</code> and <code>speed</code> only qualify a recognized movement. The RunPod prompt compiler owns the resulting prose.

## Style object

| Field | Exact accepted values |
| --- | --- |
| <code>visual</code> | <code>cinematic</code>, <code>documentary</code>, <code>animation</code>, <code>anime</code>, <code>photoreal</code>, <code>vintage</code>, <code>noir</code> |
| <code>lighting</code> | <code>natural</code>, <code>golden_hour</code>, <code>high_key</code>, <code>low_key</code>, <code>neon</code>, <code>candlelit</code>, <code>overcast</code> |
| <code>motion</code> | <code>natural</code>, <code>slow_motion</code>, <code>timelapse</code>, <code>hyperlapse</code> |

# 6. Animate requests

## First-frame only

    {
      "input": {
        "mode": "animate",
        "prompt": "The subject turns toward camera as wind moves through the scene.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "generationMode": "quality",
        "firstFrame": {
          "type": "image",
          "role": "general",
          "url": "https://<trusted-media-host>/<job-scoped-path>",
          "token": "<job-scoped asset-download token>"
        },
        "privacy": {
          "mode": "standard"
        }
      }
    }

## Last-frame only

    {
      "input": {
        "mode": "animate",
        "prompt": "The camera moves through the scene and settles on the supplied final composition.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "generationMode": "quality",
        "lastFrame": {
          "type": "image",
          "role": "general",
          "url": "https://<trusted-media-host>/<job-scoped-path>",
          "token": "<job-scoped asset-download token>"
        },
        "privacy": {
          "mode": "standard"
        }
      }
    }

## First + last

    {
      "input": {
        "mode": "animate",
        "prompt": "A continuous natural transition connects the supplied opening and closing frames.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "generationMode": "quality",
        "firstFrame": {
          "type": "image",
          "role": "general",
          "url": "https://<trusted-media-host>/<job-scoped-first-path>",
          "token": "<job-scoped asset-download token>"
        },
        "lastFrame": {
          "type": "image",
          "role": "general",
          "url": "https://<trusted-media-host>/<job-scoped-last-path>",
          "token": "<job-scoped asset-download token>"
        },
        "privacy": {
          "mode": "standard"
        }
      }
    }

Each keyframe is the same reference object accepted for an image in <code>references</code> mode:

| Exact field | Type | Required? | Meaning |
| --- | --- | ---: | --- |
| <code>type</code> | string | no for keyframe | Defaults to <code>image</code>; canonical downstream should send <code>image</code>. |
| <code>role</code> | string | no | Defaults to <code>general</code>. Keyframe roles are not used for Ref2VA prompt clauses. |
| <code>id</code> | string/opaque | no | Control-plane handle only. It is not a byte source. |
| <code>url</code> | string | exactly one of URL/data | HTTPS source fetched by RunPod. |
| <code>data</code> | string | exactly one of URL/data | Standard base64 bytes, optionally a base64 data URI. |
| <code>dataBase64</code> | string | no | Accepted alias for <code>data</code>. Canonical downstream should emit <code>data</code>. |
| <code>token</code> | string | only for protected URL | Sent as <code>Authorization: Bearer ...</code> on the first fetch hop only. |

For example, an inline keyframe is:

    "firstFrame": {
      "type": "image",
      "role": "general",
      "data": "<standard-base64 PNG, JPEG, or WebP bytes>"
    }

Actual image requirements:

- formats: PNG, JPEG, or WebP, verified by magic bytes and Pillow decode;
- maximum bytes: <code>H3_MAX_IMAGE_BYTES</code>, default 32 MiB;
- maximum decoded pixels: <code>H3_MAX_IMAGE_PIXELS</code>, default 64,000,000;
- URL scheme: HTTPS in production;
- every redirect destination is SSRF-validated, with at most three redirects;
- the bearer token and Cloudflare Access headers are sent only on the first hop and are never forwarded to a redirect target;
- RunPod stages bytes under a generated filename inside a per-job ComfyUI input directory and deletes them on success, validation failure, timeout, or exception.

Do not use the legacy raw-graph <code>assets.first_frame.node_id</code>/<code>assets.last_frame.node_id</code> map on the canonical path. The rebuilt workflow builder creates and wires the loader nodes itself.

# 7. References request

## Exact mixed-reference request

    {
      "input": {
        "mode": "references",
        "prompt": "The character crosses the market with the supplied movement and ambient rhythm.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 10,
        "generationMode": "turboFast",
        "referenceFidelity": "high",
        "references": [
          {
            "type": "image",
            "role": "character",
            "id": "character-1",
            "url": "https://<trusted-media-host>/<job-scoped-image-path>",
            "token": "<job-scoped asset-download token>"
          },
          {
            "type": "video",
            "role": "motion",
            "id": "motion-1",
            "url": "https://<trusted-media-host>/<job-scoped-video-path>",
            "token": "<job-scoped asset-download token>",
            "soundtrack": {
              "type": "audio",
              "role": "ambience",
              "id": "motion-1-soundtrack",
              "url": "https://<trusted-media-host>/<job-scoped-soundtrack-path>",
              "token": "<job-scoped asset-download token>"
            }
          },
          {
            "type": "audio",
            "role": "rhythm",
            "id": "rhythm-1",
            "url": "https://<trusted-media-host>/<job-scoped-audio-path>",
            "token": "<job-scoped asset-download token>"
          }
        ],
        "privacy": {
          "mode": "standard"
        }
      }
    }

Cloudflare must add <code>progress</code> and <code>output</code> as in section 3.

## Reference object schema

| Exact field | JSON type | Required? | Allowed values/meaning |
| --- | --- | ---: | --- |
| <code>type</code> | string | yes in <code>references[]</code> | <code>image</code>, <code>video</code>, or <code>audio</code>; trimmed/lowercased. |
| <code>role</code> | string | no | Defaults to <code>general</code>; exact per-type enums are in section 10. |
| <code>id</code> | any opaque value in current parser | no | Carried as an opaque control-plane handle. It is not returned, logged as a filename, or accepted as the location of bytes. Downstream should constrain it to a safe opaque string. |
| <code>url</code> | string | exactly one of URL/data | HTTPS source from which RunPod obtains plaintext media. |
| <code>data</code> | string | exactly one of URL/data | Standard base64 bytes, optionally with a base64 <code>data:</code> URI prefix. |
| <code>dataBase64</code> | string | no | Alias for <code>data</code>; do not emit both. Canonical downstream should emit <code>data</code>. |
| <code>token</code> | string | only for a protected URL | First-hop bearer token for the URL. It is not returned or logged. |
| <code>soundtrack</code> | reference object | video only, optional | A separate audio file paired with that video. Use <code>type: "audio"</code> and an audio role. It counts as a separate reference file. |

The handler requires exactly one actual source, <code>url</code> or inline <code>data</code>. An <code>id</code> by itself fails with <code>INVALID_REFERENCE_TYPE</code> and details <code>{"accepted":["url","data"]}</code>.

## Exact Remix request

<code>remix</code> uses the same array and the Ref2VA family. The source clip is a video reference with role <code>source</code>:

    {
      "input": {
        "mode": "remix",
        "prompt": "Regenerate the source footage in the supplied visual style while preserving its broad action.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 10,
        "generationMode": "quality",
        "referenceFidelity": "standard",
        "references": [
          {
            "type": "video",
            "role": "source",
            "url": "https://<trusted-media-host>/<job-scoped-source-video-path>",
            "token": "<job-scoped asset-download token>"
          },
          {
            "type": "image",
            "role": "style",
            "url": "https://<trusted-media-host>/<job-scoped-style-image-path>",
            "token": "<job-scoped asset-download token>"
          }
        ],
        "privacy": {
          "mode": "standard"
        }
      }
    }

This is reference-guided regeneration, not deterministic video editing. The rebuilt source says unreferenced detail will change.

RunPod now enforces the product meaning: <code>remix</code> is rejected unless at least one member is exactly a video reference with <code>role: "source"</code>. Image-only, audio-only, or motion-video-only sets return a structured <code>INVALID_REFERENCE_COUNT</code> error. This remains reference-guided regeneration, not deterministic editing.

RunPod also enforces soundtrack placement and type: <code>soundtrack</code> is valid only on a video, must be an object that explicitly declares <code>type: "audio"</code>, must use an audio role, and cannot contain another soundtrack.

# 8. Reference media transport

The rebuilt RunPod worker implements two byte transports for every canonical reference:

1. <code>url</code> plus optional <code>token</code>: RunPod performs an HTTPS GET and receives plaintext bytes.
2. <code>data</code>/<code>dataBase64</code>: base64 is carried inside the RunPod job input and decoded in the handler.

There is no implemented canonical transport based on an R2 object key, multipart data sent directly to RunPod, a local path chosen by downstream, or an opaque <code>id</code> lookup.

## Image

Accepted transport:

    {
      "type": "image",
      "role": "character",
      "url": "https://...",
      "token": "..."
    }

or:

    {
      "type": "image",
      "role": "character",
      "data": "<base64 bytes>"
    }

Validation: PNG/JPEG/WebP, 32 MiB default byte limit, 64,000,000 decoded-pixel default.

The current Cloudflare asset route can ingest only images:

    POST /jobs/{cloudflareJobId}/assets?id={assetId}
    Content-Type: image/png | image/jpeg | image/webp
    <raw body>

It returns:

    {
      "asset": {
        "id": "<assetId>",
        "key": "inputs/<cloudflareJobId>/<assetId>.<png|jpg|webp>",
        "contentType": "<submitted image content type>",
        "size": 123
      }
    }

The corresponding RunPod-only read route is:

    GET /internal/jobs/{cloudflareJobId}/assets/{assetId}
    Authorization: Bearer <asset-download token>

The public upload currently buffers and stores plaintext in R2. Its limit is exactly <code>32 * 1024 * 1024</code> bytes.

## Video

Canonical RunPod transport is the same URL/token or inline-data shape.

Implemented validation:

- maximum bytes: 256 MiB by default;
- containers detected by magic bytes: MP4, WebM, MOV;
- codecs: <code>h264</code>, <code>hevc</code>, <code>vp9</code>, <code>av1</code>, <code>mpeg4</code>, <code>vp8</code>;
- at most four total streams;
- maximum dimensions 4096x4096;
- duration 2.0–15.0 seconds;
- <code>ffprobe</code> runs without a shell and with a 20-second default timeout.

**DOWNSTREAM WORK REQUIRED:** the current Cloudflare upload/read implementation has no video content types and a 32 MiB limit. A canonical video URL can point at another HTTPS origin today, but the existing job-scoped R2 asset path cannot transport the advertised 256 MiB video contract.

## Standalone audio

Canonical RunPod transport is the same URL/token or inline-data shape.

Implemented validation:

- maximum bytes: 64 MiB by default;
- containers detected by magic bytes: WAV, MP3, FLAC, OGG, M4A;
- codecs: <code>aac</code>, <code>mp3</code>, <code>opus</code>, <code>vorbis</code>, <code>flac</code>, <code>pcm_s16le</code>, <code>pcm_f32le</code>;
- maximum eight channels;
- maximum duration 15.0 seconds;
- <code>ffprobe</code> timeout 20 seconds by default.

**DOWNSTREAM WORK REQUIRED:** the current Cloudflare asset route accepts no audio content types and cannot serve the 64 MiB contract.

## Video soundtrack

A soundtrack is not extracted implicitly from the supplied video. It is a separate nested audio reference:

    {
      "type": "video",
      "role": "motion",
      "url": "https://.../motion.mp4",
      "token": "...",
      "soundtrack": {
        "type": "audio",
        "role": "ambience",
        "url": "https://.../ambience.m4a",
        "token": "..."
      }
    }

It is staged as the node input paired with that same video index and consumes one audio ordinal and one file from the total product limit.

The nested soundtrack travels through the same audio byte limit, Accept header, container/codec/channel probe, and 15-second measured-duration validation as standalone audio. A soundtrack longer than 15 seconds is rejected before model loading.

## Remix source video

The source video is not a special top-level byte field. It is a normal member of <code>references</code>:

    {
      "type": "video",
      "role": "source",
      "url": "https://...",
      "token": "..."
    }

RunPod performs Ref2VA routing and compiles the source-footage prompt clause. Downstream must not turn the source into <code>&lt;Video 1&gt;</code> text.

## URL fetch security and credential behavior

For all reference URL types:

- HTTPS is required unless the test-only <code>H3_ALLOW_INSECURE_IMAGE_URL=1</code> is set.
- DNS resolution must produce public addresses; loopback, private, link-local, reserved, multicast, and unspecified addresses are rejected.
- redirects are followed manually and every destination is revalidated;
- maximum redirects: three;
- <code>Authorization: Bearer {token}</code> and Cloudflare Access service-token headers are sent only on hop zero;
- image Accept header: <code>image/*</code>;
- video Accept header: <code>video/*, application/octet-stream</code>;
- audio Accept header: <code>audio/*, application/octet-stream</code>;
- byte ceilings are enforced against both <code>Content-Length</code> and streamed bytes.

## Storage/staging behavior

RunPod creates <code>{COMFY_INPUT_DIR}/job-{sanitizedGenerationId}</code>, generates filenames, stages plaintext only for inference, and removes the entire directory on every exit path. User filenames, IDs, and URL path components never become local filenames.

API Runtime and Cloudflare must not write local filesystem paths into the request. Only RunPod knows its ComfyUI input tree.

# 9. Confidential reference transport

## What is implemented now

Confidential Generation v2 protects the generated output:

- the client supplies only an encryption public key;
- RunPod generates a fresh AES-256 file key after inference;
- RunPod encrypts the output inside the inference environment;
- RunPod wraps the file key using RSA-OAEP-256;
- Cloudflare refuses plaintext output for a job whose signed token says <code>confidential</code>;
- R2 persistently stores <code>artifact.enc</code>, not the MP4;
- the private key and passphrase stay in the client.

For reference inputs, the rebuilt worker presently requires plaintext at inference time through <code>url</code> or inline base64. It provides strong transient cleanup inside the RunPod container, but it does not implement client-encrypted reference containers or reference decryption.

The current Cloudflare <code>POST /jobs/:jobId/assets</code> path persistently writes plaintext input images under <code>inputs/</code>. It has no Confidential-specific behavior and no consume-once deletion. It is therefore not sufficient for a claim that persistent Confidential reference media is ciphertext-only.

## Required trust boundary

Plaintext may exist transiently where the model technically requires it: in transport buffers and the per-job RunPod staging directory. It must not be added to persistent:

- API Runtime storage;
- Cloudflare Worker storage;
- R2 objects;
- RunPod job input records;
- logs;
- job/status documents;
- progress callbacks.

The RunPod input for a Confidential reference should still be the existing exact interface:

    {
      "type": "image|video|audio",
      "role": "<valid role>",
      "url": "https://<job-scoped plaintext delivery endpoint>",
      "token": "<single-job/asset credential>"
    }

The URL endpoint must deliver plaintext over TLS only when RunPod fetches it, without creating a durable plaintext copy. It should be job-scoped, short-lived, and preferably single-consumption, with deletion/expiry on success, failure, cancellation, and timeout.

**DOWNSTREAM WORK REQUIRED:** no such durable-storage-free relay is implemented in this repository. The downstream agent must either connect an existing ephemeral media service or add one. If that requires a RunPod reference-decryption change, it is a separate worker contract change and must be designed explicitly.

## Transport choices and their privacy consequences

| Choice | Works with current RunPod handler? | Confidential consequence |
| --- | ---: | --- |
| Inline <code>data</code> | yes | Plaintext becomes part of the RunPod job input/queue record. Do not use for Confidential references. |
| Current Cloudflare R2 input route | images only | Persistent plaintext in R2. Do not describe this as ciphertext-only Confidential reference handling. |
| Direct third-party HTTPS URL | yes | Plaintext may persist at that third party. Acceptable only if the user understands that origin is inside their chosen trust boundary. |
| Short-lived non-persistent HTTPS relay | yes | Preferred with the current handler; must prove it does not durably retain bodies or tokens. |
| Client-encrypted reference in R2 | no | Current handler cannot decrypt it. Adding decryption requires a new trust/key design and must not send the user's private output-decryption capability. |

If temporary plaintext R2 storage is used as an interim migration, state the privacy downgrade explicitly. At minimum it needs immediate deletion after the first successful RunPod fetch, deletion on cancel/failure, and an R2 lifecycle backstop. Those controls reduce exposure but do not make the period before deletion ciphertext-only.

Reference tokens and URLs must be structurally redacted from logs. The rebuilt handler never returns them. Downstream telemetry must follow the same rule.

## Exact Confidential request fields

The canonical RunPod fields are:

    {
      "privacy": {
        "mode": "confidential"
      },
      "encryption": {
        "version": 2,
        "algorithm": "AES-256-GCM",
        "keyWrapAlgorithm": "RSA-OAEP-256",
        "publicKeyAlgorithm": "RSA-OAEP-256",
        "publicKey": "<base64url DER SubjectPublicKeyInfo>",
        "keyId": "<optional derived key id>"
      },
      "output": {
        "url": "https://<worker-host>/internal/jobs/<jobId>/output",
        "token": "<output-upload token signed with pm=confidential and cv=2>",
        "jobId": "<jobId>"
      }
    }

Exact validation:

| Field | Rule |
| --- | --- |
| <code>privacy.mode</code> | <code>confidential</code> |
| <code>encryption.version</code> | optional default 2; only 2 may be written |
| <code>encryption.algorithm</code> | optional default <code>AES-256-GCM</code>; only that value |
| <code>encryption.keyWrapAlgorithm</code> | optional default <code>RSA-OAEP-256</code>; alias <code>key_wrap_algorithm</code> accepted |
| <code>encryption.publicKeyAlgorithm</code> | Cloudflare public validation accepts it and requires it to equal the wrap algorithm; canonical Worker forwards it. Python uses the decoded RSA key/wrap algorithm. |
| <code>encryption.publicKey</code> | required; base64url DER SPKI RSA public key; alias <code>public_key</code> accepted |
| RSA modulus | 3072–8192 bits in the authoritative Python decoder |
| <code>encryption.keyId</code> | optional; if supplied it must equal the first 16 bytes of SHA-256(SPKI), base64url without padding; alias <code>key_id</code> accepted |
| <code>output.url</code> | required for Confidential; the handler refuses inline output |

The current public Cloudflare adapter also accepts top-level <code>privacyMode</code> or <code>privacy_mode</code>, then converts it to <code>privacy: {"mode": ...}</code>. Canonical downstream should use <code>privacy.mode</code> internally and retain <code>privacyMode</code> only as a public compatibility alias.

The current public Cloudflare request supports one of:

- <code>retentionSeconds</code>, integer 60–7,776,000;
- <code>expiresAt</code>, ISO-8601 timestamp 60–7,776,000 seconds in the future.

It records that expiry in the signed output token/status metadata. The source explicitly calls it advisory: there is no per-object scheduled deletion enforcer yet.

Fields forbidden at any depth of the public request, after case/separator normalization:

    passphrase
    password
    privatekey
    privateencryptionkey
    encryptedprivatekey
    keyencryptionkey
    kek
    fileencryptionkey
    fek
    decryptionkey
    derivedkey
    aeskey
    symmetrickey
    secretkey

Confidential v1 and any <code>encryption.key</code> are refused for new generation. Existing v1 artifacts remain readable.

The encrypted container begins with <code>CGEN</code> and has:

- version 2;
- suite <code>AES-256-GCM</code>;
- 12-byte nonce;
- 16-byte tag;
- maximum JSON header 8192 bytes.

Authenticated header fields are exactly:

    {
      "v": 2,
      "alg": "AES-256-GCM",
      "artifactId": "<cloudflareJobId>",
      "contentType": "video/mp4",
      "plaintextBytes": 123,
      "privacyMode": "confidential",
      "createdAt": "<UTC ISO-8601>",
      "kw": {
        "alg": "RSA-OAEP-256",
        "wrappedFileKey": "<base64url>",
        "keyId": "<derived key id>"
      }
    }

Cloudflare's output endpoint verifies the signed privacy/version claims, parses this header before storage, and checks <code>artifactId</code> matches the route job ID.

# 10. Reference role enums and prompt effects

## Image roles

    character
    identity
    face
    clothing
    product
    object
    environment
    style
    pose
    composition
    general

## Video roles

    source
    motion
    body_performance
    camera_motion
    scene_structure
    timing
    action
    visual_style
    general

## Audio roles

    voice
    dialogue
    music
    rhythm
    ambience
    sound_effect
    general

Roles compile to exact guidance concepts:

| Type/role | Compiled role phrase |
| --- | --- |
| image <code>character</code>, <code>identity</code> | <code>the character</code> |
| image <code>face</code> | <code>the face</code> |
| image <code>clothing</code> | <code>the clothing</code> |
| image <code>product</code> | <code>the product</code> |
| image <code>object</code> | <code>the object</code> |
| image <code>environment</code> | <code>the setting</code> |
| image <code>style</code> | <code>the visual style</code> |
| image <code>pose</code> | <code>the pose</code> |
| image <code>composition</code> | <code>the composition</code> |
| video <code>source</code> | <code>the source footage</code> |
| video <code>motion</code> | <code>the motion</code> |
| video <code>body_performance</code> | <code>the body performance</code> |
| video <code>camera_motion</code> | <code>the camera movement</code> |
| video <code>scene_structure</code> | <code>the scene structure</code> |
| video <code>timing</code> | <code>the timing</code> |
| video <code>action</code> | <code>the action</code> |
| video <code>visual_style</code> | <code>the visual style</code> |
| audio <code>voice</code> | <code>the voice</code> |
| audio <code>dialogue</code> | <code>the dialogue</code> |
| audio <code>music</code> | <code>the music</code> |
| audio <code>rhythm</code> | <code>the rhythm</code> |
| audio <code>ambience</code> | <code>the ambience</code> |
| audio <code>sound_effect</code> | <code>the sound effect</code> |

<code>general</code> adds no role clause but the media still conditions the model.

RunPod assigns tags and compiles clauses such as:

    Use the character from <Picture 1>.
    Use the motion from <Video 1>.
    Use the ambience from <Audio 1>.

Downstream must never emit those strings.

Ordinal rules are exact:

1. Images are numbered <code>Picture 1..n</code>.
2. Videos are numbered <code>Video 1..n</code>.
3. Audio has one counter. Each video soundtrack consumes the next <code>Audio</code> ordinal during video ordering; standalone audio continues from that counter.
4. A soundtrack on video 1 followed by one standalone audio yields <code>Audio 1</code> for the soundtrack and <code>Audio 2</code> for the standalone clip.

The prompt compiler neutralizes any user text shaped like an H3 reference tag, so caller prose cannot impersonate structural references.

# 11. Reference limits

Exact constants:

    PRODUCT_MAX_REFERENCES = 12
    MODEL_MAX_REFERENCES = 18
    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_VIDEO_SOUNDTRACKS = 3
    MAX_STANDALONE_AUDIO = 3

Semantics:

| Limit | Value | Counting rule |
| --- | ---: | --- |
| Images | 9 | Each image is one file. |
| Videos | 3 | Each video is one file. |
| Video soundtracks | 3 | At most one paired soundtrack per video; each is a separate file and audio ordinal. |
| Standalone audio | 3 | Separate from soundtracks. |
| Product total | 12 | Images + videos + video soundtracks + standalone audio. This is what frontend/API Runtime must enforce. |
| Underlying node/model total | 18 | 9 + 3 + 3 + 3. This is not the product allowance. |

The theoretical per-slot maximum is 18 files, but <code>PRODUCT_MAX_REFERENCES</code> intentionally rejects more than 12. RunPod revalidates counts and must remain authoritative even when frontend/API Runtime enforce the same product limit.

Duration limits:

| Media | Per-file | Aggregate implementation |
| --- | --- | --- |
| Reference video | 2.0–15.0 seconds | Up to 45 seconds across three video references |
| Standalone audio | greater than 0 as probed, maximum 15.0 seconds | Up to 45 seconds across three standalone audio references |
| Video soundtrack | greater than 0 as probed, maximum 15.0 seconds | Measured and enforced per nested soundtrack before model loading |

The product total counts files, not seconds or logical reference objects.

# 12. Duration limits

Exact constants and arithmetic:

    FPS = 24
    MIN_FRAMES = 5
    MAX_FRAMES = 3600
    FRAME_GRID_MODULUS = 17
    FRAME_GRID_REMAINDER = 5
    TRAINED_MIN_FRAMES = 124
    TRAINED_MAX_FRAMES = 362

The frontend/API Runtime sends <code>duration</code> in seconds. RunPod performs:

    requested_frames = max(5, round(duration * 24))
    frames = smallest integer >= requested_frames where frames % 17 == 5
    reject if frames > 3600
    actual durationSeconds = frames / 24

Examples:

| Requested seconds | Rounded frames before alignment | Actual frames | Returned <code>durationSeconds</code> |
| ---: | ---: | ---: | ---: |
| 5 | 120 | 124 | 5.1667 |
| 10 | 240 | 243 | 10.125 |
| 15 | 360 | 362 | 15.0833 |

The returned <code>generation</code> object includes <code>durationAdjusted: true</code> whenever aligned frames differ from the pre-alignment count. It includes <code>outsideTrainedRange: true</code> outside 124–362 frames.

There are three different "minimum/maximum" facts:

- Request validation currently accepts any numeric <code>duration &gt; 0</code>. Values below 5/24 seconds are clamped to five frames.
- Capabilities report <code>minSeconds: 0.2083</code> (5/24) and <code>maxSeconds: 149.6667</code> (3592/24, rounded to four decimal places).
- The maximum is derived from <code>MAX_FRAMES</code> and the same 17k+5 grid constants used by request validation. A request using the advertised value resolves to 3592 frames and is accepted. A request for exactly 150 seconds rounds to 3600 and aligns upward to 3609, so it is rejected.

Downstream should advertise the returned capability value rather than deriving 3600/24 independently. RunPod remains authoritative for the final aligned frame count.

# 13. Aspect ratios

Exact accepted semantic values:

    16:9
    9:16
    1:1
    4:3
    3:4
    21:9

Exact resolved dimensions:

| Quality | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 21:9 |
| --- | --- | --- | --- | --- | --- | --- |
| <code>draft</code> | 512x288 | 288x512 | 384x384 | 448x320 | 320x448 | 576x256 |
| <code>standard</code> | 1024x576 | 576x1024 | 768x768 | 896x672 | 672x896 | 1152x512 |
| <code>hd</code> | 1248x704 | 704x1248 | 768x768 | 1024x768 | 768x1024 | 1440x640 |
| <code>ultra</code> | 1344x768 | 768x1344 | 768x768 | 1024x768 | 768x1024 | 1536x672 |

The frontend/API Runtime sends only <code>aspectRatio</code>. RunPod resolves 32-pixel alignment and the 1,032,192-pixel model area ceiling and returns authoritative <code>width</code>/<code>height</code>.

# 14. Quality tiers

Exact canonical names:

    draft
    standard
    hd
    ultra

All canonical quality tiers use 20 base steps unless <code>generationMode</code> selects a Turbo configuration. A quality tier is canvas semantics; generation speed is a separate field.

At 16:9:

| Canonical quality | Dimensions | Base steps |
| --- | --- | ---: |
| <code>draft</code> | 512x288 | 20 |
| <code>standard</code> | 1024x576 | 20 |
| <code>hd</code> | 1248x704 | 20 |
| <code>ultra</code> | 1344x768 | 20 |

## Naming collision with the legacy Worker

The current Cloudflare quality table is:

| Legacy quality | 16:9 | Steps |
| --- | --- | ---: |
| <code>fast</code> | 1024x576 | 14 |
| <code>standard</code> | 1024x576 | 20 |
| <code>hd</code> | 1344x768 | 20 |

Therefore:

- legacy <code>standard</code> matches canonical <code>standard</code> at 16:9;
- legacy <code>hd</code> has the pixel meaning now named canonical <code>ultra</code>;
- canonical <code>hd</code> is a new 1248x704 tier;
- legacy <code>fast</code> is not a quality tier in the rebuilt API and is not a Turbo mode.

Migration recommendation: make <code>draft</code>, <code>standard</code>, <code>hd</code>, and <code>ultra</code> the only names in the new frontend. In the compatibility adapter, map old <code>hd</code> to canonical <code>ultra</code> for pixel parity and old <code>standard</code> to canonical <code>standard</code>. Keep old <code>fast</code> explicitly legacy: the RunPod legacy adapter now preserves its 14-step base-model behavior, but it is still not a canonical quality or Turbo value.

# 15. Generation speed / Turbo

Exact request field:

    "generationMode": "quality|turbo|turboFast"

Exact mappings:

| Model family | Wire value | Actual steps | Returned <code>acceleration</code> | Availability/note |
| --- | --- | ---: | --- | --- |
| FL2VA | <code>quality</code> | 20 | <code>none</code> | Base checkpoint, no LoRA |
| FL2VA | <code>turbo</code> | 8 | <code>turbo_lora</code> | FL2VA 8-step distilled LoRA |
| FL2VA | <code>turboFast</code> | 4 | <code>turbo_lora</code> | FL2VA 4-step v1.0, distilled for the 768p canvas; behavior at smaller tiers is unvalidated |
| Ref2VA | <code>quality</code> | 20 | <code>none</code> | Base checkpoint, no LoRA |
| Ref2VA | <code>turboFast</code> | 4 | <code>turbo_lora</code> | Ref2VA 4-step v0.1 early release |
| Ref2VA | <code>turbo</code> | n/a | n/a | Unsupported: this build has no 8-step Ref2VA LoRA |

Mode-to-family routing:

| Product mode | Family |
| --- | --- |
| <code>create</code> | <code>fl2va</code> |
| <code>animate</code> | <code>fl2va</code> |
| <code>references</code> | <code>ref2va</code> |
| <code>remix</code> | <code>ref2va</code> |

The parser normalizes <code>turbo_fast</code> and <code>turbo-fast</code>, but the canonical response/request spelling is <code>turboFast</code>.

Returned generation metadata includes:

    {
      "generationMode": "quality|turbo|turboFast",
      "steps": 20,
      "acceleration": "none|turbo_lora",
      "accelerationNote": "<present only for a caveated configuration>"
    }

LoRA and checkpoint filenames are deliberately not returned. Downstream must branch on <code>byFamily</code>/<code>turbo</code> capabilities and semantic <code>generationMode</code>, never on a weight filename.

# 16. Legacy <code>steps</code> and <code>fast</code>

The current public Worker does not implement a Boolean request field named <code>fast</code>. Its fast mode is exactly:

    "quality": "fast"

That legacy preset means:

- 576-pixel short-edge canvas, producing 1024x576 at 16:9;
- 14 sampling steps;
- the same FL2VA base checkpoint as legacy <code>standard</code>;
- no Turbo LoRA;
- sampler <code>res_multistep</code>;
- scheduler <code>simple</code>;
- denoise <code>1.0</code>;
- native audio decode and <code>CreateVideo</code> remain in the graph;
- no other resolution/model/sampler behavior changes.

The legacy <code>steps</code> field, when present, overrides the preset and is accepted from 1 through 100 by the current Cloudflare Worker. The Worker writes that number directly into <code>BasicScheduler.inputs.steps</code>.

The legacy <code>audio</code> field defaults to true and is recorded in normalized settings, but the current graph always creates and saves native audio regardless of its value. It is not a reliable "disable audio" control.

Do not map legacy <code>quality: "fast"</code> or <code>steps: 14</code> to <code>generationMode: "turbo"</code> or <code>"turboFast"</code>. Turbo changes the model with a distilled LoRA; legacy fast is a reduced-step run on base weights.

## Rebuilt legacy-parser behavior

The RunPod legacy adapter now preserves every explicit legal <code>steps</code> value from 1 through 100. The value is written verbatim to <code>BasicScheduler.inputs.steps</code> and returned as <code>generation.steps</code>. It always uses the base FL2VA checkpoint with <code>generationMode: "quality"</code>, <code>acceleration: "none"</code>, and no LoRA. Regression coverage pins exact scheduler values for 4, 8, 14, 20, and 30 steps.

Raw workflows remain supported during cutover. Unsupported or lossy legacy semantics may stay on the existing Cloudflare graph route temporarily, but explicit legacy sampling depth is no longer itself a reason to do so. Do not map an arbitrary step count to canonical Turbo.

# 17. Legacy generation request and compatibility behavior

## Exact current public request surface

The current Worker accepts a normalized request:

    {
      "backend": "h3-blackwell",
      "mode": "text_to_video",
      "prompt": "A red fox runs through wet ferns.",
      "quality": "standard",
      "duration": 5,
      "aspect_ratio": "16:9",
      "seed": 51
    }

It also accepts raw geometry:

    {
      "backend": "h3-blackwell",
      "prompt": "A red fox runs through wet ferns.",
      "width": 1024,
      "height": 576,
      "frames": 124,
      "steps": 20,
      "seed": 51
    }

Exact fields and behavior:

| Field | Current behavior |
| --- | --- |
| <code>backend</code> | Selects the current backend map/aliases. |
| <code>model</code> | Backend-selection alias; if both are present and disagree, request is rejected. |
| <code>mode</code> | <code>text_to_video</code>, <code>first_frame_to_video</code>, <code>last_frame_to_video</code>, <code>first_last_frame_to_video</code>; <code>reference</code> and <code>regenerate_2k</code> exist but return 501. |
| <code>prompt</code> | Required non-empty string after string conversion/trim. |
| <code>quality</code> | <code>fast</code>, <code>standard</code>, <code>hd</code>; default <code>standard</code>. |
| <code>aspect_ratio</code> | <code>16:9</code>, <code>9:16</code>, <code>1:1</code>, <code>4:3</code>, <code>3:4</code>; default <code>16:9</code>. |
| <code>width</code>, <code>height</code> | Must be supplied together, positive integer multiples of 32, area at most 1,032,192 pixels. They override preset dimensions. If <code>aspect_ratio</code> is also supplied and materially contradicts them, request is rejected. |
| <code>duration</code> | Positive seconds converted to aligned frames. |
| <code>frames</code> | Explicit legal value 5–3600 satisfying <code>frames % 17 == 5</code>; overrides duration derivation. Contradiction with an explicit <code>duration</code> is rejected. |
| <code>steps</code> | Explicit integer 1–100; overrides quality preset steps. |
| <code>seed</code> | Non-negative; otherwise randomly chosen below 2,147,483,647. |
| <code>audio</code> | Defaults true; recorded but does not remove native audio nodes. |
| <code>first_frame</code>, <code>last_frame</code> | Asset object described below. Their presence can infer the legacy mode when <code>mode</code> is omitted. |
| <code>reference_images</code>, <code>reference_videos</code>, <code>reference_audio</code> | A non-empty array infers legacy <code>reference</code>, which currently returns 501 rather than transporting it. |
| <code>camera</code>, <code>shot</code>, <code>lighting</code>, <code>style</code>, <code>motion</code>, <code>audio_prompt</code> | Free-form legacy values appended by <code>buildPrompt()</code>. These are not the new enumerated <code>camera</code>/<code>style</code> objects. |
| <code>privacyMode</code>, <code>privacy_mode</code> | Public privacy mode; default <code>standard</code>. |
| <code>encryption</code> | Confidential v2 public-key block. |
| <code>retentionSeconds</code>, <code>expiresAt</code> | Mutually exclusive advisory retention fields. |

Each old keyframe object must choose exactly one:

    {"url": "https://..."}
    {"base64": "<base64 image bytes>"}
    {"asset_id": "<asset id previously uploaded for this job>"}
    {"r2_key": "inputs/<jobId>/<assetId>.<extension>"}

<code>r2_key</code> is validated under <code>inputs/</code> and reduced to an asset ID; RunPod receives a job-scoped URL, never the raw R2 key.

## Where the old Worker builds the graph

In <code>generateVideo()</code>, the Worker currently performs:

    normalizeRequest(body)
      -> buildWorkflowForSettings(settings)
      -> loadWorkflowTemplate(settings.mode)
      -> baseFl2vaTemplate()
      -> applySettings(template, settings)
      -> buildRunPodInput(..., workflow, ...)

<code>applySettings()</code> writes exact prompt, width, height, length, seed, steps, and fps into ComfyUI nodes. <code>buildRunPodInput()</code> starts with:

    const input = { workflow };

It then adds:

    privacy
    encryption
    progress
    output
    assets

For old keyframes, <code>assets</code> contains <code>url</code> or <code>base64</code>, optional <code>token</code>, and a graph-specific <code>node_id</code> of <code>first_frame_image</code> or <code>last_frame_image</code>.

That <code>workflow</code> field is precisely why the rebuilt handler's canonical parser never sees the product request.

## Existing behavior that must remain compatible

- both endpoint selections and backend aliases;
- old normalized and explicit-geometry requests;
- all four implemented FL2VA keyframe combinations;
- old quality names and exact pixel/step behavior;
- explicit legal frames and arbitrary 1–100 base steps;
- legacy prompt additions until a RunPod-side equivalent exists;
- standard and Confidential v2 output behavior;
- Cloudflare job IDs plus RunPod job IDs;
- progress WebSocket/status behavior;
- status and cancellation routes, including legacy default-backend aliases;
- R2 artifact retrieval, deletion, and Confidential client-side decryption.

## Recommended staged adapter

New requests:

    new frontend request
      -> API Runtime authorization/product validation
      -> Cloudflare trusted callback/reference enrichment
      -> canonical RunPod input with no workflow

Legacy requests that map without changing behavior:

| Legacy | Canonical |
| --- | --- |
| <code>text_to_video</code> | <code>mode: "create"</code> |
| first/last/both keyframe modes | <code>mode: "animate"</code> with <code>firstFrame</code>/<code>lastFrame</code> |
| <code>standard</code> | <code>quality: "standard"</code> |
| <code>hd</code> | <code>quality: "ultra"</code> for pixel parity |
| <code>aspect_ratio</code> | <code>aspectRatio</code> |

Requests that cannot yet be mapped losslessly:

- legacy <code>fast</code> or arbitrary <code>steps</code>, because of the RunPod legacy-step defect and because they must not select Turbo;
- raw width/height/frames when exact values do not correspond to a canonical semantic tier/duration;
- free-form legacy <code>camera</code>, <code>shot</code>, <code>lighting</code>, <code>style</code>, <code>motion</code>, and <code>audio_prompt</code> values outside the new enums;
- any behavior relying on the legacy <code>audio</code> flag, which was never actually enforced.

Keep those on the old graph-building compatibility route temporarily. The end state should extend/fix the RunPod-owned legacy adapter and then remove downstream graph construction. A downstream adapter may rename public fields and select semantic modes; it must not rebuild ComfyUI graphs or H3 prompt tags.

# 18. Response contract

There are three envelopes. They must not be collapsed conceptually:

1. RunPod transport fields: queue ID, raw queue status, queue timing, worker ID.
2. Privora handler result: <code>output</code> after the handler returns.
3. Cloudflare/API Runtime product response: Cloudflare job ID, routes, normalized product status, artifact access.

## Accepted/submitted

RunPod asynchronous submission returns at least a RunPod job ID and queue status:

    {
      "id": "<runpodJobId>",
      "status": "IN_QUEUE"
    }

The current Cloudflare <code>POST /generate</code> responds HTTP 202 with this exact field set:

    {
      "backend": "h3-blackwell",
      "id": "<runpodJobId>",
      "jobId": "<cloudflareJobId>",
      "status": "IN_QUEUE",
      "seed": 51,
      "mode": "text_to_video",
      "quality": "standard",
      "aspectRatio": "16:9",
      "privacyMode": "standard",
      "settings": {
        "width": 1024,
        "height": 576,
        "frames": 124,
        "fps": 24,
        "durationSeconds": 5.166666666666667,
        "steps": 20
      },
      "resolvedFrom": {
        "canvas": "preset",
        "frames": "duration",
        "steps": "preset"
      },
      "routes": {
        "status": "/status/h3-blackwell/<runpodJobId>?jobId=<cloudflareJobId>",
        "cancel": "/cancel/h3-blackwell/<runpodJobId>?jobId=<cloudflareJobId>",
        "events": "/ws/jobs/<cloudflareJobId>",
        "video": "/jobs/<cloudflareJobId>/video",
        "deleteVideo": "/jobs/<cloudflareJobId>/video",
        "artifact": "/jobs/<cloudflareJobId>/artifact",
        "deleteGeneration": "/jobs/<cloudflareJobId>"
      }
    }

For Confidential, it additionally includes:

    {
      "privacyMode": "confidential",
      "encryption": {
        "version": 2,
        "algorithm": "AES-256-GCM",
        "keyWrapAlgorithm": "RSA-OAEP-256",
        "keyId": "<derived public-key id>"
      },
      "expiresAt": "<present only when retention intent was supplied>"
    }

The current <code>settings</code> values are computed by Cloudflare because it owns the old graph. The new canonical path must not fabricate those actual values at submission. Preserve the old 202 shape on the legacy route; for canonical jobs, actual <code>width</code>, <code>height</code>, <code>frames</code>, <code>durationSeconds</code>, and <code>steps</code> become authoritative only in <code>output.generation</code>.

If the product requires a seed in the 202 response, it must generate a valid seed, include that exact seed in the canonical RunPod request, and return the same one. If it omits the request seed, only RunPod knows it and it arrives at completion.

## Queued

RunPod transport:

    {
      "id": "<runpodJobId>",
      "status": "IN_QUEUE",
      "delayTime": 123,
      "executionTime": null,
      "workerId": null
    }

The current Cloudflare realtime state begins with:

    {
      "status": "IN_QUEUE",
      "phase": "queued",
      "percent": 0,
      "runpodId": "<runpodJobId>",
      "backend": "h3-blackwell",
      "privacyMode": "standard|confidential"
    }

## Running

RunPod uses <code>IN_PROGRESS</code> and may expose <code>RUNNING</code>. The handler's stable progress phases are:

    starting_worker
    comfy_ready
    loading_models
    sampling
    decoding
    uploading
    completed
    failed
    cancelled

Progress callback body always begins:

    {
      "jobId": "<cloudflareJobId>",
      "phase": "<phase>"
    }

Sampling can add:

    {
      "step": 4,
      "steps": 20,
      "percent": 20
    }

Other exact percentages assigned by the handler are <code>90</code> when decoding begins, <code>95</code> when callback output upload begins, and <code>100</code> on completion. Phase callbacks may omit percent.

The active deployed Worker exposes a status fragment:

    {
      "progress": {
        "phase": "sampling",
        "step": 4,
        "steps": 20,
        "percent": 20,
        "updatedAt": "<ISO-8601 timestamp>"
      }
    }

It is merged only when the public status URL includes the Cloudflare-side <code>?jobId=...</code> query. Phase is sanitized to lower-case <code>[a-z0-9_-]</code>, maximum 40 characters; step/steps are non-negative integers; percent is clamped to 0–100.

## Completed: exact Privora handler result

A standard canonical create job using callback output returns under RunPod's outer <code>output</code>:

    {
      "images": [
        {
          "filename": "<ComfyUI-generated MP4 filename>",
          "subfolder": "<ComfyUI output subfolder>",
          "type": "output",
          "size": 123456,
          "key": "outputs/<cloudflareJobId>/video.mp4",
          "url": "/jobs/<cloudflareJobId>/artifact",
          "contentType": "video/mp4",
          "privacyMode": "standard",
          "encrypted": false
        }
      ],
      "prompt_id": "<ComfyUI prompt UUID>",
      "privacyMode": "standard",
      "generation": {
        "mode": "create",
        "seed": 51,
        "model": "fl2va",
        "width": 1024,
        "height": 576,
        "frames": 124,
        "fps": 24,
        "durationSeconds": 5.1667,
        "quality": "standard",
        "aspectRatio": "16:9",
        "steps": 20,
        "durationAdjusted": true,
        "generationMode": "quality",
        "acceleration": "none"
      },
      "video": {
        "filename": "<same filename>",
        "subfolder": "<same subfolder>",
        "type": "output",
        "size": 123456,
        "key": "outputs/<cloudflareJobId>/video.mp4",
        "url": "/jobs/<cloudflareJobId>/artifact",
        "contentType": "video/mp4",
        "privacyMode": "standard",
        "encrypted": false
      }
    }

The <code>video</code> object is a metadata copy of <code>images[0]</code> with any inline <code>data</code> removed. The target callback path never returns video bytes inside status.

Optional <code>generation</code> additions:

    "outsideTrainedRange": true

    "keyframes": {
      "first": true,
      "last": false
    }

    "references": {
      "images": 1,
      "videos": 1,
      "audio": 1,
      "soundtracks": 1,
      "total": 4,
      "fidelity": "high"
    }

    "schema": "legacy"

Turbo adds:

    "generationMode": "turbo",
    "steps": 8,
    "acceleration": "turbo_lora"

and may add <code>accelerationNote</code>.

## Completed: Confidential artifact metadata

For Confidential v2, each <code>images[]</code> item and the metadata <code>video</code> item include:

    {
      "key": "outputs/<cloudflareJobId>/artifact.enc",
      "url": "/jobs/<cloudflareJobId>/artifact",
      "contentType": "application/octet-stream",
      "privacyMode": "confidential",
      "encrypted": true,
      "artifact": {
        "privacyMode": "confidential",
        "encrypted": true,
        "cryptoVersion": 2,
        "algorithm": "AES-256-GCM",
        "encryptionVersion": 2,
        "keyWrapAlgorithm": "RSA-OAEP-256",
        "keyId": "<derived key id>",
        "contentType": "application/octet-stream",
        "originalContentType": "video/mp4",
        "plaintextBytes": 123456
      }
    }

No private key, passphrase, symmetric file key, wrapped file key, reference token, callback token, prompt, or reference ID is returned.

## Output fields not implemented

The handler does not currently return:

- a separate audio track URL or audio codec/sample-rate metadata;
- response-level inference-stage timings;
- model/LoRA filenames;
- Docker image ID/digest;
- requested prompt or compiled prompt.

Native audio is embedded in the generated video. RunPod outer <code>delayTime</code> and <code>executionTime</code> are queue transport timing fields. Detailed handler timings are logs, not API response fields.

## Failed

Structured Privora validation/execution failure:

    {
      "error": "Human-safe message",
      "errorCode": "INVALID_REFERENCE_COUNT",
      "errorDetails": {
        "limit": 12,
        "supplied": 13,
        "modelLimit": 18
      }
    }

<code>errorDetails</code> is omitted when empty.

Workflow/media/output failure:

    {
      "error": "Human-safe workflow error message"
    }

Unexpected exception:

    {
      "error": "ExceptionType: message"
    }

No-output failure:

    {
      "error": "The workflow completed but produced no saved output. Ensure it ends in SaveVideo (or another Save* node).",
      "prompt_id": "<ComfyUI prompt UUID>"
    }

Critical transport rule: the handler catches these exceptions and returns a dictionary. RunPod can therefore return:

    {
      "status": "COMPLETED",
      "output": {
        "error": "...",
        "errorCode": "..."
      }
    }

API Runtime must classify that as failed.

# 19. Status polling and cancellation

## Current public paths

| Operation | Exact path |
| --- | --- |
| Status with explicit backend | <code>GET /status/{backend}/{runpodJobId}?jobId={cloudflareJobId}</code> |
| Legacy status | <code>GET /status/{runpodJobId}</code>, defaults to <code>h3</code>; <code>backend</code>/<code>model</code> query aliases are accepted |
| Cancel with explicit backend | <code>POST /cancel/{backend}/{runpodJobId}?jobId={cloudflareJobId}</code> |
| Legacy cancel | <code>POST /cancel/{runpodJobId}</code>, defaults to <code>h3</code> |
| Realtime | <code>GET /ws/jobs/{cloudflareJobId}</code> |

Calls to an Access-protected public Worker require:

    CF-Access-Client-Id: ...
    CF-Access-Client-Secret: ...

The frontend should call API Runtime rather than receive those service-token values. Cloudflare uses the selected RunPod API key for its downstream status/cancel operation.

## Exact fields retained from RunPod status

The Worker sanitizer returns:

    {
      "backend": "h3-blackwell",
      "id": "<runpodJobId or null>",
      "status": "<raw RunPod status or null>",
      "delayTime": "<number or null>",
      "executionTime": "<number or null>",
      "workerId": "<workerId, worker_id, or null>",
      "error": "<when supplied>",
      "errors": "<when supplied>",
      "output": "<sanitized handler output when supplied>",
      "message": "<when supplied>",
      "detail": "<when supplied>",
      "code": "<when supplied>",
      "raw": "<when supplied>"
    }

Binary/base64 strings are removed, large strings/collections are bounded, and secret-shaped fields are redacted. With <code>?jobId</code>, status may also add <code>privacyMode</code>, <code>expiresAt</code>, <code>video</code>, <code>artifact</code>, and live <code>progress</code>.

## Raw RunPod statuses and product mapping

Observed/defined RunPod queue states:

    IN_QUEUE
    IN_PROGRESS
    RUNNING
    COMPLETED
    FAILED
    CANCELLED
    TIMED_OUT

Required product translation:

| Raw/inner condition | Product status |
| --- | --- |
| <code>IN_QUEUE</code> | <code>queued</code> |
| <code>IN_PROGRESS</code> or <code>RUNNING</code> | <code>running</code> |
| <code>COMPLETED</code> and no <code>output.error</code> | <code>completed</code> |
| <code>COMPLETED</code> with <code>output.error</code> | <code>failed</code> |
| <code>FAILED</code> | <code>failed</code> |
| <code>CANCELLED</code> | <code>cancelled</code> |
| <code>TIMED_OUT</code> | <code>failed</code> with timeout reason, or a distinct internal <code>timed_out</code> state |

The frontend should consume API Runtime's product status and stable progress phase, not branch directly on raw RunPod states.

## Realtime event shapes

Progress:

    {
      "type": "progress",
      "jobId": "<cloudflareJobId>",
      "phase": "sampling",
      "step": 4,
      "steps": 20,
      "percent": 20
    }

Completed:

    {
      "type": "completed",
      "jobId": "<cloudflareJobId>",
      "video": {
        "url": "/jobs/<cloudflareJobId>/video",
        "key": "outputs/<cloudflareJobId>/video.mp4",
        "deleted": false
      }
    }

Failed:

    {
      "type": "failed",
      "jobId": "<cloudflareJobId>",
      "error": {
        "code": "<stable code when available>",
        "message": "<safe message>"
      }
    }

Cancelled:

    {
      "type": "cancelled",
      "jobId": "<cloudflareJobId>"
    }

After a successful RunPod cancel, Cloudflare also updates the job channel to <code>status: "CANCELLED"</code>, <code>phase: "cancelled"</code>. A job's output may need explicit deletion through the artifact/generation routes; cancellation and deletion are separate operations.

# 20. Error contract

Exact codes defined in <code>privora/errors.py</code>:

## Caller-correctable codes

    UNSUPPORTED_MODE
    INVALID_ASPECT_RATIO
    INVALID_QUALITY
    INVALID_DURATION
    INVALID_SEED
    INVALID_STEPS
    UNKNOWN_FIELD
    MISSING_PROMPT
    MISSING_FRAME
    INVALID_REFERENCE_COUNT
    INVALID_REFERENCE_TYPE
    INVALID_REFERENCE_ROLE
    INVALID_REFERENCE_DURATION

These have <code>PrivoraError.is_client_error == true</code>.

## Other defined codes

    REFERENCE_PREPROCESSING_FAILED
    MODEL_LOAD_FAILED
    GENERATION_FAILED
    OUT_OF_MEMORY
    ENCODE_FAILED
    CONFIDENTIAL_ENCRYPTION_FAILED
    UPLOAD_FAILED

Not every defined execution code is currently wired at every lower-level exception boundary. For example, generic workflow/output failures may still return only <code>{"error":"..."}</code>. API Runtime must tolerate both the structured and unstructured forms while preferring <code>errorCode</code> when present.

Progress failure callback uses:

    {
      "jobId": "<cloudflareJobId>",
      "phase": "failed",
      "error": {
        "code": "<Privora code, workflow_error, no_output, or exception class>",
        "message": "<safe message>"
      }
    }

The public Cloudflare HTTP error shape is:

    {
      "error": "<message>",
      "details": "<optional structured details>"
    }

RunPod upstream failures are sanitized before being copied to that response/status surface.

Privacy requirements for every error layer:

- never include prompt text, compiled prompt, media bytes, user filename, local path, reference URL token, callback token, Cloudflare Access credential, RunPod credential, public/private bundle material other than explicitly public key metadata, or symmetric key material;
- <code>PrivoraError.internal</code> is container-log-only and must never be returned;
- downstream logs should record stable code, counts/limits, job IDs, phase, and endpoint/backend, not the whole request.

# 21. Artifact retrieval, deletion, and frontend Confidential fields

## Artifact paths

| Operation | Exact path |
| --- | --- |
| Retrieve | <code>GET /jobs/{cloudflareJobId}/artifact</code> |
| Retrieve legacy alias | <code>GET /jobs/{cloudflareJobId}/video</code> |
| Delete output | <code>DELETE /jobs/{cloudflareJobId}/artifact</code> or <code>/video</code> |
| Delete generation plus inputs | <code>DELETE /jobs/{cloudflareJobId}</code> |

Standard retrieval returns <code>video/mp4</code> with byte-range support.

Confidential retrieval returns <code>application/octet-stream</code>, attachment filename <code>artifact.enc</code>, <code>Cache-Control: private, no-store</code>, and:

    X-Privacy-Mode: confidential
    X-Artifact-Encrypted: true
    X-Artifact-Encryption-Version: 2
    X-Artifact-Algorithm: AES-256-GCM
    X-Artifact-Original-Content-Type: video/mp4

The browser downloads and decrypts the full container before playback.

## Existing frontend client contract

<code>client/confidential-generation.js</code> currently submits to:

    POST {apiBase}/generate

and retrieves:

    GET {apiBase}/jobs/{jobId}/artifact

Its current public submission helper writes legacy public fields:

    {
      "privacyMode": "confidential",
      "encryption": {
        "version": 2,
        "algorithm": "AES-256-GCM",
        "keyWrapAlgorithm": "RSA-OAEP-256",
        "publicKey": "<base64url SPKI>",
        "keyId": "<optional derived id>"
      }
    }

API Runtime may retain that public-facing shape for compatibility, but the RunPod-facing Worker should normalize it to canonical <code>privacy: {"mode":"confidential"}</code>.

The browser-side key bundle has exact fields:

    {
      "v": 1,
      "keyId": "<derived id>",
      "publicKey": "<base64url SPKI>",
      "publicKeyAlgorithm": "RSA-OAEP-256",
      "kdf": {
        "name": "pbkdf2-sha256",
        "salt": "<base64url 16 random bytes>",
        "parameters": {
          "iterations": 600000
        }
      },
      "createdAt": "<UTC ISO-8601>",
      "encryptedPrivateKey": "<AES-GCM ciphertext>",
      "privateKeyNonce": "<base64url 12-byte nonce>"
    }

The bundle may be persisted because its private key is encrypted under the user's passphrase. The passphrase, derived KEK, unlocked private key, and unwrapped video file key must never be sent to API Runtime, Cloudflare, RunPod, logs, callbacks, analytics, or job records.

The frontend must pass <code>expectArtifactId: jobId</code> when decrypting. The authenticated container header then prevents a valid artifact for one generation being substituted for another.

# 22. Known RunPod contract defects and cutover gates

These were found while deriving the downstream contract. They are not reasons for downstream systems to recreate the rules.

| Issue | Exact impact | Cutover action |
| --- | --- | --- |
| Canonical <code>generation_id</code> ordering defect | Original canonical requests read <code>generation_id</code> before assignment and raised <code>UnboundLocalError</code>. | **FIXED**: assignment precedes canonical routing and entrypoint regression coverage reaches the ComfyUI boundary. |
| Failed partial reference staging cleanup | A staging error before <code>build_privora_job()</code> returned could leave plaintext in a warm worker directory. | **FIXED**: the builder removes its job directory on every failure; tests cover image, video, audio, invalid-later-reference, fetch-timeout, and post-staging validation failures. |
| Video/audio download used image defaults | Original canonical media fetch used the 32 MiB image cap and <code>Accept: image/*</code>. | **FIXED**: image/video/audio use their exact 32/256/64 MiB caps and media-specific Accept headers. |
| Legacy arbitrary steps are overwritten | No-workflow legacy <code>steps: 14</code> built 20 steps. | **FIXED**: explicit legal steps reach the base-model scheduler and metadata verbatim with no Turbo LoRA. |
| Capability max duration was not executable at 150s | 150 seconds aligns to 3609 frames and is rejected. | **FIXED**: advertised maximum is derived as 3592 frames / 24 = 149.6667 seconds and is executable. |
| Remix source was not server-enforced | Any non-empty reference array previously passed Remix mode validation. | **FIXED**: RunPod requires at least one source-role video. |
| Soundtrack placement/type/duration was under-validated | Non-video attachment/explicit wrong nested type could parse; nested duration was not checked. | **FIXED**: exact nested audio shape, role, transport/media validation, and measured duration are enforced. |
| Mode/keyframe consistency was incomplete | Parser did not reject every irrelevant keyframe/reference combination. | **FIXED**: mode-specific input rules are enforced by the canonical parser. |
| Unknown canonical fields were ignored | Misspellings could silently select defaults. | **FIXED**: canonical requests use an exact allowlist and return <code>UNKNOWN_FIELD</code>; legacy parsing stays permissive. |
| Confidential reference relay absent | Current R2 input route stores plaintext and supports images only. | Implement/attach a non-persistent job-scoped media relay or explicitly declare the privacy downgrade. |

The fixes above belong only to the new committed/published image identity recorded below. A previously built tag, especially <code>multimodal-2</code>, must not be assumed to contain them.

## Hardened immutable image identity

| Field | Exact value |
| --- | --- |
| Repository | <code>ghcr.io/chrisditfort/minimax-h3-blackwell-serverless</code> |
| Tag | <code>multimodal-3</code> |
| Digest | <code>&lt;pending immutable publication&gt;</code> |
| Source commit | <code>&lt;injected by the release build as H3_BUILD_SOURCE_COMMIT&gt;</code> |
| Build ID | <code>&lt;injected by the release build as H3_BUILD_ID&gt;</code> |
| Compressed size | <code>&lt;pending manifest verification&gt;</code> |
| Base-model revision | <code>eb8a16107c595128b3a578f82d2ce2f75920c355</code> |
| Turbo-LoRA revision | <code>4cc1d817b6184899b41293954329f576cb5ae86b</code> |
| ComfyUI revision | <code>dec5d945</code> (<code>v0.30.2</code>) |

The pending fields above must be replaced from the published OCI manifest and completed CI run. They are never inferred from the old <code>multimodal-2</code> image.

# 23. Downstream implementation checklist

## RunPod-facing Cloudflare Worker

- Add a canonical generation branch that does not call <code>buildWorkflowForSettings()</code>.
- Preserve current graph construction only behind an explicit legacy compatibility branch.
- Remove routing-only <code>backend</code>/<code>model</code> before the canonical <code>input</code>.
- Forward exact canonical camel-case fields.
- Replace caller-supplied <code>progress</code>, <code>output</code>, reference tokens, and internal URLs with trusted values.
- Query capabilities using <code>{"input":{"mode":"capabilities"}}</code>; do not serve the stale static Ref2VA result.
- Continue using RunPod <code>/run</code>, <code>/status/{id}</code>, and <code>/cancel/{id}</code> with Bearer auth.
- Preserve <code>JOB_TOKEN_SECRET</code> purpose scoping and signed Confidential privacy/version claims.
- Preserve R2 output keys and artifact routes.
- Extend reference transport for video/audio without silently persisting Confidential plaintext.
- Treat <code>COMPLETED + output.error</code> as failed.

## API Runtime

- Keep Cloudflare Access credentials server-side and send exact Access headers.
- Expose semantic request fields, not ComfyUI fields.
- Enforce capability-derived availability and the 12-file product limit while allowing RunPod to revalidate.
- Use a mode-specific allowlist; do not forward unknown/misspelled fields.
- Store Cloudflare job ID and RunPod job ID separately.
- Normalize raw RunPod states to product states.
- Persist only safe metadata; never persist reference bytes/tokens or Confidential private capability.
- Return <code>output.generation</code> as authoritative actual settings after completion.
- Keep legacy clients on a compatibility route where canonical mapping is not lossless.

## PrivoraVideo frontend

- Use modes <code>create</code>, <code>animate</code>, <code>references</code>, <code>remix</code>.
- Use qualities <code>draft</code>, <code>standard</code>, <code>hd</code>, <code>ultra</code>.
- Use <code>generationMode</code> <code>quality</code>, <code>turbo</code>, <code>turboFast</code> according to <code>byFamily</code>.
- Send duration seconds and semantic <code>aspectRatio</code>; display returned actual duration/dimensions.
- Forward reference role metadata; never generate H3 tags.
- Enforce 12 total files and per-type limits in UX.
- Explain that Remix regenerates rather than deterministically edits.
- Keep the Confidential private key/passphrase client-side and decrypt <code>artifact.enc</code> locally.
- Do not inline Confidential reference bytes into a JSON job request.

# 24. Final canonical examples

## Standard Create, fully enriched by Cloudflare

    {
      "input": {
        "mode": "create",
        "prompt": "A red fox runs through wet ferns at dawn.",
        "quality": "standard",
        "aspectRatio": "16:9",
        "duration": 5,
        "seed": 51,
        "generationMode": "quality",
        "privacy": {
          "mode": "standard"
        },
        "progress": {
          "url": "https://<worker-host>/internal/jobs/<cloudflareJobId>/progress",
          "token": "<progress token>",
          "jobId": "<cloudflareJobId>"
        },
        "output": {
          "url": "https://<worker-host>/internal/jobs/<cloudflareJobId>/output",
          "token": "<output token>",
          "jobId": "<cloudflareJobId>"
        }
      },
      "policy": {
        "executionTimeout": 600000,
        "ttl": 1800000
      }
    }

## Confidential References, fully enriched by Cloudflare

    {
      "input": {
        "mode": "references",
        "prompt": "The character follows the supplied motion through a neon market.",
        "quality": "ultra",
        "aspectRatio": "16:9",
        "duration": 10,
        "seed": 51,
        "generationMode": "turboFast",
        "referenceFidelity": "high",
        "references": [
          {
            "type": "image",
            "role": "character",
            "url": "https://<non-persistent-media-relay>/<image>",
            "token": "<image token>"
          },
          {
            "type": "video",
            "role": "motion",
            "url": "https://<non-persistent-media-relay>/<video>",
            "token": "<video token>"
          }
        ],
        "privacy": {
          "mode": "confidential"
        },
        "encryption": {
          "version": 2,
          "algorithm": "AES-256-GCM",
          "keyWrapAlgorithm": "RSA-OAEP-256",
          "publicKeyAlgorithm": "RSA-OAEP-256",
          "publicKey": "<base64url DER SPKI>",
          "keyId": "<derived key id>"
        },
        "progress": {
          "url": "https://<worker-host>/internal/jobs/<cloudflareJobId>/progress",
          "token": "<progress token>",
          "jobId": "<cloudflareJobId>"
        },
        "output": {
          "url": "https://<worker-host>/internal/jobs/<cloudflareJobId>/output",
          "token": "<output token signed for confidential v2>",
          "jobId": "<cloudflareJobId>"
        }
      },
      "policy": {
        "executionTimeout": 600000,
        "ttl": 1800000
      }
    }

No <code>workflow</code>, raw dimensions, raw frames, raw steps, checkpoint/LoRA names, H3 tags, or ComfyUI node IDs belong in either canonical request.

# 25. Read-only live Cloudflare snapshot

This section records what was actually active during the 2026-08-29 inspection. It is evidence for the compatibility adapter, not the target capability contract.

## Active binding set

    R2 binding:
      H3_OUTPUTS -> minimax-h3-private-output

    Durable Object binding:
      JOB_CHANNEL -> JobChannel

    Plain variables:
      RUNPOD_ENDPOINT_ID = xa6b4vs5gdva3r
      RUNPOD_BLACKWELL_ENDPOINT_ID = d0p4f4hgxyqsy2
      JOB_TOKEN_TTL_SECONDS = 3600

    Secret names:
      JOB_TOKEN_SECRET
      RUNPOD_API_KEY
      RUNPOD_BLACKWELL_API_KEY

No secret values were read into this artifact.

## Live <code>GET /health</code>

    {
      "ok": true,
      "service": "minimax-h3-backend",
      "defaultBackend": "h3",
      "backends": {
        "h3": {
          "configured": true
        },
        "h3-blackwell": {
          "configured": true
        }
      },
      "features": {
        "statusProgress": true
      },
      "routes": [
        "GET /health",
        "GET /capabilities",
        "POST /jobs/:jobId/assets",
        "GET /jobs/:jobId/artifact (alias: /video)",
        "DELETE /jobs/:jobId/artifact (alias: /video)",
        "DELETE /jobs/:jobId",
        "GET /ws/jobs/:jobId",
        "POST /generate",
        "GET /status/:backend/:jobId",
        "POST /cancel/:backend/:jobId",
        "GET /status/:jobId (legacy; defaults to h3)",
        "POST /cancel/:jobId (legacy; defaults to h3)"
      ]
    }

This is Cloudflare service health. It does not start or query the rebuilt RunPod handler.

## Live stale <code>GET /capabilities</code>

The live document reports:

    {
      "fps": 24,
      "maxPixels": 1032192,
      "canvasMultiple": 32,
      "frameGrid": "frames % 17 == 5",
      "defaultQuality": "standard",
      "defaultAspectRatio": "16:9",
      "defaultDurationSeconds": 5,
      "statusProgress": {
        "available": true,
        "fields": [
          "phase",
          "percent",
          "step",
          "steps",
          "updatedAt"
        ]
      },
      "qualities": {
        "fast": {
          "steps": 14,
          "dimensions": {
            "16:9": "1024x576",
            "9:16": "576x1024",
            "1:1": "576x576",
            "4:3": "768x576",
            "3:4": "576x768"
          }
        },
        "standard": {
          "steps": 20,
          "dimensions": {
            "16:9": "1024x576",
            "9:16": "576x1024",
            "1:1": "576x576",
            "4:3": "768x576",
            "3:4": "576x768"
          }
        },
        "hd": {
          "steps": 20,
          "dimensions": {
            "16:9": "1344x768",
            "9:16": "768x1344",
            "1:1": "768x768",
            "4:3": "1024x768",
            "3:4": "768x1024"
          }
        }
      },
      "aspectRatios": [
        "16:9",
        "9:16",
        "1:1",
        "4:3",
        "3:4"
      ],
      "modes": {
        "text_to_video": {
          "available": true
        },
        "first_frame_to_video": {
          "available": true
        },
        "last_frame_to_video": {
          "available": true
        },
        "first_last_frame_to_video": {
          "available": true
        },
        "reference": {
          "available": false
        },
        "regenerate_2k": {
          "available": false
        }
      },
      "defaultPrivacyMode": "standard",
      "privacyModes": {
        "standard": {
          "available": true,
          "encrypts": false
        },
        "confidential": {
          "available": true,
          "encrypts": true
        },
        "private": {
          "available": false,
          "encrypts": false
        },
        "ephemeral": {
          "available": false,
          "encrypts": true
        }
      },
      "encryption": {
        "cryptoVersion": 2,
        "readableCryptoVersions": [
          1,
          2
        ],
        "algorithms": [
          "AES-256-GCM"
        ],
        "keyWrapAlgorithms": [
          "RSA-OAEP-256"
        ],
        "publicKeyFormat": "spki-der-base64url",
        "minPublicKeyBits": 3072,
        "retentionSeconds": {
          "min": 60,
          "max": 7776000
        }
      }
    }

The live response also includes human descriptions/reasons. Its Ref2VA reason says the checkpoint was deliberately excluded. That is the stale assertion the canonical runtime probe must replace.

# 26. Verification performed

No Cloudflare/API Runtime/frontend deployment and no paid RunPod generation was performed for this artifact.

Local verification in the rebuilt worktree:

    python -m pytest -q
    474 passed, 43 skipped, 153 subtests passed

    node --test worker.test.js worker.routes.test.js worker.internal.test.js worker.delete.test.js worker.confidential.test.js
    164 passed, 0 failed

    git diff --check
    clean

The skipped Python tests include environment/GPU-dependent coverage. The successful handler entrypoint tests stop at the first operation that genuinely needs ComfyUI/GPU, which verifies canonical routing and graph construction without spending GPU time.
