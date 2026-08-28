"""RunPod Serverless handler for MiniMax H3 FL2VA on RTX PRO 6000 Blackwell (sm_120).

The container starts this module as PID 1. It owns a single private ComfyUI process
bound to 127.0.0.1 and translates RunPod jobs into ComfyUI prompt executions.

Job shape (matches what the Cloudflare Worker already submits):

    {"input": {"workflow": { ...ComfyUI API-format workflow... }}}

Response shape (matches worker-comfyui / the existing Cloudflare Worker):

    {"images": [{"filename": ..., "type": ..., "data": <base64>}]}

Output delivery is pluggable (see OutputStore). H3_OUTPUT_MODE selects it:
  * "base64" (default) - inline base64, what the current Cloudflare Worker consumes.
  * "r2"               - AES-256-GCM encrypt in-worker, upload ciphertext only to R2.

Artefact protection is a separate, model-independent concern (see artifacts.py). A job may
carry a privacy mode and, for `confidential`, a caller-derived encryption key:

    {"input": {"workflow": {...},
               "privacy": {"mode": "confidential"},
               "encryption": {"algorithm": "AES-256-GCM", "key": "<base64url 32 bytes>",
                              "kdf": {...}, "keyId": "..."}}}

In that mode the finished MP4 is encrypted here, inside the inference container, and the
plaintext is deleted before anything is uploaded. Only ciphertext leaves this process.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Iterable

import requests
import runpod
import websocket

# Model-independent artefact protection: privacy modes, the encrypted container format,
# plaintext hygiene and redaction. Nothing in it knows about H3, ComfyUI or R2, which is
# what lets a second model reuse it unchanged.
import artifacts
from artifacts import (
    ArtifactError,
    GeneratedArtifact,
    ProtectedArtifact,
    redact,
)

# Experimental multi-GPU execution (H3_GPU_MODE=dual). Importing this here is inert: the
# package only touches ComfyUI when it is imported *inside* a rank, which is the only place
# H3_SP_RANK is set. In the handler it is just configuration and workflow rewriting.
from h3_parallel import config as gpu_config
from h3_parallel import shadow as shadow_graph

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# ComfyUI is baked at /opt/comfyui-baked in the Nightfall Blackwell base image. The base
# image's Pod entrypoint copies it to /workspace/runpod-slim/ComfyUI on first boot, but we
# deliberately bypass that entrypoint, and /workspace is a plausible network-volume mount
# point on RunPod. Running it in place from /opt is both faster and mount-proof.
COMFY_DIR = os.environ.get("COMFY_DIR", "/opt/comfyui-baked")
COMFY_HOST = "127.0.0.1"
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/tmp/comfy-output")
COMFY_TEMP_DIR = os.environ.get("COMFY_TEMP_DIR", "/tmp/comfy-temp")
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", os.path.join(COMFY_DIR, "input"))

# Seconds to wait for ComfyUI to answer /system_stats on cold start. Loading the CUDA
# stack on a Blackwell host is slow, so this is generous.
COMFY_STARTUP_TIMEOUT = int(os.environ.get("COMFY_STARTUP_TIMEOUT", "600"))
# Hard ceiling for a single workflow execution. Keep it under the RunPod endpoint's
# execution timeout so we can return a clean error instead of being killed mid-job.
JOB_TIMEOUT = int(os.environ.get("H3_JOB_TIMEOUT", "3000"))
# Individual websocket recv timeout. Generation routinely goes quiet for minutes while a
# sampler step runs, so a timeout here is normal and must NOT abort the job.
WS_RECV_TIMEOUT = int(os.environ.get("H3_WS_RECV_TIMEOUT", "30"))

OUTPUT_MODE = os.environ.get("H3_OUTPUT_MODE", "base64").strip().lower()

# Generous: a 5s 1024x576 MP4 is only a few MB, but a long clip on a slow link should not
# be abandoned after the video has already been paid for in GPU time.
OUTPUT_UPLOAD_TIMEOUT = float(os.environ.get("H3_OUTPUT_UPLOAD_TIMEOUT", "120"))

# --------------------------------------------------------------------------------------
# FlashBoot preload (opt-in, default off)
# --------------------------------------------------------------------------------------
PRELOAD_ENABLED = os.environ.get("H3_FLASHBOOT_PRELOAD", "0") == "1"
PRELOAD_TIMEOUT = int(os.environ.get("H3_FLASHBOOT_PRELOAD_TIMEOUT", "60"))

# The smallest graph the node schema allows: width/height min 32 step 32, length min 5
# step 17 (the model's 17k+5 grid). Read off MiniMaxH3ImageToVideo's own schema, not
# guessed. Overridable in case a base-image bump tightens the limits.
PRELOAD_WIDTH = int(os.environ.get("H3_FLASHBOOT_PRELOAD_WIDTH", "32"))
PRELOAD_HEIGHT = int(os.environ.get("H3_FLASHBOOT_PRELOAD_HEIGHT", "32"))
PRELOAD_LENGTH = int(os.environ.get("H3_FLASHBOOT_PRELOAD_LENGTH", "5"))
PRELOAD_PROMPT = os.environ.get("H3_FLASHBOOT_PRELOAD_PROMPT", "preload")

# These MUST match the real workflow's loader inputs byte for byte. ComfyUI keys its
# output cache on class_type + inputs and explicitly excludes node id
# (CacheKeySetInputSignature.include_node_id_in_input() -> False), so identical loader
# inputs are what makes the real job reuse the objects this preload put on the GPU.
# Defaults are the four models baked by models.tsv.
PRELOAD_UNET = os.environ.get(
    "H3_FLASHBOOT_PRELOAD_UNET", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)
PRELOAD_CLIP = os.environ.get(
    "H3_FLASHBOOT_PRELOAD_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
)
PRELOAD_VIDEO_VAE = os.environ.get(
    "H3_FLASHBOOT_PRELOAD_VIDEO_VAE", "minimax_h3_video_vae_fp16.safetensors"
)
PRELOAD_AUDIO_VAE = os.environ.get(
    "H3_FLASHBOOT_PRELOAD_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors"
)

# ComfyUI history buckets that carry saved artefacts. SaveVideo emits ui.PreviewVideo,
# whose as_dict() is {"images": [...], "animated": (True,)} - so MP4s land under "images",
# which is exactly what the Cloudflare Worker already reads.
OUTPUT_KEYS = ("images", "gifs", "videos", "audio")

_comfy_process: subprocess.Popen | None = None
_start_lock = threading.Lock()


def log(message: str) -> None:
    print(f"[handler] {message}", flush=True)


# --------------------------------------------------------------------------------------
# Multi-GPU execution (experimental)
# --------------------------------------------------------------------------------------
#
# In dual mode this container runs one ComfyUI per GPU:
#
#     handler.py (PID 1)
#     +-- rank 0   CUDA_VISIBLE_DEVICES=0   127.0.0.1:8188   <- the only rank the
#     |                                                          handler reads results from
#     +-- rank 1   CUDA_VISIBLE_DEVICES=1   127.0.0.1:8189   <- shadow: holds up its half
#                                                                of every collective
#
# Both ranks execute the same graph, in step, and split the packed token sequence between
# them inside the DiT (see h3_parallel/ulysses.py). The shadow's copy of the workflow is
# rewritten to stop at the VAE decodes, so only rank 0 ever produces a file.


def _visible_cuda_devices() -> int:
    """How many GPUs this container was actually scheduled. 0 if torch cannot say."""
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as error:  # pragma: no cover - torch is always present in the image
        log(f"WARNING: could not count CUDA devices: {error}")
        return 0


# Resolved at import so every part of the handler can read it, but a failure is *carried*
# rather than raised: main() logs it properly and exits, instead of the worker dying inside
# an import traceback that says nothing about which environment variable was wrong.
GPU_MODE_ERROR: Exception | None = None
GPU_CONFIG: "gpu_config.GpuConfig | None" = None
try:
    GPU_CONFIG = gpu_config.resolve(device_count=_visible_cuda_devices())
except Exception as _gpu_error:  # noqa: BLE001 - reported by main(), which then exits
    GPU_MODE_ERROR = _gpu_error


class ComfyRank:
    """One ComfyUI process, pinned to one GPU."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.port = gpu_config.comfy_port_for_rank(rank, COMFY_PORT)
        self.url = f"http://{COMFY_HOST}:{self.port}"
        self.process: subprocess.Popen | None = None
        self.shadow = rank > 0
        # Rank 0 keeps the paths the known-good image uses, so the single-GPU path and the
        # dual-GPU path write to exactly the same places. Shadows get their own, which are
        # emptied after every job.
        self.output_dir = COMFY_OUTPUT_DIR if rank == 0 else f"{COMFY_OUTPUT_DIR}-rank{rank}"
        self.temp_dir = COMFY_TEMP_DIR if rank == 0 else f"{COMFY_TEMP_DIR}-rank{rank}"
        # A private user directory keeps two processes off one SQLite asset database.
        self.user_dir = None if rank == 0 else f"/tmp/comfy-user-rank{rank}"

    def is_ready(self) -> bool:
        return _comfy_is_ready(self.url)

    def gpu_status(self, timeout: float = 10.0) -> dict:
        """Ask the rank's /h3/gpu route what it actually managed to set up."""
        response = requests.get(f"{self.url}/h3/gpu", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"<ComfyRank {self.rank} port={self.port}>"


_ranks: list[ComfyRank] = []
_shadow_ranks: list[ComfyRank] = []


# --------------------------------------------------------------------------------------
# Performance instrumentation
# --------------------------------------------------------------------------------------
#
# Every phase boundary below comes from something the handler can already see: its own
# call sites, plus the `executing` / `progress` frames ComfyUI sends over the websocket it
# is already connected to. Nothing inside ComfyUI is patched or wrapped, so this cannot
# change what a job computes - the worst a bug in here can do is print "n/a".

# Identifies this worker *process*. Two jobs logging the same proc= ran back-to-back on
# one warm process; different values mean each paid its own cold start. On a scale-to-zero
# endpoint that is the only way to tell from the logs alone whether a worker was actually
# reused, which is precisely the question the cold-start work needs answered.
PROCESS_ID = uuid.uuid4().hex[:12]
_PROCESS_START = time.monotonic()

# Set once, by whichever call to start_comfyui() actually launched ComfyUI.
_comfy_boot_seconds: float | None = None
_jobs_served = 0
_jobs_lock = threading.Lock()

# The per-node breakdown is a second line per job; H3_PERF_NODES=0 turns it off.
PERF_NODES = os.environ.get("H3_PERF_NODES", "1") == "1"
PERF_NODES_TOP = int(os.environ.get("H3_PERF_NODES_TOP", "6"))


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


# Public progress phases. Deliberately a small, stable vocabulary: ComfyUI's own log text
# is never forwarded, so the browser sees a contract we control rather than internals.
PHASE_STARTING = "starting_worker"
PHASE_COMFY_READY = "comfy_ready"
PHASE_LOADING = "loading_models"
PHASE_SAMPLING = "sampling"
PHASE_DECODING = "decoding"
PHASE_UPLOADING = "uploading"
PHASE_COMPLETED = "completed"
PHASE_FAILED = "failed"
PHASE_CANCELLED = "cancelled"

# Cloudflare Access sits in front of the Worker, so every call back into it - progress,
# output upload, asset download - has to present an Access service token as well as the
# job-scoped bearer token. The two are different things and both are required: Access
# decides whether the request reaches the Worker at all, and the bearer token decides what
# it is allowed to do once it gets there.
#
# Set CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET on the RunPod endpoint. Left unset,
# the headers are simply omitted, which is correct for a Worker not behind Access.
# Three spellings are accepted, in this order:
#
#   CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET
#   CLOUDFLARE_ACCESS_CLIENT_ID / CLOUDFLARE_ACCESS_CLIENT_SECRET   <- Cloudflare's own
#       convention; wrangler asks for exactly these names when it hits an Access-protected
#       host, so this is the spelling to prefer.
#   CLOUDFLARE_ACCESS_KEY_ID / CLOUDFLARE_SECRET_ACCESS_KEY
#
# The third pair is accepted only because it is easy to reach for, but be careful: those
# names conventionally mean R2's *S3-API* credentials, which are a completely different
# credential and will not get past Access. What belongs in all three is an Access
# **service token** from Zero Trust -> Access -> Service Auth.
CF_ACCESS_CLIENT_ID = (
    os.environ.get("CF_ACCESS_CLIENT_ID")
    or os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    or os.environ.get("CLOUDFLARE_ACCESS_KEY_ID")
    or ""
).strip()
CF_ACCESS_CLIENT_SECRET = (
    os.environ.get("CF_ACCESS_CLIENT_SECRET")
    or os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    or os.environ.get("CLOUDFLARE_SECRET_ACCESS_KEY")
    or ""
).strip()


def _describe_non_2xx(response) -> str:
    """Explain a non-2xx, naming Cloudflare Access when that is what happened.

    Access answers an unauthenticated call with a 302 to its login page. `requests`
    follows redirects by default and Response.ok is merely `status_code < 400`, so both
    the redirect and the resulting HTML login page read as success unless this is checked
    explicitly. That is exactly how a job once reported a video it had never uploaded.
    """
    # Every field is read defensively: this only ever builds a log string, and must not be
    # able to raise into the caller's error handling and have the failure counted twice.
    status = getattr(response, "status_code", "?")
    location = (getattr(response, "headers", None) or {}).get("Location", "") or ""
    if status in (301, 302, 303, 307, 308):
        if "cloudflareaccess.com" in location or "/cdn-cgi/access/login" in location:
            # State whether we even sent a token. "Rejected" and "never sent" produce
            # the identical 302, and the startup line that distinguishes them scrolls out
            # of RunPod's log window - so the answer belongs in the error itself.
            if _is_unresolved_secret_reference(CF_ACCESS_CLIENT_ID) or (
                _is_unresolved_secret_reference(CF_ACCESS_CLIENT_SECRET)
            ):
                return (
                    f"HTTP {status} to a Cloudflare Access login page. The Access service "
                    "token is an UNRESOLVED RunPod secret reference - the value is still "
                    "the literal '{{ RUNPOD_SECRET_... }}' placeholder, so no real "
                    "credential was sent. Create account-level secrets with those exact "
                    "names under RunPod Settings -> Secrets, or replace the reference with "
                    "the value itself."
                )
            if not (CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET):
                which = []
                if not CF_ACCESS_CLIENT_ID:
                    which.append("client id")
                if not CF_ACCESS_CLIENT_SECRET:
                    which.append("secret")
                return (
                    f"HTTP {status} to a Cloudflare Access login page, and NO Access "
                    f"service token was sent because the {' and '.join(which)} is missing "
                    "from this container's environment. Set CLOUDFLARE_ACCESS_KEY_ID and "
                    "CLOUDFLARE_SECRET_ACCESS_KEY on the RunPod endpoint. This is a "
                    "configuration problem, not an Access policy problem."
                )
            return (
                f"HTTP {status} to a Cloudflare Access login page. An Access service token "
                f"WAS sent (client id {CF_ACCESS_CLIENT_ID[:6]}...{CF_ACCESS_CLIENT_ID[-8:]}) "
                "and Access rejected it. Check the token is valid and that a Service Auth "
                "policy on this application allows it."
            )
        return f"HTTP {status} redirect to {location[:120]!r}"
    body = str(getattr(response, "text", "") or "")[:200]
    return f"HTTP {status}: {body}"


def _is_2xx(response) -> bool:
    """True only for a real success.

    Deliberately not `response.ok`, which is `status_code < 400` and therefore treats a
    302 as a success.
    """
    return 200 <= response.status_code < 300


def log_callback_configuration() -> None:
    """Say at boot whether an Access service token reached the container.

    Worth its own line because of how this fails otherwise. A missing token and a rejected
    token both end as the same 302, and the operator's own curl works because they pass the
    headers by hand - so "it works from my machine" tells you nothing about what the worker
    is sending. Only the variable names and the client-id shape are logged; the secret is
    never printed, and neither is the full id.
    """
    if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
        shape = "ends in '.access'" if CF_ACCESS_CLIENT_ID.endswith(".access") else (
            "does NOT end in '.access' - this may be an R2 S3 key rather than an Access "
            "service token"
        )
        log(
            f"Cloudflare Access service token: configured "
            f"(client id {CF_ACCESS_CLIENT_ID[:6]}...{CF_ACCESS_CLIENT_ID[-8:]}, {shape})"
        )
    elif CF_ACCESS_CLIENT_ID or CF_ACCESS_CLIENT_SECRET:
        missing = "secret" if CF_ACCESS_CLIENT_ID else "client id"
        log(
            f"WARNING: Cloudflare Access service token is HALF configured - the {missing} "
            "is missing, so no Access headers will be sent and callbacks will be rejected."
        )
    else:
        log(
            "Cloudflare Access service token: NOT configured. Set "
            "CLOUDFLARE_ACCESS_KEY_ID and CLOUDFLARE_SECRET_ACCESS_KEY on the endpoint if "
            "the Worker is behind Access, or progress callbacks and the R2 upload will be "
            "bounced to a login page."
        )


def _is_unresolved_secret_reference(value: str) -> bool:
    """True if the value is a RunPod secret reference that never got substituted.

    RunPod env values may be written as {{ RUNPOD_SECRET_<name> }}. If no account-level
    secret of that name exists, RunPod passes the literal braces through instead of
    failing - so the variable looks set, the handler sends it as a header, and Cloudflare
    Access rejects a credential that is really a template string. Worth naming explicitly,
    because "configured but wrong" and "configured correctly" are otherwise identical from
    inside the container.
    """
    text = (value or "").strip()
    return text.startswith("{{") and text.endswith("}}")


def cloudflare_access_headers() -> dict:
    """Service-token headers for Cloudflare Access, or {} when not usably configured."""
    if _is_unresolved_secret_reference(CF_ACCESS_CLIENT_ID) or _is_unresolved_secret_reference(
        CF_ACCESS_CLIENT_SECRET
    ):
        return {}
    if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
        return {
            "CF-Access-Client-Id": CF_ACCESS_CLIENT_ID,
            "CF-Access-Client-Secret": CF_ACCESS_CLIENT_SECRET,
        }
    return {}


PROGRESS_TIMEOUT = float(os.environ.get("H3_PROGRESS_TIMEOUT", "3"))
# Floor between *step* events. Phase changes and the final step ignore it.
PROGRESS_MIN_INTERVAL = float(os.environ.get("H3_PROGRESS_MIN_INTERVAL", "0.4"))


class ProgressReporter:
    """Best-effort progress delivery to the Cloudflare Worker.

    Three properties matter more than completeness here:

      * It never raises. A generation that has already cost minutes of GPU time must not
        fail because a callback timed out.
      * It never blocks for long. The timeout is seconds, not the default connect/read.
      * Its cost is never charged to sampling. The caller records callback time as its own
        [perf] field, so a slow Worker cannot inflate the sampling measurement.

    The endpoint and token are job-scoped and arrive in the job payload, so nothing
    permanent is stored in this image.
    """

    def __init__(self, job_id: str, url: str | None = None, token: str | None = None) -> None:
        self.job_id = job_id
        self.url = url
        self.token = token
        self.sent = 0
        self.failed = 0
        self.seconds = 0.0
        self._last_step_at = 0.0
        self._phase: str | None = None
        self._warned = False

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def phase(self, phase: str, **extra) -> None:
        """Report a phase change. Always delivered - these are the events clients act on."""
        self._phase = phase
        self._emit({"phase": phase, **extra})

    def step(self, step: int, steps: int) -> None:
        """Report sampler progress, coalesced so a fast sampler cannot flood the Worker."""
        now = time.monotonic()
        is_last = bool(steps) and step >= steps
        if not is_last and (now - self._last_step_at) < PROGRESS_MIN_INTERVAL:
            return
        self._last_step_at = now

        payload = {"phase": PHASE_SAMPLING, "step": step, "steps": steps}
        if steps:
            payload["percent"] = int(round(step / steps * 100))
        self._emit(payload)

    def _emit(self, payload: dict) -> None:
        if not self.enabled:
            return

        body = {"jobId": self.job_id, **payload}
        headers = {"Content-Type": "application/json", **cloudflare_access_headers()}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        began = time.monotonic()
        try:
            # allow_redirects=False: a redirect here is never a real delivery, and
            # following one silently turns an auth rejection into a counted success.
            response = requests.post(
                self.url,
                json=body,
                headers=headers,
                timeout=PROGRESS_TIMEOUT,
                allow_redirects=False,
            )
            if _is_2xx(response):
                self.sent += 1
            else:
                self.failed += 1
                if not self._warned:
                    # Once per job: 20+ identical warnings would bury the log.
                    self._warned = True
                    log(f"WARNING: progress callback rejected - {_describe_non_2xx(response)}")
        except Exception as error:
            # Swallowed on purpose: Cloudflare being unreachable is not a generation error.
            self.failed += 1
            log(f"WARNING: progress callback failed ({type(error).__name__}: {error})")
        finally:
            self.seconds += time.monotonic() - began


class JobTimer:
    """Wall-clock marks for a single job, emitted as one [perf] line when it finishes.

    Deliberately forgiving: a phase whose boundary never arrived (a dropped websocket, a
    job that failed before sampling) reports "n/a" rather than raising. Instrumentation
    must never be the reason a paid generation fails.
    """

    def __init__(self, workflow: dict | None = None) -> None:
        self.start = time.monotonic()
        self.marks: dict[str, float] = {}
        self.spans: dict[str, float] = {}
        self._workflow = workflow if isinstance(workflow, dict) else {}
        self._open_node: tuple[str, float] | None = None
        self._node_totals: dict[str, float] = {}
        self.steps_seen = 0
        self.steps_total: int | None = None
        self.output_bytes: int | None = None
        self.progress_callbacks: tuple[int, int] | None = None
        #: Set once the job's privacy mode is known, so a perf line can be read per mode.
        self.privacy_mode: str | None = None

    # -- marks -------------------------------------------------------------------------

    def mark(self, name: str) -> None:
        """Record the first time `name` happened."""
        self.marks.setdefault(name, time.monotonic())

    def remark(self, name: str) -> None:
        """Record the most recent time `name` happened."""
        self.marks[name] = time.monotonic()

    def add_span(self, name: str, seconds: float) -> None:
        self.spans[name] = self.spans.get(name, 0.0) + seconds

    def between(self, first: str, second: str) -> float | None:
        a, b = self.marks.get(first), self.marks.get(second)
        if a is None or b is None or b < a:
            return None
        return b - a

    # -- websocket-driven phase detection ----------------------------------------------

    def on_progress(self, value, total) -> None:
        """A sampler progress frame. The first one is where sampling begins."""
        self.mark("sampling_start")
        self.remark("sampling_end")
        if isinstance(total, int) and total > 0:
            self.steps_total = total
        if isinstance(value, int):
            self.steps_seen = max(self.steps_seen, value)
            # The first *completed* step carries the model-initialisation cost, which on a
            # cold worker is most of the stall before steady-state sampling. Break it out
            # so it is not mistaken for per-step sampler cost.
            if value >= 1:
                self.mark("first_step_done")

    def on_node(self, node_id: str) -> None:
        """An `executing` frame: the named node starts, so the previous one has ended."""
        now = time.monotonic()
        self._close_node(now)
        self._open_node = (self._label(node_id), now)

    def on_execution_end(self) -> None:
        self._close_node(time.monotonic())
        self.mark("execution_end")

    def _close_node(self, now: float) -> None:
        if self._open_node is None:
            return
        label, began = self._open_node
        self._node_totals[label] = self._node_totals.get(label, 0.0) + (now - began)
        self._open_node = None

    def _label(self, node_id: str) -> str:
        node = self._workflow.get(str(node_id))
        if isinstance(node, dict):
            class_type = node.get("class_type")
            if class_type:
                return str(class_type)
        return f"node{node_id}"

    # -- output ------------------------------------------------------------------------

    def summary(self, *, job_index: int, status: str) -> str:
        total = time.monotonic() - self.start
        sampling = self.between("sampling_start", "sampling_end")

        fields = [
            f"proc={PROCESS_ID}",
            # Which execution path produced this number. Without it a log line from the
            # experimental image is indistinguishable from a single-GPU one, and the whole
            # A/B comparison rests on being able to tell them apart.
            f"gpu_mode={GPU_CONFIG.mode if GPU_CONFIG else 'unresolved'}",
            f"gpu_count={GPU_CONFIG.world_size if GPU_CONFIG else '?'}",
            f"strategy={GPU_CONFIG.strategy if GPU_CONFIG else 'unresolved'}",
            # First job in this process, so this job paid for the cold start.
            f"cold_process={'true' if job_index == 1 else 'false'}",
            f"job_in_proc={job_index}",
            f"proc_age={_secs(self.start - _PROCESS_START)}",
            f"comfy_boot={_secs(_comfy_boot_seconds)}",
            # ~0 means ComfyUI was already up, so its boot was NOT on this job's clock.
            f"comfy_wait={_secs(self.spans.get('comfy_wait'))}",
        ]
        if "stage_image" in self.spans:
            fields.append(f"stage_image={_secs(self.spans['stage_image'])}")
        if "input_download" in self.spans:
            fields.append(f"input_download={_secs(self.spans['input_download'])}")
        fields += [
            f"submit={_secs(self.spans.get('submit'))}",
            # Queued -> first sampler frame: model staging, text encode, graph setup.
            f"pre_sampling={_secs(self.between('submitted', 'sampling_start'))}",
            f"first_step={_secs(self.between('sampling_start', 'first_step_done'))}",
            f"sampling={_secs(sampling)}",
            f"steps={self.steps_seen}/{self.steps_total if self.steps_total else '?'}",
            # Last sampler frame -> workflow finished: VAE decode, audio, mux, save.
            f"decode={_secs(self.between('sampling_end', 'execution_end'))}",
            f"output={_secs(self.spans.get('output'))}",
        ]
        # Encryption is broken out so the cost of Confidential Generation is a number
        # somebody can read off a log line rather than an argument. It sits inside
        # `output`, alongside the upload.
        #
        # In milliseconds, not the seconds every other field uses, because encrypting a
        # five-second clip takes single-digit milliseconds - at this line's one-decimal
        # precision it would always print 0.0s, which is exactly not the point.
        if self.privacy_mode:
            fields.append(f"privacy={self.privacy_mode}")
        if "encryption" in self.spans:
            fields.append(f"encryption_ms={round(self.spans['encryption'] * 1000)}")
        # Upload is broken out from `output` so a slow R2 write is never mistaken for
        # slow inference, and callback latency is reported separately from both.
        if "output_upload" in self.spans:
            fields.append(f"output_upload={_secs(self.spans['output_upload'])}")
        if self.output_bytes is not None:
            fields.append(f"output_bytes={self.output_bytes}")
        if self.progress_callbacks is not None:
            sent, failed = self.progress_callbacks
            fields.append(f"progress_callbacks={sent}/{sent + failed}")
        if "progress_callback_time" in self.spans:
            fields.append(f"progress_time={_secs(self.spans['progress_callback_time'])}")
        fields += [
            f"total={_secs(total)}",
            f"status={status}",
        ]
        if sampling and self.steps_seen > 1:
            fields.insert(-1, f"per_step={sampling / self.steps_seen:.2f}s")
        return "[perf] " + " ".join(fields)

    def node_summary(self) -> str | None:
        if not self._node_totals:
            return None
        ranked = sorted(self._node_totals.items(), key=lambda kv: kv[1], reverse=True)
        shown = ranked[: max(1, PERF_NODES_TOP)]
        return "[perf] nodes " + " ".join(f"{name}={value:.1f}s" for name, value in shown)


def emit_perf(timer: JobTimer, *, job_index: int, status: str) -> None:
    """Print the timing lines. Never raises - a broken metric must not fail a job."""
    try:
        log(timer.summary(job_index=job_index, status=status))
        if PERF_NODES:
            nodes = timer.node_summary()
            if nodes:
                log(nodes)
        # Separate line: it costs two loopback HTTP calls, which have no business inside
        # JobTimer, and it only exists in dual mode.
        gpu_fields = gpu_perf_fields()
        if gpu_fields:
            log("[perf] gpu " + " ".join(gpu_fields))
    except Exception as error:  # pragma: no cover - defensive only
        log(f"WARNING: could not emit perf summary: {error}")


# --------------------------------------------------------------------------------------
# Blackwell / CUDA validation
# --------------------------------------------------------------------------------------


def log_torch_environment() -> None:
    """Log the GPU stack once at boot so a bad rollout is obvious in the RunPod logs.

    This must never raise: a logging problem should not stop the worker from serving.
    """
    try:
        import torch
    except Exception as error:  # pragma: no cover - torch is always present in the image
        log(f"WARNING: could not import torch: {error}")
        return

    log(f"torch.__version__      = {torch.__version__}")
    log(f"torch.version.cuda     = {torch.version.cuda}")

    try:
        if not torch.cuda.is_available():
            log("WARNING: torch.cuda.is_available() is False - no GPU visible to the worker.")
            return

        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        log(f"torch.cuda.get_device_name(0)       = {name}")
        log(f"torch.cuda.get_device_capability(0) = {capability}")

        if capability[0] < 12:
            log(
                f"WARNING: expected Blackwell sm_120 (12, 0) but found {capability}. "
                "This image ships a cu130 / sm_120 PyTorch build; a lower capability "
                "means the wrong GPU was scheduled."
            )
        if not str(torch.version.cuda or "").startswith("13"):
            log(
                f"WARNING: expected a CUDA 13 build but torch reports {torch.version.cuda}. "
                "Blackwell support requires the cu130 runtime."
            )
    except Exception as error:
        log(f"WARNING: GPU probe failed: {error}")


def _device_capability() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def build_comfy_env() -> dict:
    """Return the environment for the ComfyUI subprocess, with attention backend resolved.

    The sageattn3 wheel in the base image is compiled for a single compute capability
    (SAGE_SUPPORTED_CC, 12.0 / Blackwell). Importing it succeeds on any GPU, so import
    alone proves nothing - the failure only appears when a kernel is launched, as
    `cudaErrorNoKernelImageForDevice` on *every* attention call. That floods the log and
    silently falls back to PyTorch attention for the whole run.

    So if the scheduled GPU is not the capability the wheel was built for, turn
    SageAttention3 off up front. ComfyUI then picks its own backend cleanly. Set
    H3_SAGE_AUTODETECT=0 to force whatever COMFY_SAGE_ATTENTION3 says instead.
    """
    env = os.environ.copy()
    requested = env.get("COMFY_SAGE_ATTENTION3", "0") == "1"
    log(f"COMFY_SAGE_ATTENTION3 = {env.get('COMFY_SAGE_ATTENTION3', '<unset>')}")

    if not requested:
        log("SageAttention3 disabled by env; ComfyUI will pick its default attention backend.")
        return env

    try:
        import sageattn3  # noqa: F401
    except Exception as error:
        # ComfyUI's sage3 patch raises at import if the package is missing, which would
        # stop ComfyUI booting at all. Disable rather than let that happen.
        log(f"WARNING: sageattn3 is unusable ({error}); disabling SageAttention3.")
        env["COMFY_SAGE_ATTENTION3"] = "0"
        return env

    # Version lookup is a separate, best-effort step on purpose. It keys off the
    # *distribution* name, which need not match the module name, and a wheel installed
    # without dist-info has no metadata at all - so a lookup failure says nothing about
    # whether the backend works. Folding it into the import check above would disable
    # SageAttention3 on a perfectly good Blackwell card over a missing version string.
    try:
        import importlib.metadata

        log(f"sageattn3 present, version {importlib.metadata.version('sageattn3')}")
    except Exception:
        log("sageattn3 present (version metadata unavailable)")

    if env.get("H3_SAGE_AUTODETECT", "1") != "1":
        log("H3_SAGE_AUTODETECT=0; leaving SageAttention3 enabled without a capability check.")
        return env

    capability = _device_capability()
    if capability is None:
        log("WARNING: could not read GPU capability; leaving SageAttention3 as configured.")
        return env

    supported = env.get("SAGE_SUPPORTED_CC", "12.0")
    try:
        supported_major = int(float(supported))
    except ValueError:
        supported_major = 12

    if capability[0] != supported_major:
        log(
            f"WARNING: SageAttention3 was built for compute capability {supported} but this "
            f"worker was scheduled a {capability[0]}.{capability[1]} GPU. Disabling it - "
            "otherwise every attention call fails with "
            "'no kernel image is available for execution on the device' and falls back to "
            "PyTorch attention anyway, one error per call."
        )
        log(
            "  This is a GPU scheduling problem: pin the RunPod endpoint to an "
            "RTX PRO 6000 Blackwell to get SageAttention3. Generation still works without it."
        )
        env["COMFY_SAGE_ATTENTION3"] = "0"
    else:
        log(f"SageAttention3 enabled: GPU capability {capability} matches the wheel ({supported}).")

    return env


# --------------------------------------------------------------------------------------
# ComfyUI lifecycle
# --------------------------------------------------------------------------------------


def _comfy_is_ready(url: str = COMFY_URL) -> bool:
    try:
        response = requests.get(f"{url}/system_stats", timeout=5)
    except requests.RequestException:
        return False
    return response.ok


def parse_extra_args(raw: str) -> list[str]:
    """Split COMFY_EXTRA_ARGS into argv, tolerating quoting and bad input.

    shlex rather than str.split() so a quoted value survives ("--foo 'a b'" used to become
    three broken tokens). If the string is unbalanced, shlex raises - fall back to a naive
    split rather than refusing to start ComfyUI at all, since a malformed tuning flag is
    never worth taking the worker down for.
    """
    if not raw or not raw.strip():
        return []
    try:
        return shlex.split(raw)
    except ValueError as error:
        log(f"WARNING: COMFY_EXTRA_ARGS is not valid shell syntax ({error}); splitting naively.")
        return raw.split()


def comfy_command(rank: ComfyRank) -> list[str]:
    """The argv for one rank's ComfyUI. Rank 0's is byte-identical to single-GPU mode."""
    command = [
        sys.executable,
        "main.py",
        "--listen",
        COMFY_HOST,  # localhost only: RunPod Serverless exposes no ports
        "--port",
        str(rank.port),
        "--preview-method",
        "none",  # previews waste VRAM and bandwidth in a headless worker
        "--disable-auto-launch",
        "--disable-metadata",
        "--output-directory",
        rank.output_dir,
        "--temp-directory",
        rank.temp_dir,
        "--input-directory",
        COMFY_INPUT_DIR,  # shared: a staged keyframe must be visible to every rank
    ]
    if rank.user_dir:
        command += ["--user-directory", rank.user_dir]
    command.extend(parse_extra_args(os.environ.get("COMFY_EXTRA_ARGS", "")))
    return command


def rank_env(rank: ComfyRank) -> dict:
    """Environment for one rank: its GPU, its place in the group, its attention backend."""
    env = build_comfy_env()
    if GPU_CONFIG is None or not GPU_CONFIG.dual:
        # Single mode must not leave a stale rank marker behind, or h3_parallel would try
        # to join a group that nobody else is in.
        env.pop("H3_SP_RANK", None)
        return env

    # One GPU each. Every rank then sees its own device as cuda:0, which is what lets
    # ComfyUI's device handling and NCCL's rank-to-device mapping both stay conventional.
    env["CUDA_VISIBLE_DEVICES"] = str(rank.rank)
    env["H3_SP_RANK"] = str(rank.rank)
    env["H3_SP_WORLD_SIZE"] = str(GPU_CONFIG.world_size)
    env["H3_SP_MASTER_ADDR"] = GPU_CONFIG.master_addr
    env["H3_SP_MASTER_PORT"] = str(GPU_CONFIG.master_port)
    return env


def _launch_rank(rank: ComfyRank) -> None:
    for directory in (rank.output_dir, rank.temp_dir, COMFY_INPUT_DIR, rank.user_dir):
        if not directory:
            continue
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as error:
            log(f"WARNING: could not create {directory}: {error}")

    command = comfy_command(rank)
    env = rank_env(rank)
    extra_args = parse_extra_args(os.environ.get("COMFY_EXTRA_ARGS", ""))
    # Logged in full and on its own line so an A/B run can be attributed to the exact
    # flags it used - `--highvram` in particular is invisible otherwise. The wording of
    # the second line is what README's --highvram A/B recipe tells you to grep for.
    log(f"Starting ComfyUI rank {rank.rank}: {' '.join(command)} (cwd={COMFY_DIR})")
    log(
        "ComfyUI effective args: "
        f"COMFY_EXTRA_ARGS={os.environ.get('COMFY_EXTRA_ARGS', '<unset>')!r} "
        f"-> extra={extra_args or '[]'}"
    )
    log(
        f"ComfyUI rank {rank.rank} device: "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<all>')}"
    )
    rank.process = subprocess.Popen(command, cwd=COMFY_DIR, env=env)


def _verify_dual_ranks() -> None:
    """Refuse to serve unless every rank says it really is running the parallel path.

    ComfyUI swallows a custom node that raises during import, so a NCCL failure or a
    self-test mismatch would otherwise leave two healthy-looking ComfyUIs that each
    quietly compute the whole sequence - correct video, single-GPU speed, and a benchmark
    number that means nothing. /h3/gpu is the rank's own account of what it managed to do.
    """
    for rank in _ranks:
        try:
            status = rank.gpu_status()
        except Exception as error:
            raise RuntimeError(
                f"rank {rank.rank} did not answer /h3/gpu ({error}). The h3_parallel "
                "custom node is not loaded in that ComfyUI, so dual mode is not active."
            ) from error

        if not status.get("ready"):
            raise RuntimeError(
                f"rank {rank.rank} reports the sequence-parallel path is NOT ready: "
                f"{status.get('error') or status}"
            )
        patched = status.get("patched") or {}
        if not (patched.get("dit") and patched.get("attention")):
            raise RuntimeError(
                f"rank {rank.rank} came up but the H3 model was not patched ({patched}). "
                "It would run the full sequence on one GPU."
            )
        log(
            f"[H3-GPU] rank {rank.rank} verified: strategy={status.get('strategy')} "
            f"world_size={status.get('world_size')} patched={patched} "
            f"selftest={status.get('selftest')}"
        )


def start_comfyui() -> None:
    """Start every rank's ComfyUI and block until they all genuinely answer HTTP.

    Serialised behind a lock because RunPod may hand us concurrent jobs, and re-entrant
    starts would race two ComfyUI processes onto the same port.

    In dual mode this is also where the NCCL group forms: each rank blocks inside its
    custom-node import until its peer arrives, so a rank answering /system_stats already
    proves the group came up.
    """
    global _comfy_process, _comfy_boot_seconds, _ranks, _shadow_ranks

    with _start_lock:
        if _ranks and all(
            rank.process is not None and rank.process.poll() is None and rank.is_ready()
            for rank in _ranks
        ):
            return

        for rank in _ranks:
            if rank.process is not None and rank.process.poll() is not None:
                log(
                    f"ComfyUI rank {rank.rank} exited with code {rank.process.returncode}; "
                    "restarting the whole group."
                )
                _stop_ranks()
                break

        # Covers the wait even when we join a start already in flight, so comfy_boot
        # reflects time-to-ready rather than time-since-fork.
        boot_began = time.monotonic()

        if not _ranks or all(rank.process is None for rank in _ranks):
            world = GPU_CONFIG.world_size if (GPU_CONFIG and GPU_CONFIG.dual) else 1
            _ranks = [ComfyRank(index) for index in range(world)]
            _shadow_ranks = [rank for rank in _ranks if rank.shadow]
            for rank in _ranks:
                _launch_rank(rank)
            _comfy_process = _ranks[0].process

        deadline = time.monotonic() + COMFY_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            for rank in _ranks:
                if rank.process is not None and rank.process.poll() is not None:
                    code = rank.process.returncode
                    _stop_ranks()
                    raise RuntimeError(
                        f"ComfyUI rank {rank.rank} exited during startup with code {code}. "
                        "Check the ComfyUI log above for the underlying error."
                    )
            if all(rank.is_ready() for rank in _ranks):
                elapsed = time.monotonic() - boot_began
                _comfy_boot_seconds = elapsed
                log(f"ComfyUI is ready after ~{elapsed:.1f}s ({len(_ranks)} rank(s))")
                if GPU_CONFIG is not None and GPU_CONFIG.dual:
                    _verify_dual_ranks()
                return
            time.sleep(1)

        _stop_ranks()
        raise RuntimeError(f"ComfyUI did not become ready within {COMFY_STARTUP_TIMEOUT}s")


def _stop_ranks() -> None:
    """Terminate every rank. Used on a failed start and on shutdown."""
    global _comfy_process
    for rank in _ranks:
        process = rank.process
        rank.process = None
        if process is None or process.poll() is not None:
            continue
        try:
            process.terminate()
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception as error:  # pragma: no cover - defensive
            log(f"WARNING: could not stop rank {rank.rank}: {error}")
    _comfy_process = None


# --------------------------------------------------------------------------------------
# Workflow execution
# --------------------------------------------------------------------------------------


class WorkflowError(RuntimeError):
    """A workflow was rejected or failed inside ComfyUI (a user/job error, not a bug)."""


def queue_prompt(workflow: dict, client_id: str) -> str:
    # Shadows first. They only ever wait for rank 0, never the other way round, so telling
    # them last would put a needless gap at the front of every job - and if a shadow
    # rejects the graph we want to know before rank 0 has started spending GPU seconds.
    _queue_on_shadows(workflow, client_id)

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )

    if response.status_code >= 400:
        # ComfyUI returns a structured validation report on 400; pass it through verbatim
        # so the caller can see which node/field was rejected.
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise WorkflowError(f"ComfyUI rejected the workflow: {json.dumps(detail, default=str)}")

    payload = response.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise WorkflowError(f"ComfyUI did not return a prompt_id: {payload}")
    return prompt_id


def _queue_on_shadows(workflow: dict, client_id: str) -> None:
    """Give every shadow rank its own copy of the graph, minus the file-producing tail."""
    if not _shadow_ranks:
        return

    shadow, description = shadow_graph.build_shadow_workflow(workflow)
    for rank in _shadow_ranks:
        try:
            response = requests.post(
                f"{rank.url}/prompt",
                json={"prompt": shadow, "client_id": client_id},
                timeout=60,
            )
        except requests.RequestException as error:
            raise WorkflowError(
                f"Could not submit the workflow to rank {rank.rank} ({error}). Rank 0 "
                "would deadlock on the first collective, so the job is refused instead."
            ) from error

        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise WorkflowError(
                f"Rank {rank.rank} rejected the shadow workflow ({description}): "
                f"{json.dumps(detail, default=str)}"
            )
    log(f"Queued the shadow workflow on {len(_shadow_ranks)} rank(s): {description}")


def _drain_shadow_ranks(reason: str) -> None:
    """Interrupt shadows and empty their scratch directories.

    Runs after every job, successful or not. Two jobs' worth of reasons:

      * a shadow that is still executing when the next job arrives would answer the next
        job's collectives with the previous job's tensors; and
      * even with the sink rewrite, an unrecognised output node in a caller's workflow
        could still leave a rendered file on a shadow's disk. Under Confidential
        Generation that would be a second plaintext copy of the customer's video, so it
        is deleted on the same job boundary as rank 0's plaintext.
    """
    for rank in _shadow_ranks:
        try:
            requests.post(f"{rank.url}/interrupt", timeout=10)
        except requests.RequestException as error:
            log(f"WARNING: could not interrupt rank {rank.rank} ({reason}): {error}")
        for directory in (rank.output_dir, rank.temp_dir):
            _empty_directory(directory)


def _empty_directory(path: str) -> None:
    """Remove everything inside a directory this handler owns, keeping the directory."""
    try:
        if not os.path.isdir(path):
            return
        for name in os.listdir(path):
            target = os.path.join(path, name)
            if os.path.islink(target) or os.path.isfile(target):
                os.remove(target)
            elif os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
    except OSError as error:
        log(f"WARNING: could not empty {path}: {error}")


def gpu_perf_fields() -> list[str]:
    """Per-rank peak VRAM for the [perf] line, read from each rank's own /h3/gpu route."""
    if GPU_CONFIG is None or not GPU_CONFIG.dual:
        return []

    fields: list[str] = []
    for rank in _ranks:
        try:
            status = rank.gpu_status(timeout=5)
        except Exception as error:  # pragma: no cover - a metric must never fail a job
            fields.append(f"gpu{rank.rank}_vram=unavailable")
            log(f"WARNING: could not read rank {rank.rank} VRAM: {error}")
            continue
        for device in status.get("devices") or []:
            fields.append(
                f"gpu{rank.rank}_peak_alloc_mb={device.get('peak_allocated_mb')} "
                f"gpu{rank.rank}_peak_reserved_mb={device.get('peak_reserved_mb')}"
            )
    return fields


def reset_gpu_peaks() -> None:
    """Zero every rank's peak-memory counters so the next job's numbers are its own."""
    for rank in _ranks:
        try:
            requests.get(f"{rank.url}/h3/gpu", params={"reset": "1"}, timeout=5)
        except Exception:  # pragma: no cover - best effort
            pass


def get_history(prompt_id: str) -> dict:
    response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=60)
    response.raise_for_status()
    return response.json().get(prompt_id, {})


def _history_is_finished(history: dict) -> bool:
    status = history.get("status") or {}
    return bool(status.get("completed")) or status.get("status_str") in ("success", "error")


def _raise_if_history_failed(history: dict, prompt_id: str) -> None:
    status = history.get("status") or {}
    if status.get("status_str") != "error" and status.get("completed", True):
        return

    for entry in status.get("messages") or []:
        # messages are ["execution_error", {...}] pairs
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == "execution_error":
            detail = entry[1] or {}
            raise WorkflowError(
                "ComfyUI execution failed in node "
                f"{detail.get('node_type')} (#{detail.get('node_id')}): "
                f"{detail.get('exception_type')}: {detail.get('exception_message')}"
            )

    raise WorkflowError(f"ComfyUI reported a failed execution for prompt {prompt_id}: {status}")


# Node classes that mean the graph has moved past sampling into decode/encode work.
DECODE_CLASS_HINTS = ("VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo")


def await_execution(
    prompt_id: str,
    client_id: str,
    deadline: float,
    timer: JobTimer | None = None,
    reporter: "ProgressReporter | None" = None,
) -> dict:
    """Follow a prompt to completion over the websocket, falling back to /history polling.

    A recv timeout is expected and harmless - generation frequently produces no traffic for
    minutes. We only give up when the wall-clock deadline passes. If the socket dies we
    keep polling /history, because the job itself is still running server-side.
    """
    connection = None
    try:
        connection = websocket.create_connection(
            f"ws://{COMFY_HOST}:{COMFY_PORT}/ws?clientId={client_id}",
            timeout=WS_RECV_TIMEOUT,
        )
    except Exception as error:
        log(f"WARNING: websocket connect failed ({error}); falling back to history polling.")

    last_poll = 0.0
    try:
        while True:
            if time.monotonic() > deadline:
                raise WorkflowError(
                    f"Workflow exceeded H3_JOB_TIMEOUT ({JOB_TIMEOUT}s); giving up on "
                    f"prompt {prompt_id}."
                )

            if connection is not None:
                try:
                    message = connection.recv()
                except websocket.WebSocketTimeoutException:
                    message = None  # normal: generation is still running quietly
                except Exception as error:
                    log(f"WARNING: websocket dropped ({error}); continuing via history polling.")
                    connection = None
                    message = None

                if isinstance(message, (bytes, bytearray)):
                    continue  # binary frames are preview images; we disabled previews
                if message:
                    try:
                        event = json.loads(message)
                    except ValueError:
                        event = None

                    if event:
                        data = event.get("data") or {}
                        event_type = event.get("type")

                        if event_type == "progress" and data.get("prompt_id") == prompt_id:
                            value, total = data.get("value"), data.get("max")
                            if timer is not None:
                                # Fed every frame, including value=0, so sampling_start is
                                # the sampler actually beginning rather than step 1 ending.
                                timer.on_progress(value, total)
                            if value and total:
                                log(f"  progress {value}/{total}")
                                if reporter is not None:
                                    reporter.step(int(value), int(total))
                        elif event_type == "execution_error" and data.get("prompt_id") == prompt_id:
                            raise WorkflowError(
                                "ComfyUI execution error in node "
                                f"{data.get('node_type')} (#{data.get('node_id')}): "
                                f"{data.get('exception_type')}: {data.get('exception_message')}"
                            )
                        elif (
                            event_type in ("execution_success", "execution_interrupted")
                            and data.get("prompt_id") == prompt_id
                        ):
                            break
                        elif (
                            event_type == "executing"
                            and data.get("prompt_id") == prompt_id
                            and data.get("node") is None
                        ):
                            break  # legacy completion signal
                        elif (
                            event_type == "executing"
                            and data.get("prompt_id") == prompt_id
                            and timer is not None
                        ):
                            # A named node started, so the previous one just finished.
                            # This is what attributes time to the text encoder, the
                            # sampler and each VAE decode without touching ComfyUI.
                            node_id = str(data.get("node"))
                            timer.on_node(node_id)

                            # Translate the node into a public phase. ComfyUI's own log
                            # text is never forwarded - only this stable vocabulary.
                            if reporter is not None:
                                label = timer._label(node_id)
                                if any(hint in label for hint in DECODE_CLASS_HINTS):
                                    if reporter._phase != PHASE_DECODING:
                                        reporter.phase(PHASE_DECODING, percent=90)

            # Poll /history regardless: it is the authoritative record and covers the case
            # where the websocket never connected or missed the terminal event.
            now = time.monotonic()
            if now - last_poll >= 5:
                last_poll = now
                history = get_history(prompt_id)
                if history and _history_is_finished(history):
                    break
            if connection is None:
                time.sleep(1)
    finally:
        # Closes the decode phase whichever way we left the loop. On a failure path the
        # span is meaningless, which is what status= on the [perf] line is there to say.
        if timer is not None:
            timer.on_execution_end()
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    history = get_history(prompt_id)
    if not history:
        raise WorkflowError(f"ComfyUI returned no history for prompt {prompt_id}.")
    _raise_if_history_failed(history, prompt_id)
    return history


def iter_output_entries(history: dict) -> Iterable[dict]:
    for node_id, node_output in (history.get("outputs") or {}).items():
        if not isinstance(node_output, dict):
            continue
        for key in OUTPUT_KEYS:
            for entry in node_output.get(key) or []:
                if isinstance(entry, dict) and entry.get("filename"):
                    yield {**entry, "node_id": node_id}


def resolve_output_path(entry: dict) -> str | None:
    """Map a history entry to a file on disk, preferring the local path over an HTTP fetch."""
    folder_type = entry.get("type", "output")
    roots = {
        "output": COMFY_OUTPUT_DIR,
        "temp": COMFY_TEMP_DIR,
        "input": COMFY_INPUT_DIR,
    }
    root = roots.get(folder_type, COMFY_OUTPUT_DIR)
    path = os.path.join(root, entry.get("subfolder") or "", entry["filename"])
    return path if os.path.isfile(path) else None


def download_output(entry: dict, destination: str) -> str:
    """Fall back to ComfyUI's /view endpoint when the file is not where we expect it."""
    params = {
        "filename": entry["filename"],
        "subfolder": entry.get("subfolder", ""),
        "type": entry.get("type", "output"),
    }
    with requests.get(f"{COMFY_URL}/view", params=params, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


# --------------------------------------------------------------------------------------
# Input image staging (first-frame image-to-video)
# --------------------------------------------------------------------------------------

# Nodes that read a file from ComfyUI's input directory via an "image" widget. Native
# LoadImage and both Pixaroma loaders all use the same field name.
IMAGE_LOADER_CLASS_TYPES = frozenset(
    {"LoadImage", "LoadImageMask", "LoadImageOutput", "PixaromaLoadImage", "PixaromaLoadImageMini"}
)
IMAGE_LOADER_FIELD = "image"

MAX_IMAGE_BYTES = int(os.environ.get("H3_MAX_IMAGE_BYTES", str(32 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("H3_MAX_IMAGE_PIXELS", str(64_000_000)))
IMAGE_DOWNLOAD_TIMEOUT = int(os.environ.get("H3_IMAGE_TIMEOUT", "30"))
MAX_IMAGE_REDIRECTS = 3

# Magic-byte signatures for the formats we accept. Extension is derived from the actual
# bytes, never from the URL or a caller-supplied name.
_IMAGE_SIGNATURES = (
    ("png", ".png", lambda d: d.startswith(b"\x89PNG\r\n\x1a\n")),
    ("jpeg", ".jpg", lambda d: d.startswith(b"\xff\xd8\xff")),
    ("webp", ".webp", lambda d: d[:4] == b"RIFF" and d[8:12] == b"WEBP"),
)


class ImageInputError(WorkflowError):
    """The caller's image input was missing, malformed, unsupported or unsafe."""


def _detect_image_type(data: bytes) -> tuple[str, str]:
    for kind, extension, matches in _IMAGE_SIGNATURES:
        if matches(data):
            return kind, extension
    raise ImageInputError(
        "Unsupported image format. Supported formats are PNG, JPEG and WebP."
    )


def _validate_image_bytes(data: bytes) -> str:
    """Confirm the bytes really are a supported image. Returns the file extension."""
    if not data:
        raise ImageInputError("The supplied image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            f"Image is {len(data)} bytes, over the {MAX_IMAGE_BYTES}-byte limit "
            "(H3_MAX_IMAGE_BYTES)."
        )

    kind, extension = _detect_image_type(data)

    # Magic bytes alone are trivial to forge, so also make Pillow parse it. verify()
    # decodes structure without materialising full pixel data.
    try:
        import io as _io

        from PIL import Image

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        with Image.open(_io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageInputError(
                    f"Image is {width}x{height} ({width * height} pixels), over the "
                    f"{MAX_IMAGE_PIXELS}-pixel limit (H3_MAX_IMAGE_PIXELS)."
                )
            image.verify()
    except ImageInputError:
        raise
    except Exception as error:
        raise ImageInputError(f"The supplied data is not a decodable image: {error}") from error

    log(f"Accepted {kind} input image ({len(data)} bytes, {width}x{height})")
    return extension


def _assert_fetchable_url(url: str) -> None:
    """Reject non-HTTPS and anything resolving to a non-public address (SSRF guard)."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)

    # HTTPS only. http is available strictly as an opt-in escape hatch for trusted
    # internal testing, never by default.
    allow_insecure = os.environ.get("H3_ALLOW_INSECURE_IMAGE_URL") == "1"
    allowed_schemes = {"https", "http"} if allow_insecure else {"https"}
    if parsed.scheme not in allowed_schemes:
        raise ImageInputError(
            f"image_url must use https (got {parsed.scheme or 'no'!r} scheme)."
        )
    if not parsed.hostname:
        raise ImageInputError("image_url has no host.")

    default_port = 443 if parsed.scheme == "https" else 80
    try:
        resolved = socket.getaddrinfo(
            parsed.hostname, parsed.port or default_port, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as error:
        raise ImageInputError(f"Could not resolve image_url host: {error}") from error

    for entry in resolved:
        address = ipaddress.ip_address(entry[4][0])
        # Covers 127.0.0.0/8, ::1, 10/8, 172.16/12, 192.168/16, 169.254/16, fc00::/7,
        # multicast and the reserved ranges in one shot.
        if (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ImageInputError(
                f"image_url resolves to the non-public address {address}, which is not allowed."
            )


def _download_image(url: str, bearer_token: str | None = None) -> bytes:
    """Fetch an image over HTTPS, validating every redirect hop and capping the size.

    `bearer_token` is the job-scoped credential for the Worker's internal asset route. It
    is sent only on the first hop: a redirect can point anywhere, and forwarding the token
    across it would hand a job credential to whatever host the redirect names.
    """
    current = url
    for hop in range(MAX_IMAGE_REDIRECTS + 1):
        _assert_fetchable_url(current)

        headers = {"Accept": "image/*"}
        if bearer_token and hop == 0:
            # Both credentials are first-hop only. A redirect can point anywhere, and
            # forwarding either one across it would hand a credential to that host.
            headers["Authorization"] = f"Bearer {bearer_token}"
            headers.update(cloudflare_access_headers())

        # Redirects are followed manually so each destination is re-validated; letting
        # requests follow them would let a redirect land on an internal address.
        response = requests.get(
            current,
            timeout=IMAGE_DOWNLOAD_TIMEOUT,
            stream=True,
            allow_redirects=False,
            headers=headers,
        )

        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ImageInputError("Redirect response had no Location header.")
            if hop == MAX_IMAGE_REDIRECTS:
                raise ImageInputError(f"image_url exceeded {MAX_IMAGE_REDIRECTS} redirects.")
            current = requests.compat.urljoin(current, location)
            continue

        with response:
            if not response.ok:
                raise ImageInputError(f"Downloading image_url returned HTTP {response.status_code}.")

            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_IMAGE_BYTES:
                raise ImageInputError(
                    f"image_url advertises {declared} bytes, over the "
                    f"{MAX_IMAGE_BYTES}-byte limit (H3_MAX_IMAGE_BYTES)."
                )

            chunks = bytearray()
            for chunk in response.iter_content(chunk_size=256 * 1024):
                chunks.extend(chunk)
                # Enforce the cap while streaming: a lying Content-Length must not let an
                # unbounded body into memory.
                if len(chunks) > MAX_IMAGE_BYTES:
                    raise ImageInputError(
                        f"image_url body exceeded the {MAX_IMAGE_BYTES}-byte limit "
                        "(H3_MAX_IMAGE_BYTES)."
                    )
            return bytes(chunks)

    raise ImageInputError("Too many redirects while fetching image_url.")


def _decode_base64_image(value: str) -> bytes:
    """Decode base64 image input, accepting an optional data: URI prefix.

    The payload itself is never logged.
    """
    if not isinstance(value, str) or not value.strip():
        raise ImageInputError("image_base64 must be a non-empty string.")

    payload = value.strip()
    if payload.startswith("data:"):
        header, _, remainder = payload.partition(",")
        if not remainder or "base64" not in header:
            raise ImageInputError("image_base64 data URI must be base64-encoded.")
        payload = remainder

    payload = "".join(payload.split())
    # Reject before decoding: base64 inflates by 4/3, so this bounds the decoded size too.
    if len(payload) > (MAX_IMAGE_BYTES // 3) * 4 + 4:
        raise ImageInputError(
            f"image_base64 decodes to more than the {MAX_IMAGE_BYTES}-byte limit "
            "(H3_MAX_IMAGE_BYTES)."
        )

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageInputError(f"image_base64 is not valid base64: {error}") from error


def _reachable_image_loaders(workflow: dict, start_node: str) -> set[str]:
    """Walk a workflow's links backwards from a node, collecting image-loader ancestors.

    The bundled first-frame workflow wires the loader through an intermediate resize node
    (PixaromaLoadImageMini -> PixaromaLongestSide -> first_frame), so a direct-edge check
    would miss it.
    """
    found: set[str] = set()
    seen: set[str] = set()
    queue = [start_node]

    while queue:
        node_id = queue.pop()
        if node_id in seen or node_id not in workflow:
            continue
        seen.add(node_id)

        node = workflow[node_id]
        if not isinstance(node, dict):
            continue

        if node.get("class_type") in IMAGE_LOADER_CLASS_TYPES:
            found.add(node_id)
            continue  # the loader is the source; nothing upstream of it matters

        for value in (node.get("inputs") or {}).values():
            # An API-format link is ["<source node id>", <output slot int>]. The slot check
            # keeps a two-element list of widget strings from being mistaken for a link.
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], (str, int))
                and isinstance(value[1], int)
                and not isinstance(value[1], bool)
            ):
                queue.append(str(value[0]))

    return found


def find_image_node(workflow: dict, explicit_id: str | None, role: str = "first_frame") -> str:
    """Decide which node receives a staged image, or fail rather than guess.

    `role` is the MiniMaxH3ImageToVideo input the image is destined for - "first_frame" or
    "last_frame", the two optional Image inputs on that node's schema. With two keyframes
    in one graph the workflow has two loaders, so "the only image loader" is no longer a
    usable tiebreak and the role is what disambiguates them.
    """
    candidates = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict)
        and node.get("class_type") in IMAGE_LOADER_CLASS_TYPES
        and IMAGE_LOADER_FIELD in (node.get("inputs") or {})
    ]

    # Treat an empty/blank image_node_id as "not supplied" rather than as node "".
    if explicit_id is not None and str(explicit_id).strip():
        node_id = str(explicit_id).strip()
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise ImageInputError(f"image_node_id {node_id!r} is not a node in the workflow.")
        if IMAGE_LOADER_FIELD not in (node.get("inputs") or {}):
            raise ImageInputError(
                f"image_node_id {node_id!r} is a {node.get('class_type')!r} node with no "
                f"{IMAGE_LOADER_FIELD!r} input, so an image cannot be attached to it."
            )
        return node_id

    if not candidates:
        raise ImageInputError(
            "An image was supplied but the workflow has no image-loader node "
            f"({', '.join(sorted(IMAGE_LOADER_CLASS_TYPES))}). Add one, or omit the image "
            "for text-to-video."
        )

    # Prefer a loader that actually feeds MiniMaxH3ImageToVideo.<role>.
    preferred: set[str] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "MiniMaxH3ImageToVideo":
            continue
        link = (node.get("inputs") or {}).get(role)
        if isinstance(link, list) and len(link) == 2:
            preferred |= _reachable_image_loaders(workflow, str(link[0]))

    if len(preferred) == 1:
        node_id = next(iter(preferred))
        log(f"Image node auto-detected via MiniMaxH3ImageToVideo.{role}: #{node_id}")
        return node_id

    # Falling back to "the only image loader" is only safe when that loader is not
    # already committed to a different keyframe. Otherwise a last_frame request on a
    # first-frame graph would silently attach the image to the wrong end of the clip.
    claimed_by_other_role: set[str] = set()
    for other_role in KEYFRAME_ROLES:
        if other_role == role:
            continue
        for node in workflow.values():
            if not isinstance(node, dict) or node.get("class_type") != "MiniMaxH3ImageToVideo":
                continue
            link = (node.get("inputs") or {}).get(other_role)
            if isinstance(link, list) and len(link) == 2:
                claimed_by_other_role |= _reachable_image_loaders(workflow, str(link[0]))

    unclaimed = [node_id for node_id in candidates if node_id not in claimed_by_other_role]

    if len(unclaimed) == 1 and not preferred:
        log(f"Image node auto-detected as the workflow's only image loader: #{unclaimed[0]}")
        return unclaimed[0]

    if not unclaimed and candidates:
        raise ImageInputError(
            f"No image loader is available for {role!r}: the workflow's "
            f"{len(candidates)} loader(s) already feed another keyframe input. Add a "
            f"loader wired to MiniMaxH3ImageToVideo.{role}."
        )

    raise ImageInputError(
        f"Could not unambiguously choose an image node for {role!r}: found "
        f"{len(candidates)} image loaders ({', '.join(sorted(candidates))})"
        + (f" and {len(preferred)} feeding {role}" if preferred else "")
        + ". Wire the workflow so exactly one loader feeds "
        f"MiniMaxH3ImageToVideo.{role}, or pass image_node_id (legacy) / "
        f"assets.{role}.node_id to say which one should receive the image."
    )


# The keyframe roles the FL2VA node exposes, in the order they are staged. Both are
# optional Image inputs on MiniMaxH3ImageToVideo's schema.
KEYFRAME_ROLES = ("first_frame", "last_frame")


def _write_staged_image(data: bytes, extension: str) -> str:
    """Write image bytes into the ComfyUI input directory under a generated name."""
    # Generated filename only: the remote/user-supplied name is never used, which rules
    # out path traversal and collisions between concurrent jobs.
    filename = f"h3-input-{uuid.uuid4().hex}{extension}"
    input_dir = os.path.realpath(COMFY_INPUT_DIR)
    os.makedirs(input_dir, exist_ok=True)
    path = os.path.realpath(os.path.join(input_dir, filename))

    # Defence in depth: the name is generated, but assert containment anyway.
    if os.path.dirname(path) != input_dir:
        raise ImageInputError("Refusing to write the input image outside the input directory.")

    with open(path, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o644)  # data, never executable
    return path


def _fetch_asset_bytes(spec: dict, role: str) -> bytes:
    """Read one asset's bytes from whichever source it declares.

    `url` covers both a plain https image and the Worker's job-scoped asset route - the
    latter simply carries a bearer token, so no Cloudflare credentials ever live in this
    image. Exactly one source may be given.
    """
    url = spec.get("url")
    data_b64 = spec.get("base64")

    if url and data_b64:
        raise ImageInputError(f"{role}: provide either url or base64, not both.")

    if url:
        if not isinstance(url, str):
            raise ImageInputError(f"{role}: url must be a string.")
        token = spec.get("token")
        if token is not None and not isinstance(token, str):
            raise ImageInputError(f"{role}: token must be a string.")
        log(f"Fetching {role} from url")
        return _download_image(url, bearer_token=token)

    if data_b64:
        log(f"Decoding {role} from base64")
        return _decode_base64_image(data_b64)

    raise ImageInputError(f"{role}: no url or base64 supplied.")


def _collect_asset_specs(job_input: dict) -> dict:
    """Normalise the legacy single-image fields and the new `assets` map into one dict.

    Legacy `image_url` / `image_base64` / `image_node_id` keep working untouched and are
    treated as a first_frame asset, which is what they always effectively were.
    """
    specs: dict[str, dict] = {}

    assets = job_input.get("assets")
    if assets is not None:
        if not isinstance(assets, dict):
            raise ImageInputError("assets must be an object keyed by role.")
        for role, spec in assets.items():
            if role not in KEYFRAME_ROLES:
                raise ImageInputError(
                    f"Unknown asset role {role!r}. Supported: {', '.join(KEYFRAME_ROLES)}."
                )
            if not isinstance(spec, dict):
                raise ImageInputError(f"assets.{role} must be an object.")
            specs[role] = spec

    legacy_url = job_input.get("image_url")
    legacy_b64 = job_input.get("image_base64")
    if legacy_url or legacy_b64:
        if "first_frame" in specs:
            raise ImageInputError(
                "Provide either the legacy image_url/image_base64 fields or "
                "assets.first_frame, not both."
            )
        if legacy_url and legacy_b64:
            raise ImageInputError("Provide either image_url or image_base64, not both.")
        specs["first_frame"] = {
            "url": legacy_url,
            "base64": legacy_b64,
            "node_id": job_input.get("image_node_id"),
        }

    return specs


def stage_input_assets(job_input: dict, workflow: dict) -> list[str]:
    """Validate, store and wire up every supplied keyframe.

    Returns the absolute paths of the staged files, for cleanup. An empty list is the
    text-to-video path, whose behaviour is unchanged.

    Everything is resolved and validated before anything is written, so a request we
    cannot wire up leaves no files behind.
    """
    specs = _collect_asset_specs(job_input)
    if not specs:
        return []  # text-to-video: unchanged behaviour

    # Resolve every target node first - a half-staged graph is worse than a rejected one.
    resolved: list[tuple[str, dict, str]] = []
    for role in KEYFRAME_ROLES:
        spec = specs.get(role)
        if spec is None:
            continue
        node_id = find_image_node(workflow, spec.get("node_id"), role=role)
        resolved.append((role, spec, node_id))

    if len({node_id for _, _, node_id in resolved}) != len(resolved):
        raise ImageInputError(
            "Two keyframes resolved to the same image loader node. Wire the workflow so "
            "first_frame and last_frame each have their own loader."
        )

    staged: list[str] = []
    try:
        for role, spec, node_id in resolved:
            data = _fetch_asset_bytes(spec, role)
            extension = _validate_image_bytes(data)
            path = _write_staged_image(data, extension)
            staged.append(path)
            filename = os.path.basename(path)
            workflow[node_id]["inputs"][IMAGE_LOADER_FIELD] = filename
            log(f"Staged {role} as {filename} and wired it into node #{node_id}")
    except Exception:
        # Never leak files from a partially staged multi-image request.
        for path in staged:
            cleanup_input_image(path)
        raise

    return staged


def stage_input_image(job_input: dict, workflow: dict) -> str | None:
    """Backwards-compatible single-image wrapper around stage_input_assets()."""
    staged = stage_input_assets(job_input, workflow)
    return staged[0] if staged else None


def cleanup_input_image(path: str | None) -> None:
    """Remove a staged input image. Only ever touches files this handler created."""
    if not path:
        return
    try:
        input_dir = os.path.realpath(COMFY_INPUT_DIR)
        resolved = os.path.realpath(path)
        basename = os.path.basename(resolved)
        # Belt and braces: only delete our own generated names, inside the input dir.
        if os.path.dirname(resolved) != input_dir or not basename.startswith("h3-input-"):
            log(f"WARNING: refusing to clean up unexpected path {resolved}")
            return
        if os.path.isfile(resolved):
            os.remove(resolved)
            log(f"Removed staged input image {basename}")
    except OSError as error:
        log(f"WARNING: could not remove staged input image: {error}")


# --------------------------------------------------------------------------------------
# Output storage
# --------------------------------------------------------------------------------------


class OutputStore:
    """Strategy for getting a finished artefact back to the caller.

    ``store`` receives a path to a file on the worker and returns the dict that goes into
    ``output.images[]``. Implementations own their own cleanup of that file.

    ``protected`` describes what that file now *is* - plaintext in standard mode, an
    encrypted container in confidential mode. It is optional so that every existing caller
    keeps working untouched; a store that ignores it behaves exactly as it did before.
    """

    def store(self, path: str, entry: dict, protected: ProtectedArtifact | None = None) -> dict:
        raise NotImplementedError


class Base64Store(OutputStore):
    """Inline the artefact as base64 - the format the current Cloudflare Worker expects."""

    def __init__(self) -> None:
        self.max_bytes = int(os.environ.get("H3_MAX_BASE64_BYTES", str(180 * 1024 * 1024)))

    def store(self, path: str, entry: dict, protected: ProtectedArtifact | None = None) -> dict:
        size = os.path.getsize(path)
        if size > self.max_bytes:
            raise WorkflowError(
                f"{entry['filename']} is {size} bytes, over H3_MAX_BASE64_BYTES "
                f"({self.max_bytes}). Switch H3_OUTPUT_MODE to 'r2' for large videos."
            )
        with open(path, "rb") as handle:
            data = base64.b64encode(handle.read()).decode("ascii")
        return {
            "filename": entry["filename"],
            "subfolder": entry.get("subfolder", ""),
            "type": entry.get("type", "output"),
            "data": data,
        }


class WorkerUploadStore(OutputStore):
    """PUT the raw MP4 to the Cloudflare Worker, which streams it straight into R2.

    The bytes go up as a normal binary body - never base64 - so nothing inflates the
    payload by a third just to move it. Authentication is a job-scoped bearer token that
    arrived with this job, which is why no Cloudflare credential is baked into the image.

    Returns metadata only. The video itself is served later from R2 by the Worker.
    """

    def __init__(self, url: str, token: str | None, timeout: float) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout
        self.uploaded_bytes = 0
        self.upload_seconds = 0.0

    def store(self, path: str, entry: dict, protected: ProtectedArtifact | None = None) -> dict:
        size = os.path.getsize(path)
        # The content type describes what is actually in the body. For a confidential
        # artefact that is an opaque container, not an MP4, and saying so is what stops the
        # Worker from ever labelling ciphertext as playable video.
        content_type = protected.content_type if protected else "video/mp4"
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(size),
            **cloudflare_access_headers(),
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        began = time.monotonic()
        try:
            # A file object streams; requests will not read it all into memory first.
            with open(path, "rb") as handle:
                # allow_redirects=False: on a 302 requests would re-issue as a GET without
                # the body, so the video would silently never be sent.
                response = requests.put(
                    self.url,
                    data=handle,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
        except Exception as error:
            raise WorkflowError(
                f"Uploading {entry['filename']} to the output endpoint failed: {error}"
            ) from error
        finally:
            self.upload_seconds += time.monotonic() - began

        if not _is_2xx(response):
            # Deliberately fatal. Reporting success here would hand the client a job that
            # says COMPLETED with no playable video behind it.
            raise WorkflowError(
                f"Output upload failed for {entry['filename']} - {_describe_non_2xx(response)}"
            )

        # The Worker owns key naming; echo back whatever it reports rather than inventing
        # a key here, so RunPod can never dictate where an object lands.
        try:
            body = response.json()
        except ValueError:
            body = None

        # Require our own acknowledgement, not merely a 2xx. Anything sitting in front of
        # the Worker - an auth gateway, a proxy, an error page - can answer 200 with HTML
        # while storing nothing, and a job that claims a video it does not have is the
        # single worst outcome this path can produce.
        if not isinstance(body, dict) or not body.get("key"):
            raise WorkflowError(
                f"Output upload for {entry['filename']} returned HTTP "
                f"{response.status_code} but not the Worker's JSON acknowledgement "
                f"(expected a 'key'). Got: {response.text[:200]!r}. The upload did not "
                "reach R2."
            )

        self.uploaded_bytes += size

        result = {
            "filename": entry["filename"],
            "subfolder": entry.get("subfolder", ""),
            "type": entry.get("type", "output"),
            "size": size,
        }
        for field in ("key", "url", "contentType"):
            if body.get(field):
                result[field] = body[field]

        if protected is not None:
            # Non-secret protection facts, so /status can describe the artefact without the
            # client having to download it first. There is no key material in here - see
            # ProtectedArtifact.metadata, which is built from public fields only.
            result["privacyMode"] = protected.metadata.get("privacyMode", "standard")
            result["encrypted"] = protected.encrypted
            if protected.encrypted:
                result["artifact"] = dict(protected.metadata)

        return result


class EncryptedR2Store(OutputStore):
    """Encrypt each artefact with a fresh AES-256-GCM key, then upload ciphertext only.

    Envelope encryption: a per-video data key (DEK) never leaves the worker in the clear.
    It is wrapped with a long-lived key-encryption key (KEK) supplied out of band, and only
    the wrapped form is returned. Plaintext never reaches R2, and neither key material nor
    plaintext is ever logged.
    """

    ALGORITHM = "AES-256-GCM"
    VERSION = "1"
    WRAP_ALGORITHM = "AES-256-GCM"

    def __init__(self) -> None:
        self.bucket = self._require("R2_BUCKET")
        self.prefix = os.environ.get("R2_PREFIX", "").strip("/")
        account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
        endpoint = os.environ.get("R2_ENDPOINT", "").strip()
        if not endpoint:
            if not account_id:
                raise RuntimeError("Set R2_ENDPOINT or R2_ACCOUNT_ID for H3_OUTPUT_MODE=r2.")
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

        self._kek, self._kek_id = self._load_kek()

        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:  # pragma: no cover - boto3 ships in the image
            raise RuntimeError(f"H3_OUTPUT_MODE=r2 requires boto3: {error}") from error

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=self._require("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=self._require("R2_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("R2_REGION", "auto"),
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        )
        # Log only non-secret configuration.
        log(f"Output mode: encrypted R2 (bucket={self.bucket}, kek_id={self._kek_id})")

    @staticmethod
    def _require(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"H3_OUTPUT_MODE=r2 requires the {name} environment variable.")
        return value

    def _load_kek(self) -> tuple[bytes, str]:
        """Load the key-encryption key used to wrap per-video data keys.

        Fails closed: without a KEK we would have to hand back a raw data key, so we refuse
        to start instead.
        """
        raw = os.environ.get("H3_KEY_WRAP_KEY", "").strip()
        if not raw:
            raise RuntimeError(
                "H3_OUTPUT_MODE=r2 requires H3_KEY_WRAP_KEY (base64 32-byte AES-256 key) so "
                "per-video keys can be wrapped. Refusing to start rather than return raw keys."
            )
        try:
            key = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as error:
            raise RuntimeError(f"H3_KEY_WRAP_KEY must be valid base64: {error}") from error
        if len(key) != 32:
            raise RuntimeError(
                f"H3_KEY_WRAP_KEY must decode to exactly 32 bytes, got {len(key)}."
            )
        return key, os.environ.get("H3_KEY_WRAP_KEY_ID", "default")

    def _wrap_key(self, data_key: bytes, object_key: str) -> dict:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        wrap_iv = secrets.token_bytes(12)
        # Bind the wrapped key to its object so a wrapped key cannot be replayed onto a
        # different R2 object.
        sealed = AESGCM(self._kek).encrypt(wrap_iv, data_key, object_key.encode("utf-8"))
        return {
            "wrap_algorithm": self.WRAP_ALGORITHM,
            "wrapped_key": base64.b64encode(sealed).decode("ascii"),
            "wrapped_key_iv": base64.b64encode(wrap_iv).decode("ascii"),
            "wrapped_key_aad": "r2_key",
            "key_id": self._kek_id,
        }

    @staticmethod
    def _encrypt_to_file(source: str, destination: str, data_key: bytes, iv: bytes) -> tuple[int, str, str]:
        """Stream-encrypt source -> destination. Returns (size, sha256, base64 tag).

        Streaming (rather than AESGCM.encrypt on a whole buffer) keeps memory flat for
        long videos. The GCM tag is returned separately; the R2 object is pure ciphertext.
        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        encryptor = Cipher(algorithms.AES(data_key), modes.GCM(iv)).encryptor()
        digest = hashlib.sha256()
        size = 0

        with open(source, "rb") as plaintext, open(destination, "wb") as ciphertext:
            while True:
                chunk = plaintext.read(4 * 1024 * 1024)
                if not chunk:
                    break
                sealed = encryptor.update(chunk)
                if sealed:
                    ciphertext.write(sealed)
                    digest.update(sealed)
                    size += len(sealed)
            sealed = encryptor.finalize()
            if sealed:
                ciphertext.write(sealed)
                digest.update(sealed)
                size += len(sealed)

        return size, digest.hexdigest(), base64.b64encode(encryptor.tag).decode("ascii")

    @staticmethod
    def _shred(path: str) -> None:
        """Best-effort removal of a plaintext file."""
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as error:
            log(f"WARNING: could not delete temporary file: {error}")

    def store(self, path: str, entry: dict, protected: ProtectedArtifact | None = None) -> dict:
        filename = entry["filename"]
        object_key = "/".join(
            part for part in (self.prefix, f"{uuid.uuid4().hex}", filename) if part
        )

        # Fresh CSPRNG key and 96-bit nonce for every single video.
        data_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(12)

        ciphertext_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.enc")
        try:
            size, ciphertext_sha256, tag = self._encrypt_to_file(path, ciphertext_path, data_key, iv)
            wrapped = self._wrap_key(data_key, object_key)

            with open(ciphertext_path, "rb") as body:
                self._client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=body,
                    ContentType="application/octet-stream",
                    ContentLength=size,
                    Metadata={
                        "encryption-algorithm": self.ALGORITHM,
                        "encryption-version": self.VERSION,
                        "key-id": wrapped["key_id"],
                    },
                )
            log(f"Uploaded encrypted object {object_key} ({size} bytes ciphertext)")
        finally:
            # Requirement: no plaintext (and no stray ciphertext) survives on the worker.
            self._shred(ciphertext_path)
            self._shred(path)
            del data_key

        return {
            "filename": filename,
            "subfolder": entry.get("subfolder", ""),
            "type": entry.get("type", "output"),
            "storage": "r2",
            "bucket": self.bucket,
            "r2_key": object_key,
            "ciphertext_bytes": size,
            "ciphertext_sha256": ciphertext_sha256,
            "encryption": {
                "algorithm": self.ALGORITHM,
                "version": self.VERSION,
                "iv": base64.b64encode(iv).decode("ascii"),
                "tag": tag,
                "tag_included_in_ciphertext": False,
                **wrapped,
            },
        }


def build_output_store() -> OutputStore:
    if OUTPUT_MODE == "r2":
        return EncryptedR2Store()
    if OUTPUT_MODE == "base64":
        return Base64Store()
    raise RuntimeError(f"Unknown H3_OUTPUT_MODE {OUTPUT_MODE!r}; expected 'base64' or 'r2'.")


_output_store: OutputStore | None = None


def get_output_store() -> OutputStore:
    global _output_store
    if _output_store is None:
        _output_store = build_output_store()
    return _output_store


def collect_outputs(
    history: dict,
    store: OutputStore | None = None,
    protector: artifacts.ArtifactProtector | None = None,
    generation_id: str = "",
) -> list[dict]:
    """Turn a finished execution into the artefacts the caller gets back.

    Two separable steps per artefact, in this order and never the other way round:

      1. **protect** - encrypt it, if the job asked for that. The plaintext is destroyed
         here, while the bytes are still inside this process.
      2. **store**   - hand the protected bytes to whatever delivers them.

    Protection failing is fatal by construction: ``protect()`` raises, the exception
    propagates, and no ``store()`` call is ever reached. There is no path through this
    function that uploads a plaintext artefact for a job that asked for encryption.
    """
    # `store` is injected per job so an R2-backed request can upload through the Worker
    # while everything else keeps using the process-wide configured store.
    store = store if store is not None else get_output_store()
    protector = protector if protector is not None else artifacts.PassthroughProtector()
    results: list[dict] = []
    scratch = tempfile.mkdtemp(prefix="h3-out-")
    keep_outputs = os.environ.get("H3_KEEP_OUTPUTS", "0") == "1"
    protected_artifacts: list[ProtectedArtifact] = []

    try:
        for entry in iter_output_entries(history):
            path = resolve_output_path(entry)
            if path is None:
                path = download_output(entry, os.path.join(scratch, entry["filename"]))

            # Confidential mode's contract includes destroying the plaintext, and this is
            # the only place that deletes a file without the directory guard
            # _discard_output_file applies. So the guard moves here instead: an artefact
            # that resolved outside a directory we own is refused rather than encrypted
            # and shredded. `path` is built from ComfyUI's own history entry, which is
            # trustworthy in normal operation and is exactly the wrong thing to trust when
            # the next step is an unconditional delete.
            if protector.encrypts and not _is_owned_artifact(path, scratch):
                raise WorkflowError(
                    f"Refusing to encrypt {entry['filename']}: it resolved outside this "
                    "worker's own output directories, and confidential mode would have to "
                    "delete it."
                )

            try:
                protected = protector.protect(
                    GeneratedArtifact(
                        path=path,
                        mime_type=_artifact_mime_type(entry["filename"]),
                        generation_id=generation_id or entry["filename"],
                        filename=entry["filename"],
                    )
                )
            except ArtifactError as error:
                # A protection failure is the caller's problem to see, not an unexplained
                # crash: it means their key was unusable or the artefact could not be
                # sealed. Either way `store()` is never reached, so nothing is uploaded -
                # the exception type here changes how the failure reads, never whether it
                # fails closed.
                raise WorkflowError(str(error)) from error

            protected_artifacts.append(protected)

            results.append(store.store(protected.path, entry, protected))

            # A warm worker serves many jobs; without this, generated videos pile up in the
            # output directory until the container disk fills. In confidential mode the
            # plaintext is already gone (protect() shredded it), so this finds nothing -
            # and H3_KEEP_OUTPUTS is deliberately not consulted there, because "keep the
            # outputs for debugging" must never mean "keep the plaintext of an artefact the
            # caller asked us to encrypt".
            if not keep_outputs or protected.encrypted:
                _discard_output_file(path)
    finally:
        # Runs on every path, including a failed encryption or a failed upload: no
        # ciphertext scratch directory and no downloaded plaintext outlives this call.
        for protected in protected_artifacts:
            protected.discard()
        shutil.rmtree(scratch, ignore_errors=True)

    return results


#: Filename extension -> media type of the *plaintext* artefact. Recorded in the encrypted
#: container so a decrypting client knows what it is holding without guessing.
_ARTIFACT_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def _artifact_mime_type(filename: str) -> str:
    return _ARTIFACT_MIME_TYPES.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")


def _is_owned_artifact(path: str, scratch: str) -> bool:
    """Is this file somewhere this worker put it?

    The same question _discard_output_file asks, plus the per-job scratch directory that
    download_output writes into.
    """
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return False
    roots = (COMFY_OUTPUT_DIR, COMFY_TEMP_DIR, scratch)
    return any(
        resolved.startswith(os.path.realpath(root) + os.sep) for root in roots if root
    )


def _discard_output_file(path: str) -> None:
    """Delete a generated artefact, but only from directories we own."""
    try:
        resolved = os.path.realpath(path)
        owned = [os.path.realpath(d) for d in (COMFY_OUTPUT_DIR, COMFY_TEMP_DIR)]
        if not any(resolved.startswith(root + os.sep) for root in owned):
            return  # never touch inputs, models or anything outside our output dirs
        if os.path.isfile(resolved):
            os.remove(resolved)
    except OSError as error:
        log(f"WARNING: could not remove output file: {error}")


# --------------------------------------------------------------------------------------
# RunPod entrypoint
# --------------------------------------------------------------------------------------


def _spec(job_input: dict, key: str) -> dict:
    """Read a job-scoped {url, token} block out of the job payload."""
    value = job_input.get(key)
    return value if isinstance(value, dict) else {}


def build_protector_for_job(job_input: dict) -> artifacts.ArtifactProtector:
    """Resolve this job's privacy mode into an artefact protector.

    Validated here as well as in the Worker, on purpose. The Worker is the public contract
    and rejects bad input early with a clean 4xx; this is the last line before plaintext
    would be handed to an uploader, and it is the only one that matters if a job ever
    reaches RunPod by another route.
    """
    privacy = _spec(job_input, "privacy")
    mode = privacy.get("mode") or job_input.get("privacyMode") or job_input.get("privacy_mode")
    encryption = _spec(job_input, "encryption")
    try:
        return artifacts.build_protector(mode, encryption, on_warning=lambda m: log(f"WARNING: {m}"))
    except ArtifactError as error:
        # Surfaces as a caller-facing 4xx-ish error rather than a crash, and - because
        # WorkflowError is raised before any output is touched - nothing is uploaded.
        raise WorkflowError(str(error)) from error


def build_output_store_for_job(job_input: dict) -> tuple[OutputStore, WorkerUploadStore | None]:
    """Pick where this job's artefacts go.

    A job carrying an `output` block is uploaded straight to the Worker, which streams it
    into R2. Anything else falls back to the configured store, so existing callers that
    know nothing about R2 keep getting base64 exactly as before.
    """
    output = _spec(job_input, "output")
    url = output.get("url")
    if url:
        store = WorkerUploadStore(
            url=str(url),
            token=output.get("token"),
            timeout=float(output.get("timeout") or OUTPUT_UPLOAD_TIMEOUT),
        )
        return store, store
    return get_output_store(), None


def handler(job: dict) -> dict:
    job_id = job.get("id", "<unknown>")
    staged_assets: list[str] = []
    protector: artifacts.ArtifactProtector | None = None

    global _jobs_served
    with _jobs_lock:
        _jobs_served += 1
        job_index = _jobs_served

    job_input = job.get("input") or {}
    workflow = job_input.get("workflow")
    timer = JobTimer(workflow if isinstance(workflow, dict) else None)
    status = "error"

    # Job-scoped progress endpoint. Absent means "no realtime channel", and everything
    # below then behaves exactly as it did before callbacks existed.
    progress = _spec(job_input, "progress")
    reporter = ProgressReporter(
        job_id=str(progress.get("jobId") or job_id),
        url=progress.get("url"),
        token=progress.get("token"),
    )

    try:
        if not isinstance(workflow, dict) or not workflow:
            status = "bad_request"
            return {"error": "input.workflow is required and must be a ComfyUI API-format object."}

        # Resolved before anything expensive starts: a job that cannot be protected the way
        # it asked should cost zero GPU seconds, not fail after two minutes of sampling.
        protector = build_protector_for_job(job_input)
        # The id the artefact is filed under. Taken from the output block first, because
        # that is the one the storage key is derived from: an encrypted container records
        # this id in its authenticated header, and the upload endpoint refuses a container
        # whose header names a different job.
        generation_id = str(
            _spec(job_input, "output").get("jobId")
            or _spec(job_input, "progress").get("jobId")
            or job_id
        )
        timer.privacy_mode = protector.mode
        log(
            f"generation_id={generation_id} privacy_mode={protector.mode} "
            f"encryption={'AES-256-GCM' if protector.encrypts else 'none'} "
            f"nodes={len(workflow)}"
        )

        if protector.encrypts and not _spec(job_input, "output").get("url"):
            # Fail closed. Without an upload endpoint the only way back is base64 inside
            # this job's RunPod result, which is a copy of the artefact in a store we do not
            # control. Refusing is the honest answer; silently returning the plaintext MP4
            # would be the dangerous one.
            raise WorkflowError(
                "privacyMode 'confidential' requires an output upload endpoint. This job "
                "arrived without one, and returning the artefact inline is not an option "
                "for a confidential generation."
            )

        reporter.phase(PHASE_STARTING)

        # Optional keyframes. With none supplied this is a no-op and the text-to-video
        # path runs exactly as before. Done before starting ComfyUI so a bad image is
        # rejected immediately instead of after a cold-start wait.
        began = time.monotonic()
        staged_assets = stage_input_assets(job_input, workflow)
        if staged_assets:
            timer.add_span("input_download", time.monotonic() - began)

        # Near-zero on a warm process; on a cold one this is the ComfyUI boot itself.
        began = time.monotonic()
        start_comfyui()
        timer.add_span("comfy_wait", time.monotonic() - began)
        reporter.phase(PHASE_COMFY_READY)

        client_id = str(uuid.uuid4())
        deadline = time.monotonic() + JOB_TIMEOUT

        mode = f"{len(staged_assets)}-keyframe" if staged_assets else "text-to-video"
        log(f"Job {job_id}: queueing {mode} workflow with {len(workflow)} nodes")
        began = time.monotonic()
        prompt_id = queue_prompt(workflow, client_id)
        timer.add_span("submit", time.monotonic() - began)
        timer.mark("submitted")
        log(f"Job {job_id}: prompt_id={prompt_id}")

        # Everything between here and the first sampler frame is model staging.
        reporter.phase(PHASE_LOADING)

        history = await_execution(prompt_id, client_id, deadline, timer, reporter)

        store, upload_store = build_output_store_for_job(job_input)
        if upload_store is not None:
            reporter.phase(PHASE_UPLOADING, percent=95)

        began = time.monotonic()
        images = collect_outputs(
            history, store=store, protector=protector, generation_id=generation_id
        )
        timer.add_span("output", time.monotonic() - began)

        if upload_store is not None:
            # Reported separately so a slow R2 write is never read as slow inference.
            timer.add_span("output_upload", upload_store.upload_seconds)
            timer.output_bytes = upload_store.uploaded_bytes
        if protector.total_seconds:
            # The whole point of measuring this is to be able to say what Confidential
            # Generation costs. Expect it to disappear next to sampling.
            timer.add_span("encryption", protector.total_seconds)

        if not images:
            status = "no_output"
            reporter.phase(
                PHASE_FAILED, error={"code": "no_output", "message": "workflow saved nothing"}
            )
            return {
                "error": (
                    "The workflow completed but produced no saved output. Ensure it ends in "
                    "SaveVideo (or another Save* node)."
                ),
                "prompt_id": prompt_id,
            }

        log(f"Job {job_id}: returning {len(images)} artefact(s)")
        status = "ok"

        result = {"images": images, "prompt_id": prompt_id, "privacyMode": protector.mode}

        # Only now is the job genuinely done: the upload has already succeeded, because
        # WorkerUploadStore raises rather than returning on a failed PUT.
        if upload_store is not None:
            video = dict(images[0])
            video.pop("data", None)  # metadata only; the bytes are in R2
            result["video"] = video
            reporter.phase(PHASE_COMPLETED, percent=100, video=video)
        else:
            reporter.phase(PHASE_COMPLETED, percent=100)

        return result

    except WorkflowError as error:
        # Covers ImageInputError too: a caller-facing validation failure, not a crash.
        log(f"Job {job_id}: workflow error: {error}")
        status = "workflow_error"
        reporter.phase(PHASE_FAILED, error={"code": "workflow_error", "message": str(error)})
        return {"error": str(error)}
    except Exception as error:
        log(f"Job {job_id}: unhandled error: {type(error).__name__}: {error}")
        status = "exception"
        reporter.phase(
            PHASE_FAILED,
            error={"code": type(error).__name__, "message": str(error)},
        )
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        # Always runs, so a failed or rejected job cannot orphan staged input images.
        for path in staged_assets:
            cleanup_input_image(path)
        # Shadow ranks are stopped and swept on the same boundary as rank 0's plaintext:
        # nothing job-specific may survive into the next request on a warm worker.
        if _shadow_ranks:
            _drain_shadow_ranks(reason=status)
        # The key was request-scoped; it stops being reachable here. See the "Secrets in
        # memory" section of docs/confidential-generation.md for what this is and is not
        # worth - CPython cannot promise every copy is gone.
        if protector is not None:
            protector.destroy()
        timer.progress_callbacks = (reporter.sent, reporter.failed)
        if reporter.seconds:
            timer.add_span("progress_callback_time", reporter.seconds)
        # Emitted for failures too: a slow job that times out is exactly the one whose
        # phase breakdown you want.
        emit_perf(timer, job_index=job_index, status=status)


# --------------------------------------------------------------------------------------
# FlashBoot preload
# --------------------------------------------------------------------------------------
#
# Why this graph, and why it is not loader-only.
#
# Everything below was read out of ComfyUI at the pinned commit (dec5d945) rather than
# assumed, because three of its behaviours decide whether a preload can work at all:
#
#   1. A prompt with no OUTPUT_NODE is rejected outright - execution.validate_prompt()
#      returns "Prompt has no outputs". A graph of four loaders and nothing else would
#      never run. PreviewAny (comfy_extras/nodes_preview_any.py) is OUTPUT_NODE=True,
#      takes IO.ANY and only stringifies its input, so it forces upstream execution
#      without a decode, an encode or a file write.
#
#   2. Executing a loader does NOT put weights on the GPU. UNETLoader/CLIPLoader/VAELoader
#      just build a ModelPatcher; the expensive staging happens later, inside
#      model_management.load_models_gpu(), when a node actually uses the model. The
#      production logs show exactly this - the loaders finish in ~0.6s, then
#      MiniMaxH3ImageToVideo spends ~7s staging the text encoder and SamplerCustomAdvanced
#      ~9s staging the DiT. Those two calls are the 16s of pre_sampling we are targeting,
#      so the preload has to reach them: text encode plus one sampler step.
#
#   3. The saving is real only because ComfyUI reuses the objects. Its output cache is
#      keyed on class_type + inputs with node id deliberately excluded
#      (CacheKeySetInputSignature.include_node_id_in_input() -> False), so the real job's
#      loaders - identical inputs - hit the cache and get back the *same* ModelPatcher
#      instances. load_models_gpu() then finds them already in current_loaded_models
#      (LoadedModel.__eq__ is `self.model is other.model`, i.e. identity) and skips
#      staging entirely. Models survive the preload prompt because unload_all_models() is
#      only called on OOM or under --disable-smart-memory, neither of which applies here.
#
# What is deliberately NOT preloaded: VRAM staging for the two VAEs. There is no way to
# stage a VAE without running a decode, and decode is the expensive thing. Their loaders
# do execute (cheap file open + ModelPatcher), so those cache entries are warm, but the
# decode phase is untouched by this.


def build_preload_workflow() -> dict:
    """The cheapest graph that stages the text encoder and the DiT onto the GPU.

    Loader nodes mirror the real workflow's inputs exactly - that is the whole mechanism.
    Everything downstream is shrunk to the schema minimum (32x32, 5 frames, 1 step) and
    terminates in PreviewAny so nothing is decoded, encoded or written to disk.
    """
    return {
        # --- loaders: inputs identical to examples/fl2va-text-to-video.json ---
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": PRELOAD_UNET, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": PRELOAD_CLIP, "type": "minimax"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": PRELOAD_VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": PRELOAD_AUDIO_VAE}},
        # --- text encode: stages MiniMaxH3TEModel_. No first_frame/last_frame, so the
        #     node never calls vae.encode() and the video VAE stays off the GPU. ---
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": PRELOAD_PROMPT,
                "width": PRELOAD_WIDTH,
                "height": PRELOAD_HEIGHT,
                "length": PRELOAD_LENGTH,
            },
        },
        # --- one sampler step: stages MiniMaxH3 (the DiT) ---
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "7": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 1, "denoise": 1.0},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["7", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["5", 1],
            },
        },
        # --- terminators: OUTPUT_NODEs that only stringify, so the graph is allowed to
        #     run. Node 12 exists purely to make the audio VAE loader execute. ---
        "11": {"class_type": "PreviewAny", "inputs": {"source": ["10", 0]}},
        "12": {"class_type": "PreviewAny", "inputs": {"source": ["4", 0]}},
    }


def _interrupt_comfy() -> None:
    """Ask every rank to abandon the running prompt.

    Every rank, not just rank 0: leaving a shadow mid-graph would have it answer the next
    job's collectives with this job's tensors.

    Used when the preload overruns its timeout: without this the synthetic prompt keeps
    running and the first real job queues behind it, which would be strictly worse than
    not preloading at all.
    """
    targets = [rank.url for rank in _ranks] or [COMFY_URL]
    for url in targets:
        try:
            requests.post(f"{url}/interrupt", timeout=10)
        except Exception as error:
            log(f"WARNING: could not interrupt ComfyUI at {url} after preload timeout: {error}")
    log(f"Preload: sent /interrupt to {len(targets)} rank(s).")


# Which node class stands in for which component, for the preload timing line.
_PRELOAD_COMPONENTS = (
    ("te", "MiniMaxH3ImageToVideo"),
    ("dit", "SamplerCustomAdvanced"),
    ("video_vae", "VAELoader"),
    ("unet_loader", "UNETLoader"),
    ("clip_loader", "CLIPLoader"),
)


def run_flashboot_preload() -> float | None:
    """Stage the H3 models before runpod.serverless.start(), so FlashBoot can snapshot them.

    Returns the elapsed seconds, or None if the preload did not complete. Never raises: a
    preload is an optimisation, so a failure, a timeout or a model that is not where we
    expect it must still leave the worker able to come up and serve requests normally.
    """
    began = time.monotonic()
    workflow = build_preload_workflow()
    timer = JobTimer(workflow)

    try:
        client_id = str(uuid.uuid4())
        deadline = time.monotonic() + PRELOAD_TIMEOUT
        log(
            f"Preload: queueing synthetic {PRELOAD_WIDTH}x{PRELOAD_HEIGHT} "
            f"length={PRELOAD_LENGTH} 1-step graph ({len(workflow)} nodes), "
            f"timeout={PRELOAD_TIMEOUT}s"
        )
        prompt_id = queue_prompt(workflow, client_id)
        timer.mark("submitted")
        await_execution(prompt_id, client_id, deadline, timer)

        elapsed = time.monotonic() - began
        loaded = [
            name for name, node_class in _PRELOAD_COMPONENTS
            if node_class in timer._node_totals
        ]
        parts = " ".join(
            f"preload_{name}={_secs(timer._node_totals[node_class])}"
            for name, node_class in _PRELOAD_COMPONENTS
            if node_class in timer._node_totals
        )
        log(
            f"[perf] preload proc={PROCESS_ID} status=ok total={_secs(elapsed)} "
            f"loaded={','.join(loaded) if loaded else 'none'} {parts}".rstrip()
        )
        return elapsed

    except WorkflowError as error:
        elapsed = time.monotonic() - began
        timed_out = time.monotonic() >= began + PRELOAD_TIMEOUT
        if timed_out:
            _interrupt_comfy()
        log(
            f"[perf] preload proc={PROCESS_ID} "
            f"status={'timeout' if timed_out else 'failed'} "
            f"total={_secs(elapsed)} error={error}"
        )
        return None
    except Exception as error:
        elapsed = time.monotonic() - began
        log(
            f"[perf] preload proc={PROCESS_ID} status=failed total={_secs(elapsed)} "
            f"error={type(error).__name__}: {error}"
        )
        return None


def _shutdown(signum, _frame) -> None:
    log(f"Received signal {signum}; stopping ComfyUI.")
    # Every rank, not just rank 0: a surviving shadow would hold its GPU and its half of
    # the NCCL group open after the worker is supposed to be gone.
    _stop_ranks()
    sys.exit(0)


def main() -> None:
    log("=== MiniMax H3 Blackwell serverless worker starting ===")
    log(f"process_id={PROCESS_ID}")
    log_callback_configuration()
    log(f"COMFY_DIR = {COMFY_DIR}")
    log_torch_environment()

    # The GPU execution plan, before anything else can depend on it. A dual-mode request
    # that cannot be honoured stops the worker here: quietly serving single-GPU results
    # from an endpoint configured for a two-GPU benchmark would poison the measurement
    # this image exists to produce.
    if GPU_MODE_ERROR is not None:
        log(f"FATAL: {GPU_MODE_ERROR}")
        raise SystemExit(1)
    log(GPU_CONFIG.describe())
    if GPU_CONFIG.dual:
        log(
            f"[H3-GPU] launching {GPU_CONFIG.world_size} ComfyUI ranks on ports "
            f"{[gpu_config.comfy_port_for_rank(r, COMFY_PORT) for r in range(GPU_CONFIG.world_size)]}"
        )

    # Fail fast on misconfigured output storage rather than at the end of a paid 5-minute
    # generation. Validates R2 credentials and the key-wrapping key at boot.
    get_output_store()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if os.environ.get("H3_EAGER_START", "1") == "1":
        # Warm ComfyUI during the RunPod cold start instead of inside the first billed job.
        try:
            start_comfyui()
        except Exception as error:
            if GPU_CONFIG.dual:
                # In single mode a failed eager start is recoverable - the first job just
                # pays for the boot. In dual mode it means the group never formed, and the
                # only thing a retry could produce is an unlabelled single-GPU benchmark.
                log(f"FATAL: dual-GPU start failed: {error}")
                _stop_ranks()
                raise SystemExit(1) from error
            log(f"WARNING: eager ComfyUI start failed ({error}); will retry on first job.")

    # The preload runs after ComfyUI is ready and before the serverless loop starts, so
    # the models are resident in the process FlashBoot snapshots. Opt-in: a miss costs a
    # synthetic step on every cold start, which is only worth paying once the [perf] data
    # shows workers actually being reused.
    preload_total = None
    if PRELOAD_ENABLED:
        log("FlashBoot preload enabled")
        preload_total = run_flashboot_preload()
    else:
        log("FlashBoot preload disabled")

    # The preload's staging is not the first job's memory, so its peaks must not be
    # attributed to it.
    reset_gpu_peaks()

    startup_fields = [
        f"proc={PROCESS_ID}",
        f"gpu_mode={GPU_CONFIG.mode}",
        f"gpu_count={GPU_CONFIG.world_size}",
        f"to_serverless_ready={_secs(time.monotonic() - _PROCESS_START)}",
        f"comfy_boot={_secs(_comfy_boot_seconds)}",
        f"preload_enabled={'true' if PRELOAD_ENABLED else 'false'}",
    ]
    if PRELOAD_ENABLED:
        startup_fields.append(f"preload_total={_secs(preload_total)}")
    log("[perf] startup " + " ".join(startup_fields))

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
