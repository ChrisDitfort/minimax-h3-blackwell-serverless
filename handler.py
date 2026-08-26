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

# ComfyUI history buckets that carry saved artefacts. SaveVideo emits ui.PreviewVideo,
# whose as_dict() is {"images": [...], "animated": (True,)} - so MP4s land under "images",
# which is exactly what the Cloudflare Worker already reads.
OUTPUT_KEYS = ("images", "gifs", "videos", "audio")

_comfy_process: subprocess.Popen | None = None
_start_lock = threading.Lock()


def log(message: str) -> None:
    print(f"[handler] {message}", flush=True)


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


def _comfy_is_ready() -> bool:
    try:
        response = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
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


def start_comfyui() -> None:
    """Start ComfyUI once and block until it genuinely answers HTTP.

    Serialised behind a lock because RunPod may hand us concurrent jobs, and re-entrant
    starts would race two ComfyUI processes onto the same port.
    """
    global _comfy_process, _comfy_boot_seconds

    with _start_lock:
        if _comfy_process is not None and _comfy_process.poll() is None and _comfy_is_ready():
            return

        if _comfy_process is not None and _comfy_process.poll() is not None:
            log(f"ComfyUI exited with code {_comfy_process.returncode}; restarting it.")
            _comfy_process = None

        # Covers the wait even when we join a start already in flight, so comfy_boot
        # reflects time-to-ready rather than time-since-fork.
        boot_began = time.monotonic()

        if _comfy_process is None:
            for directory in (COMFY_OUTPUT_DIR, COMFY_TEMP_DIR, COMFY_INPUT_DIR):
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as error:
                    log(f"WARNING: could not create {directory}: {error}")

            command = [
                sys.executable,
                "main.py",
                "--listen",
                COMFY_HOST,  # localhost only: RunPod Serverless exposes no ports
                "--port",
                str(COMFY_PORT),
                "--preview-method",
                "none",  # previews waste VRAM and bandwidth in a headless worker
                "--disable-auto-launch",
                "--disable-metadata",
                "--output-directory",
                COMFY_OUTPUT_DIR,
                "--temp-directory",
                COMFY_TEMP_DIR,
                "--input-directory",
                COMFY_INPUT_DIR,
            ]
            extra_args = parse_extra_args(os.environ.get("COMFY_EXTRA_ARGS", ""))
            command.extend(extra_args)

            # Logged in full and on its own line so an A/B run can be attributed to the
            # exact flags it used - `--highvram` in particular is invisible otherwise.
            log(f"Starting ComfyUI: {' '.join(command)} (cwd={COMFY_DIR})")
            log(
                "ComfyUI effective args: "
                f"COMFY_EXTRA_ARGS={os.environ.get('COMFY_EXTRA_ARGS', '<unset>')!r} "
                f"-> extra={extra_args or '[]'}"
            )
            # Resolved here, not at import: the GPU is only knowable once the worker is
            # actually placed on a host.
            _comfy_process = subprocess.Popen(command, cwd=COMFY_DIR, env=build_comfy_env())

        deadline = time.monotonic() + COMFY_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if _comfy_process.poll() is not None:
                code = _comfy_process.returncode
                _comfy_process = None
                raise RuntimeError(
                    f"ComfyUI exited during startup with code {code}. "
                    "Check the ComfyUI log above for the underlying error."
                )
            if _comfy_is_ready():
                elapsed = time.monotonic() - boot_began
                _comfy_boot_seconds = elapsed
                log(f"ComfyUI is ready after ~{elapsed:.1f}s")
                return
            time.sleep(1)

        raise RuntimeError(f"ComfyUI did not become ready within {COMFY_STARTUP_TIMEOUT}s")


# --------------------------------------------------------------------------------------
# Workflow execution
# --------------------------------------------------------------------------------------


class WorkflowError(RuntimeError):
    """A workflow was rejected or failed inside ComfyUI (a user/job error, not a bug)."""


def queue_prompt(workflow: dict, client_id: str) -> str:
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


def await_execution(
    prompt_id: str,
    client_id: str,
    deadline: float,
    timer: JobTimer | None = None,
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
                            timer.on_node(str(data.get("node")))

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


def _download_image(url: str) -> bytes:
    """Fetch an image over HTTPS, validating every redirect hop and capping the size."""
    current = url
    for hop in range(MAX_IMAGE_REDIRECTS + 1):
        _assert_fetchable_url(current)

        # Redirects are followed manually so each destination is re-validated; letting
        # requests follow them would let a redirect land on an internal address.
        response = requests.get(
            current,
            timeout=IMAGE_DOWNLOAD_TIMEOUT,
            stream=True,
            allow_redirects=False,
            headers={"Accept": "image/*"},
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


def find_image_node(workflow: dict, explicit_id: str | None) -> str:
    """Decide which node receives the staged image, or fail rather than guess."""
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

    # Prefer a loader that actually feeds MiniMaxH3ImageToVideo.first_frame.
    preferred: set[str] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "MiniMaxH3ImageToVideo":
            continue
        link = (node.get("inputs") or {}).get("first_frame")
        if isinstance(link, list) and len(link) == 2:
            preferred |= _reachable_image_loaders(workflow, str(link[0]))

    if len(preferred) == 1:
        node_id = next(iter(preferred))
        log(f"Image node auto-detected via MiniMaxH3ImageToVideo.first_frame: #{node_id}")
        return node_id

    if len(candidates) == 1:
        log(f"Image node auto-detected as the workflow's only image loader: #{candidates[0]}")
        return candidates[0]

    raise ImageInputError(
        f"Could not unambiguously choose an image node: found {len(candidates)} image "
        f"loaders ({', '.join(sorted(candidates))})"
        + (f" and {len(preferred)} feeding first_frame" if preferred else "")
        + ". Pass image_node_id to say which one should receive the image."
    )


def stage_input_image(job_input: dict, workflow: dict) -> str | None:
    """Validate, store and wire up an optional input image.

    Returns the absolute path of the staged file (for cleanup), or None for the
    text-to-video path where no image was supplied.
    """
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    if image_url and image_base64:
        raise ImageInputError(
            "Provide either image_url or image_base64, not both."
        )
    if not image_url and not image_base64:
        return None  # text-to-video: unchanged behaviour

    if image_url:
        if not isinstance(image_url, str):
            raise ImageInputError("image_url must be a string.")
        log("Fetching input image from image_url")
        data = _download_image(image_url)
    else:
        log("Decoding input image from image_base64")
        data = _decode_base64_image(image_base64)

    extension = _validate_image_bytes(data)

    # Resolve the target node before writing anything, so a workflow we cannot wire up
    # never leaves a file behind.
    node_id = find_image_node(workflow, job_input.get("image_node_id"))

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

    workflow[node_id]["inputs"][IMAGE_LOADER_FIELD] = filename
    log(f"Staged input image as {filename} and wired it into node #{node_id}")
    return path


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

    ``store`` receives a path to a plaintext file on the worker and returns the dict that
    goes into ``output.images[]``. Implementations own their own cleanup of that file.
    """

    def store(self, path: str, entry: dict) -> dict:
        raise NotImplementedError


class Base64Store(OutputStore):
    """Inline the artefact as base64 - the format the current Cloudflare Worker expects."""

    def __init__(self) -> None:
        self.max_bytes = int(os.environ.get("H3_MAX_BASE64_BYTES", str(180 * 1024 * 1024)))

    def store(self, path: str, entry: dict) -> dict:
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

    def store(self, path: str, entry: dict) -> dict:
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


def collect_outputs(history: dict) -> list[dict]:
    store = get_output_store()
    results: list[dict] = []
    scratch = tempfile.mkdtemp(prefix="h3-out-")
    keep_outputs = os.environ.get("H3_KEEP_OUTPUTS", "0") == "1"

    try:
        for entry in iter_output_entries(history):
            path = resolve_output_path(entry)
            if path is None:
                path = download_output(entry, os.path.join(scratch, entry["filename"]))
            results.append(store.store(path, entry))

            # A warm worker serves many jobs; without this, generated videos pile up in the
            # output directory until the container disk fills. The r2 store already deletes
            # its plaintext, so this is a no-op there.
            if not keep_outputs:
                _discard_output_file(path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return results


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


def handler(job: dict) -> dict:
    job_id = job.get("id", "<unknown>")
    staged_image: str | None = None

    global _jobs_served
    with _jobs_lock:
        _jobs_served += 1
        job_index = _jobs_served

    job_input = job.get("input") or {}
    workflow = job_input.get("workflow")
    timer = JobTimer(workflow if isinstance(workflow, dict) else None)
    status = "error"

    try:
        if not isinstance(workflow, dict) or not workflow:
            status = "bad_request"
            return {"error": "input.workflow is required and must be a ComfyUI API-format object."}

        # Optional first-frame image-to-video. With no image supplied this is a no-op and
        # the text-to-video path runs exactly as before. Done before starting ComfyUI so a
        # bad image is rejected immediately instead of after a cold-start wait.
        began = time.monotonic()
        staged_image = stage_input_image(job_input, workflow)
        if staged_image:
            timer.add_span("stage_image", time.monotonic() - began)

        # Near-zero on a warm process; on a cold one this is the ComfyUI boot itself.
        began = time.monotonic()
        start_comfyui()
        timer.add_span("comfy_wait", time.monotonic() - began)

        client_id = str(uuid.uuid4())
        deadline = time.monotonic() + JOB_TIMEOUT

        mode = "image-to-video" if staged_image else "text-to-video"
        log(f"Job {job_id}: queueing {mode} workflow with {len(workflow)} nodes")
        began = time.monotonic()
        prompt_id = queue_prompt(workflow, client_id)
        timer.add_span("submit", time.monotonic() - began)
        timer.mark("submitted")
        log(f"Job {job_id}: prompt_id={prompt_id}")

        history = await_execution(prompt_id, client_id, deadline, timer)

        began = time.monotonic()
        images = collect_outputs(history)
        timer.add_span("output", time.monotonic() - began)

        if not images:
            status = "no_output"
            return {
                "error": (
                    "The workflow completed but produced no saved output. Ensure it ends in "
                    "SaveVideo (or another Save* node)."
                ),
                "prompt_id": prompt_id,
            }

        log(f"Job {job_id}: returning {len(images)} artefact(s)")
        status = "ok"
        return {"images": images, "prompt_id": prompt_id}

    except WorkflowError as error:
        # Covers ImageInputError too: a caller-facing validation failure, not a crash.
        log(f"Job {job_id}: workflow error: {error}")
        status = "workflow_error"
        return {"error": str(error)}
    except Exception as error:
        log(f"Job {job_id}: unhandled error: {type(error).__name__}: {error}")
        status = "exception"
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        # Always runs, so a failed or rejected job cannot orphan an input image.
        cleanup_input_image(staged_image)
        # Emitted for failures too: a slow job that times out is exactly the one whose
        # phase breakdown you want.
        emit_perf(timer, job_index=job_index, status=status)


def _shutdown(signum, _frame) -> None:
    log(f"Received signal {signum}; stopping ComfyUI.")
    if _comfy_process is not None and _comfy_process.poll() is None:
        _comfy_process.terminate()
        try:
            _comfy_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _comfy_process.kill()
    sys.exit(0)


def main() -> None:
    log("=== MiniMax H3 Blackwell serverless worker starting ===")
    log(f"process_id={PROCESS_ID}")
    log(f"COMFY_DIR = {COMFY_DIR}")
    log_torch_environment()

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
            log(f"WARNING: eager ComfyUI start failed ({error}); will retry on first job.")

    log(
        f"[perf] startup proc={PROCESS_ID} "
        f"to_serverless_ready={_secs(time.monotonic() - _PROCESS_START)} "
        f"comfy_boot={_secs(_comfy_boot_seconds)}"
    )

    # ----------------------------------------------------------------------------------
    # NOT IMPLEMENTED (deliberately): FlashBoot model preload
    # ----------------------------------------------------------------------------------
    # RunPod's FlashBoot snapshots the worker process when it scales to zero and restores
    # it on the next request. It can only capture state that already exists at snapshot
    # time. Today ComfyUI is running by this point (H3_EAGER_START), but it holds *no*
    # models: ComfyUI loads them lazily, on the first prompt. So the snapshot restores a
    # ComfyUI with empty VRAM and every cold start re-pays model staging - measured at
    # 13-25s, i.e. the pre_sampling field on the [perf] line.
    #
    # What preloading here would require:
    #
    #   * Components to load. Four, all lazy today: MiniMaxH3TEModel_ (text encoder,
    #     ~14956 MB staged), MiniMaxH3 (the FL2VA DiT, ~19995 MB), MiniMaxH3VideoVAE
    #     (~4965 MB) and MiniMaxH3AudioVAE (~576 MB). ~41 GB against 96 GB of VRAM, so it
    #     fits with room to spare.
    #   * A synthetic workflow: yes, almost certainly. ComfyUI has no public "load model"
    #     API - loaders are graph nodes, so the only supported way to force a load is to
    #     execute a graph that touches them. That means a real (tiny) generation: minimum
    #     resolution, 1 step, output discarded. It must use the *same* loader nodes and
    #     checkpoint names as the real workflow or it warms the wrong objects.
    #   * Residency: ComfyUI keeps loaded models cached and only evicts under VRAM
    #     pressure. With ~41 GB resident on a 96 GB card there is no pressure, so they
    #     should survive to the first real job. Not guaranteed - comfy-aimdo's DynamicVRAM
    #     layer stages through pinned host RAM and makes its own residency decisions.
    #   * Risks. (1) On a FlashBoot *miss* the warmup is roughly cost-neutral - the same
    #     staging happens, just earlier - but it is NOT free: the synthetic step itself
    #     costs a few seconds of GPU time on every cold start that does not get reused.
    #     (2) It hard-codes assumptions about the workflow's loader nodes, so a workflow
    #     change silently stops warming the right models. (3) A warmup failure must never
    #     block serving; it would need the same try/except treatment as the eager start.
    #     (4) It lengthens the window before the worker registers with RunPod, which could
    #     look like a slower cold start on any request that arrives mid-warmup.
    #   * Where the hook belongs: right here - after start_comfyui() has returned ready and
    #     immediately before runpod.serverless.start(), so the models are resident in the
    #     process that FlashBoot snapshots.
    #
    # Gate on evidence before building it: compare pre_sampling across jobs where proc=
    # differs (true cold) against jobs where it repeats (reused). If reuse is rare at this
    # traffic level, preloading buys nothing and costs a synthetic step every time.
    # ----------------------------------------------------------------------------------

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
