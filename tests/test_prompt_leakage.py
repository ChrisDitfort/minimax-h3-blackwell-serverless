"""The user's prompt must not reach the logs, the job result, or the progress callback.

Run with:  python -m unittest discover -s tests -v

tests/test_redaction.py already covers *secrets* - keys, tokens, passphrases. This file
covers the other half of the privacy property, which had no coverage at all: the prompt
itself is user content, and the worker handles it in three places where it can escape.

The risk is not a careless `log(prompt)` - there is no such call. It is that ComfyUI's
error responses quote the offending input back at you. A validation failure carries
`extra_info.received_value`, and an execution failure carries `current_inputs`; both are
the node's actual argument values, which for MiniMaxH3ImageToVideo means the prompt. The
handler used to interpolate those structures wholesale into a WorkflowError, which is then
logged, returned as the RunPod job result, and posted to the progress callback.

Canaries are synthetic so a failure is unambiguous about what leaked and from where.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-leak-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-leak-output-"))
os.environ.setdefault("H3_OUTPUT_MODE", "base64")

if "runpod" not in sys.modules:
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = runpod_stub

if "websocket" not in sys.modules:
    websocket_stub = types.ModuleType("websocket")
    websocket_stub.WebSocketTimeoutException = type("WebSocketTimeoutException", (Exception,), {})
    websocket_stub.create_connection = lambda *a, **k: None
    sys.modules["websocket"] = websocket_stub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402

try:  # h3_parallel.runtime imports torch; the container always has it, a laptop may not.
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PROMPT_CANARY = "PROMPT_CANARY_PRIVORA_7F91A2"
PROMPT = f"{PROMPT_CANARY} a small red toy train on a wooden table"


def workflow_with_prompt(prompt: str = PROMPT) -> dict:
    """The shape the Cloudflare Worker submits: the prompt is a node input value."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"prompt": prompt, "width": 1024, "height": 576, "length": 124},
        },
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"model": ["1", 0]}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0]}},
    }


class CapturedLog:
    """Collect everything the handler writes through its own log()."""

    def __init__(self):
        self.lines: list[str] = []

    def __enter__(self):
        self._original = handler.log
        handler.log = lambda message: self.lines.append(str(message))
        return self

    def __exit__(self, *exc):
        handler.log = self._original
        return False

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def fake_post(status_code: int, payload):
    def post(url, json=None, timeout=None, **kwargs):
        return types.SimpleNamespace(
            status_code=status_code,
            json=lambda: payload,
            text=json_dumps(payload),
        )

    return post


def json_dumps(value) -> str:
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


class ValidationErrorLeakTests(unittest.TestCase):
    """ComfyUI answers a bad prompt graph with the offending value quoted back."""

    def setUp(self):
        self._post = handler.requests.post
        self._shadows = handler._shadow_ranks
        handler._shadow_ranks = []

    def tearDown(self):
        handler.requests.post = self._post
        handler._shadow_ranks = self._shadows

    #: A real ComfyUI 400 body. `received_value` and `details` are the node's own
    #: argument, which is the user's prompt.
    COMFY_400 = {
        "error": {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": "",
            "extra_info": {},
        },
        "node_errors": {
            "5": {
                "class_type": "MiniMaxH3ImageToVideo",
                "errors": [
                    {
                        "type": "invalid_input_type",
                        "message": "Failed to convert an input value to a STRING value",
                        "details": f"prompt, {PROMPT}, some conversion error",
                        "extra_info": {
                            "input_name": "prompt",
                            "input_config": ["STRING", {"multiline": True}],
                            "received_value": PROMPT,
                        },
                    }
                ],
            }
        },
    }

    def test_a_rejected_workflow_does_not_put_the_prompt_in_the_error(self):
        handler.requests.post = fake_post(400, self.COMFY_400)
        with self.assertRaises(handler.WorkflowError) as caught:
            handler.queue_prompt(workflow_with_prompt(), "client-id")
        self.assertNotIn(
            PROMPT_CANARY, str(caught.exception),
            "ComfyUI's received_value/details echoed the prompt into the error, which is "
            "logged, returned as the RunPod job result and posted to the callback",
        )

    def test_the_error_still_says_which_node_and_why(self):
        """Redaction must not cost the operator the diagnosis."""
        handler.requests.post = fake_post(400, self.COMFY_400)
        with self.assertRaises(handler.WorkflowError) as caught:
            handler.queue_prompt(workflow_with_prompt(), "client-id")
        message = str(caught.exception)
        self.assertIn("5", message, "the failing node id must survive")
        self.assertIn("MiniMaxH3ImageToVideo", message, "the node class must survive")
        self.assertIn("invalid_input_type", message, "the error type must survive")

    def test_nothing_is_logged_containing_the_prompt(self):
        handler.requests.post = fake_post(400, self.COMFY_400)
        with CapturedLog() as captured:
            with self.assertRaises(handler.WorkflowError):
                handler.queue_prompt(workflow_with_prompt(), "client-id")
        self.assertNotIn(PROMPT_CANARY, captured.text)

    def test_a_non_json_error_body_is_still_bounded(self):
        """A proxy returning an HTML page must not dump the whole page into the log."""
        handler.requests.post = fake_post(502, None)
        handler.requests.post = lambda url, json=None, timeout=None, **k: types.SimpleNamespace(
            status_code=502,
            json=lambda: (_ for _ in ()).throw(ValueError("not json")),
            text="<html>" + ("x" * 50000) + PROMPT + "</html>",
        )
        with self.assertRaises(handler.WorkflowError) as caught:
            handler.queue_prompt(workflow_with_prompt(), "client-id")
        message = str(caught.exception)
        self.assertNotIn(PROMPT_CANARY, message)
        self.assertLess(len(message), 2000, "an unbounded body must not reach the log")


class ExecutionErrorLeakTests(unittest.TestCase):
    """A node that raises mid-execution carries its inputs in the history record."""

    #: ComfyUI's execution_error message. `current_inputs` is every argument the node
    #: received - including the prompt - and `traceback` is arbitrary source text.
    HISTORY = {
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                [
                    "execution_error",
                    {
                        "node_id": "5",
                        "node_type": "MiniMaxH3ImageToVideo",
                        "exception_type": "RuntimeError",
                        "exception_message": "CUDA out of memory",
                        "traceback": ["  File \"nodes.py\", line 1, in encode\n"],
                        "current_inputs": {"prompt": [PROMPT], "width": [1024]},
                    },
                ]
            ],
        }
    }

    def test_an_execution_failure_does_not_leak_the_prompt(self):
        with self.assertRaises(handler.WorkflowError) as caught:
            handler._raise_if_history_failed(self.HISTORY, "prompt-id")
        self.assertNotIn(PROMPT_CANARY, str(caught.exception))

    def test_the_failure_still_identifies_the_node(self):
        with self.assertRaises(handler.WorkflowError) as caught:
            handler._raise_if_history_failed(self.HISTORY, "prompt-id")
        message = str(caught.exception)
        self.assertIn("MiniMaxH3ImageToVideo", message)
        self.assertIn("RuntimeError", message)

    def test_a_failure_with_no_structured_error_does_not_dump_the_status(self):
        """The fallback path interpolated the whole server-controlled status dict."""
        history = {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [["execution_cached", {"nodes": ["1"], "note": PROMPT}]],
            }
        }
        with self.assertRaises(handler.WorkflowError) as caught:
            handler._raise_if_history_failed(history, "prompt-id")
        self.assertNotIn(PROMPT_CANARY, str(caught.exception))


class JobLoggingTests(unittest.TestCase):
    """The per-job log lines describe the job's shape, never its content."""

    def test_the_queue_line_counts_nodes_rather_than_naming_them(self):
        workflow = workflow_with_prompt()
        with CapturedLog() as captured:
            handler.log(f"Job job-1: queueing text-to-video workflow with {len(workflow)} nodes")
        self.assertNotIn(PROMPT_CANARY, captured.text)
        self.assertIn("4 nodes", captured.text)

    def test_the_perf_line_never_contains_the_prompt(self):
        timer = handler.JobTimer(workflow_with_prompt())
        timer.privacy_mode = "confidential"
        line = timer.summary(job_index=1, status="ok")
        self.assertNotIn(PROMPT_CANARY, line)

    def test_the_node_breakdown_names_classes_not_values(self):
        timer = handler.JobTimer(workflow_with_prompt())
        timer.on_node("5")
        timer.on_execution_end()
        summary = timer.node_summary() or ""
        self.assertNotIn(PROMPT_CANARY, summary)
        self.assertIn("MiniMaxH3ImageToVideo", summary)


class MediaMetadataTests(unittest.TestCase):
    """--disable-metadata is load-bearing, not cosmetic."""

    def test_comfyui_is_launched_with_metadata_disabled(self):
        # Without this flag ComfyUI's SaveVideo embeds `metadata["prompt"] = the whole
        # workflow` - including the user's prompt - into every MP4 it writes, and in
        # standard mode that file is uploaded to R2 as-is.
        command = handler.comfy_command(handler.ComfyRank(0))
        self.assertIn("--disable-metadata", command)

    def test_previews_are_off(self):
        # Preview frames are plaintext media written outside the artefact lifecycle.
        command = handler.comfy_command(handler.ComfyRank(0))
        self.assertIn("--preview-method", command)
        self.assertEqual(command[command.index("--preview-method") + 1], "none")


class EnvironmentDumpTests(unittest.TestCase):
    """No code path prints the environment or a secret value."""

    def test_no_source_file_dumps_the_environment(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []
        targets = ["handler.py", "artifacts.py"] + [
            os.path.join("h3_parallel", name)
            for name in os.listdir(os.path.join(root, "h3_parallel"))
            if name.endswith(".py")
        ]
        for relative in targets:
            source = open(os.path.join(root, relative), encoding="utf-8").read()
            for pattern in ("print(os.environ", "log(os.environ", "dict(os.environ)"):
                if pattern in source:
                    offenders.append(f"{relative}: {pattern}")
        self.assertEqual(offenders, [], "the environment must never be dumped wholesale")

    @unittest.skipUnless(HAS_TORCH, "h3_parallel.runtime imports torch")
    def test_the_only_environment_logging_is_by_variable_name(self):
        """h3_parallel logs NCCL_* names and values; none of them are secrets."""
        from h3_parallel import runtime

        lines = runtime.transport_diagnostics()
        joined = " ".join(lines)
        for secret in ("SECRET", "PASSWORD", "PRIVATE", "TOKEN"):
            self.assertNotIn(secret, joined.upper())


if __name__ == "__main__":
    unittest.main()
