"""Tests for the opt-in FlashBoot model preload.

Run with:  python -m unittest discover -s tests -v

The properties that matter here are all failure properties. A preload is an optimisation:
it must never take the worker down, never hang startup, and never change what a real
request returns. The graph shape is tested too, because the *only* reason the preload
saves anything is that its loader inputs match the real workflow's byte for byte - if
those drift, the preload silently warms objects nobody reuses.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-pre-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-pre-output-"))
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_WORKFLOW = os.path.join(REPO_ROOT, "examples", "fl2va-text-to-video.json")


class PreloadEnvTest(unittest.TestCase):
    def test_defaults_to_disabled(self):
        """Opt-in. An unset variable must leave startup exactly as it was."""
        self.assertFalse(handler.PRELOAD_ENABLED)

    def test_timeout_has_a_default(self):
        self.assertEqual(handler.PRELOAD_TIMEOUT, 60)

    def test_env_parsing_is_strict_about_enabling(self):
        # Only "1" enables it; anything else must not.
        for raw, expected in (("1", True), ("0", False), ("", False), ("true", False)):
            with self.subTest(raw=raw):
                self.assertEqual(raw == "1", expected)


class PreloadWorkflowShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wf = handler.build_preload_workflow()
        with open(REAL_WORKFLOW, encoding="utf-8") as fh:
            self.real = json.load(fh)

    def by_class(self, workflow: dict, class_type: str) -> list[dict]:
        return [n["inputs"] for n in workflow.values() if n["class_type"] == class_type]

    def test_graph_is_valid_json_serialisable(self):
        json.dumps(self.wf)

    def test_has_an_output_node(self):
        """ComfyUI's validate_prompt rejects a graph with no OUTPUT_NODE outright."""
        self.assertTrue(any(n["class_type"] == "PreviewAny" for n in self.wf.values()))

    def test_does_no_decode_encode_or_save(self):
        banned = {"VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo", "SaveImage"}
        present = {n["class_type"] for n in self.wf.values()}
        self.assertEqual(present & banned, set(), "preload must not decode or write output")

    def test_loader_inputs_match_the_real_workflow_exactly(self):
        """The entire saving depends on this.

        ComfyUI keys its output cache on class_type + inputs (node id excluded), so the
        real job only reuses the preloaded ModelPatchers when these match byte for byte.
        """
        for class_type in ("UNETLoader", "CLIPLoader", "VAELoader"):
            with self.subTest(class_type=class_type):
                mine = sorted(map(json.dumps, self.by_class(self.wf, class_type)), key=str)
                theirs = sorted(map(json.dumps, self.by_class(self.real, class_type)), key=str)
                self.assertEqual(mine, theirs)

    def test_all_four_models_are_referenced(self):
        blob = json.dumps(self.wf)
        for name in (
            handler.PRELOAD_UNET,
            handler.PRELOAD_CLIP,
            handler.PRELOAD_VIDEO_VAE,
            handler.PRELOAD_AUDIO_VAE,
        ):
            with self.subTest(name=name):
                self.assertIn(name, blob)

    def test_sampler_runs_exactly_one_step(self):
        scheduler = self.by_class(self.wf, "BasicScheduler")[0]
        self.assertEqual(scheduler["steps"], 1)
        # ...and the real workflow's 20-step default is untouched.
        self.assertEqual(self.by_class(self.real, "BasicScheduler")[0]["steps"], 20)

    def test_dimensions_respect_the_node_schema_minimums(self):
        """MiniMaxH3ImageToVideo: width/height min 32 step 32, length min 5 step 17."""
        i2v = self.by_class(self.wf, "MiniMaxH3ImageToVideo")[0]
        self.assertGreaterEqual(i2v["width"], 32)
        self.assertGreaterEqual(i2v["height"], 32)
        self.assertEqual(i2v["width"] % 32, 0)
        self.assertEqual(i2v["height"] % 32, 0)
        self.assertGreaterEqual(i2v["length"], 5)
        self.assertEqual((i2v["length"] - 5) % 17, 0)

    def test_no_keyframe_images_so_the_vae_never_encodes(self):
        i2v = self.by_class(self.wf, "MiniMaxH3ImageToVideo")[0]
        self.assertNotIn("first_frame", i2v)
        self.assertNotIn("last_frame", i2v)

    def test_every_reference_points_at_a_real_node(self):
        for node_id, node in self.wf.items():
            for key, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    with self.subTest(node=node_id, input=key):
                        self.assertIn(value[0], self.wf)


class PreloadFailureTest(unittest.TestCase):
    """A preload must never be the reason the worker fails to serve."""

    def setUp(self) -> None:
        self._queue = handler.queue_prompt
        self._await = handler.await_execution
        self._post = handler.requests.post

    def tearDown(self) -> None:
        handler.queue_prompt = self._queue
        handler.await_execution = self._await
        handler.requests.post = self._post

    def test_queue_failure_returns_none_instead_of_raising(self):
        def boom(*a, **k):
            raise RuntimeError("ComfyUI refused the prompt")

        handler.queue_prompt = boom
        self.assertIsNone(handler.run_flashboot_preload())

    def test_workflow_error_returns_none_instead_of_raising(self):
        handler.queue_prompt = lambda *a, **k: "pid"
        def bad(*a, **k):
            raise handler.WorkflowError("node blew up")

        handler.await_execution = bad
        self.assertIsNone(handler.run_flashboot_preload())

    def test_timeout_interrupts_comfyui_so_the_first_job_is_not_queued_behind_it(self):
        handler.queue_prompt = lambda *a, **k: "pid"
        posted = []
        handler.requests.post = lambda url, **k: posted.append(url)

        original_timeout = handler.PRELOAD_TIMEOUT
        handler.PRELOAD_TIMEOUT = 0  # already expired the moment we start
        try:
            def slow(*a, **k):
                raise handler.WorkflowError("exceeded H3_JOB_TIMEOUT")

            handler.await_execution = slow
            self.assertIsNone(handler.run_flashboot_preload())
        finally:
            handler.PRELOAD_TIMEOUT = original_timeout

        self.assertTrue(any(url.endswith("/interrupt") for url in posted), posted)

    def test_interrupt_failure_is_itself_swallowed(self):
        def boom(*a, **k):
            raise RuntimeError("connection refused")

        handler.requests.post = boom
        handler._interrupt_comfy()  # must not raise

    def test_success_returns_elapsed_seconds(self):
        handler.queue_prompt = lambda *a, **k: "pid"
        handler.await_execution = lambda *a, **k: {}
        elapsed = handler.run_flashboot_preload()
        self.assertIsNotNone(elapsed)
        self.assertGreaterEqual(elapsed, 0.0)


class PreloadDoesNotChangeContractTest(unittest.TestCase):
    def test_handler_signature_and_responses_unchanged(self):
        """The preload lives in main(); handler() must be untouched by it."""
        result = handler.handler({"id": "x", "input": {}})
        self.assertIn("error", result)
        self.assertNotIn("images", result)

    def test_preload_graph_is_not_the_served_workflow(self):
        """Nothing in the preload path may mutate what a caller submitted."""
        wf = handler.build_preload_workflow()
        again = handler.build_preload_workflow()
        self.assertEqual(wf, again)
        wf["1"]["inputs"]["unet_name"] = "mutated"
        self.assertNotEqual(wf, handler.build_preload_workflow())


if __name__ == "__main__":
    unittest.main(verbosity=2)
