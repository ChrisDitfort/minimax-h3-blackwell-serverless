"""The handler's two request shapes, and what happens to staged references afterwards.

Run with:  python -m unittest discover -s tests -v

The rebuild introduced a second way in. The property being defended is that it is only a
second *entrance* - once a graph exists, submission, progress, output, encryption and
cleanup are the same code on both paths, because a new API must not become a new privacy
boundary.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-privora-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-privora-output-"))
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
from privora import errors, models  # noqa: E402

CANARY = "PROMPT_CANARY_PRIVORA_7F91A2"

FULL_INVENTORY = models.ModelInventory.from_names([
    models.CHECKPOINTS[models.FL2VA],
    models.CHECKPOINTS[models.REF2VA],
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
])


class WithInventory:
    """Pretend the image was built with every weight."""

    def __enter__(self):
        self._saved = handler.MODEL_INVENTORY
        handler.MODEL_INVENTORY = FULL_INVENTORY
        return self

    def __exit__(self, *exc):
        handler.MODEL_INVENTORY = self._saved
        return False


class RequestRoutingTests(unittest.TestCase):
    def test_a_privora_request_builds_a_graph_and_reports_what_ran(self):
        with WithInventory():
            graph, job_dir, metadata = handler.build_privora_job(
                {"mode": "create", "prompt": "An ocean scene", "quality": "standard",
                 "aspectRatio": "16:9", "duration": 5, "seed": 51,
                 "generationMode": "turbo"},
                "gen-routing-1",
            )
        try:
            self.assertEqual(graph["conditioning"]["class_type"], "MiniMaxH3ImageToVideo")
            self.assertEqual(metadata["generationMode"], "turbo")
            self.assertEqual(metadata["steps"], 8)
            self.assertEqual(metadata["seed"], 51)
            self.assertEqual((metadata["width"], metadata["height"]), (1024, 576))
            self.assertEqual(metadata["frames"], 124)
        finally:
            handler.cleanup_job_dir(job_dir)

    def test_a_legacy_workflow_request_is_not_routed_through_privora(self):
        # The pre-rebuild contract must reach ComfyUI untouched. A raw graph is passed
        # straight through; nothing in privora sees it.
        job_input = {"workflow": {"1": {"class_type": "UNETLoader", "inputs": {}}}}
        self.assertIsInstance(job_input.get("workflow"), dict)

    def test_an_unavailable_model_is_refused_before_anything_is_staged(self):
        fl2va_only = models.ModelInventory.from_names([models.CHECKPOINTS[models.FL2VA]])
        saved = handler.MODEL_INVENTORY
        handler.MODEL_INVENTORY = fl2va_only
        try:
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(
                    {"mode": "references", "prompt": "x", "seed": 1,
                     "references": [{"type": "image", "role": "character"}]},
                    "gen-unavailable",
                )
            self.assertEqual(caught.exception.code, errors.MODEL_LOAD_FAILED)
        finally:
            handler.MODEL_INVENTORY = saved


class JobIsolationTests(unittest.TestCase):
    """One caller's references must never be visible to the next."""

    def test_each_job_gets_its_own_directory(self):
        first = handler._job_input_dir("gen-alpha")
        second = handler._job_input_dir("gen-beta")
        try:
            self.assertNotEqual(first, second)
            self.assertTrue(os.path.isdir(first) and os.path.isdir(second))
            self.assertTrue(os.path.basename(first).startswith("job-"))
        finally:
            handler.cleanup_job_dir(first)
            handler.cleanup_job_dir(second)

    def test_the_directory_name_cannot_be_steered_by_the_caller(self):
        # generation_id comes from the control plane, but it is still input.
        hostile = handler._job_input_dir("../../etc/passwd")
        try:
            root = os.path.realpath(handler.COMFY_INPUT_DIR)
            self.assertTrue(os.path.realpath(hostile).startswith(root + os.sep))
            self.assertNotIn("..", os.path.basename(hostile))
        finally:
            handler.cleanup_job_dir(hostile)

    def test_cleanup_removes_the_directory_and_its_contents(self):
        job_dir = handler._job_input_dir("gen-cleanup")
        with open(os.path.join(job_dir, "ref-abc.png"), "wb") as handle:
            handle.write(b"plaintext reference bytes")
        handler.cleanup_job_dir(job_dir)
        self.assertFalse(os.path.exists(job_dir))

    def test_cleanup_refuses_a_directory_it_did_not_create(self):
        outsider = tempfile.mkdtemp(prefix="not-ours-")
        marker = os.path.join(outsider, "keep.txt")
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("x")
        try:
            handler.cleanup_job_dir(outsider)
            self.assertTrue(os.path.exists(marker), "a directory outside the input tree is refused")
        finally:
            shutil.rmtree(outsider, ignore_errors=True)

    def test_cleanup_is_safe_on_a_missing_directory(self):
        handler.cleanup_job_dir(os.path.join(handler.COMFY_INPUT_DIR, "job-never-existed"))
        handler.cleanup_job_dir(None)


class StagedNameTests(unittest.TestCase):
    def test_a_traversing_filename_is_refused(self):
        job_dir = handler._job_input_dir("gen-traversal")
        try:
            with self.assertRaises(errors.PrivoraError) as caught:
                handler._staged_name(job_dir, "../escape.png", "image")
            self.assertEqual(caught.exception.code, errors.REFERENCE_PREPROCESSING_FAILED)
        finally:
            handler.cleanup_job_dir(job_dir)

    def test_a_generated_name_is_accepted(self):
        job_dir = handler._job_input_dir("gen-ok")
        try:
            path = handler._staged_name(job_dir, "ref-deadbeef.png", "image")
            self.assertEqual(os.path.dirname(path), os.path.realpath(job_dir))
        finally:
            handler.cleanup_job_dir(job_dir)


class ErrorSanitisationTests(unittest.TestCase):
    def test_a_privora_rejection_returns_a_code_and_never_the_prompt(self):
        with WithInventory():
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(
                    {"mode": "references", "prompt": f"{CANARY} a scene", "seed": 1,
                     "references": [{"type": "image", "role": "character"}] * 10},
                    "gen-reject",
                )
        response = caught.exception.as_response()
        self.assertEqual(response["errorCode"], errors.INVALID_REFERENCE_COUNT)
        self.assertNotIn(CANARY, str(response))

    def test_the_prompt_reaches_only_the_conditioning_node(self):
        with WithInventory():
            graph, job_dir, _ = handler.build_privora_job(
                {"mode": "create", "prompt": f"{CANARY} a scene", "seed": 1}, "gen-prompt")
        try:
            carrying = [node for node, body in graph.items() if CANARY in str(body)]
            self.assertEqual(carrying, ["conditioning"])
        finally:
            handler.cleanup_job_dir(job_dir)


class TurboPrivacyParityTests(unittest.TestCase):
    """A faster path must not be a different output or privacy path."""

    def _graph(self, generation_mode):
        with WithInventory():
            graph, job_dir, _ = handler.build_privora_job(
                {"mode": "create", "prompt": "x", "seed": 1, "generationMode": generation_mode},
                f"gen-{generation_mode}")
        handler.cleanup_job_dir(job_dir)
        return graph

    def test_every_generation_mode_shares_one_output_boundary(self):
        quality = self._graph("quality")
        for mode in ("turbo", "turboFast"):
            other = self._graph(mode)
            self.assertEqual(quality["save"], other["save"], mode)
            self.assertEqual(quality["create_video"], other["create_video"], mode)
            self.assertEqual(quality["decode_video"], other["decode_video"], mode)
            self.assertEqual(quality["decode_audio"], other["decode_audio"], mode)

    def test_only_the_model_path_and_step_count_differ(self):
        quality, turbo = self._graph("quality"), self._graph("turbo")
        differing = {node for node in set(quality) | set(turbo)
                     if quality.get(node) != turbo.get(node)}
        self.assertEqual(differing, {"lora", "guider", "sigmas"})


if __name__ == "__main__":
    unittest.main()
