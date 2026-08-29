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
from unittest import mock

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
from privora import errors, media, models  # noqa: E402

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


class HandlerEntrypointTests(unittest.TestCase):
    """handler() itself, not just the bridge underneath it.

    Every other test in this file calls build_privora_job() directly. That left the
    routing in handler() untested, and it was broken: generation_id was read by the
    PrivoraVideo branch and only assigned below it, so a canonical request raised
    UnboundLocalError before it could reach a model. The legacy path never entered that
    branch, so the whole suite stayed green and a live legacy job still passed.

    These tests exercise the entrypoint and stop at ComfyUI, which is the first thing
    that genuinely needs a GPU.
    """

    def _run(self, job_input, inventory=FULL_INVENTORY):
        reached = {}

        def fake_start():
            reached["comfy"] = True
            raise handler.WorkflowError("stop: ComfyUI is out of scope for this test")

        saved_inventory, saved_start = handler.MODEL_INVENTORY, handler.start_comfyui
        handler.MODEL_INVENTORY, handler.start_comfyui = inventory, fake_start
        try:
            return handler.handler({"id": "test-job", "input": job_input}), reached
        finally:
            handler.MODEL_INVENTORY, handler.start_comfyui = saved_inventory, saved_start

    def test_a_canonical_create_request_is_routed_and_built(self):
        result, reached = self._run(
            {"mode": "create", "prompt": "a calm sea", "quality": "standard",
             "aspectRatio": "16:9", "duration": 5, "seed": 7}
        )
        # It got all the way to ComfyUI, which means routing, validation, canvas
        # resolution and graph construction all ran.
        self.assertTrue(reached.get("comfy"), f"never reached ComfyUI: {result}")
        self.assertNotIn("UnboundLocalError", str(result))

    def test_a_canonical_request_never_fails_with_an_unbound_local(self):
        for mode, extra in (
            ("create", {}),
            ("animate", {"firstFrame": {"url": "https://example.invalid/a.png"}}),
            ("references", {"references": [{"type": "image", "role": "character",
                                            "url": "https://example.invalid/a.png"}]}),
        ):
            with self.subTest(mode=mode):
                result, _ = self._run({"mode": mode, "prompt": "x", "seed": 1, **extra})
                self.assertNotIn("UnboundLocalError", str(result))

    def test_capabilities_is_answered_without_touching_comfyui(self):
        result, reached = self._run({"mode": "capabilities"})
        self.assertIn("capabilities", result)
        self.assertFalse(reached.get("comfy"), "a capabilities probe must not start ComfyUI")
        self.assertIn("modes", result["capabilities"])
        self.assertIn("generationModes", result["capabilities"])

    def test_a_legacy_workflow_request_still_reaches_comfyui(self):
        result, reached = self._run(
            {"workflow": {"1": {"class_type": "UNETLoader", "inputs": {}}}}
        )
        self.assertTrue(reached.get("comfy"), f"legacy path regressed: {result}")


class StagingFailureTests(unittest.TestCase):
    """A rejected job must not leave reference plaintext on a warm worker."""

    PLAINTEXT = b"CONFIDENTIAL_REFERENCE_PLAINTEXT_CANARY"

    def _job_path(self, generation_id):
        return os.path.join(os.path.realpath(handler.COMFY_INPUT_DIR), "job-" + generation_id)

    def _assert_job_removed(self, generation_id):
        self.assertFalse(
            os.path.exists(self._job_path(generation_id)),
            "a failed staging run left its plaintext job directory behind",
        )

    def _write_stage(self, reference, kind, job_dir):
        index = len(os.listdir(job_dir)) + 1
        filename = f"ref-{index}.{'png' if kind == 'image' else kind}"
        path = handler._staged_name(job_dir, filename, kind)
        with open(path, "wb") as handle:
            handle.write(self.PLAINTEXT)
        return os.path.join(os.path.basename(job_dir), filename)

    def test_a_failed_reference_leaves_no_staged_files_behind(self):
        before = set(os.listdir(handler.COMFY_INPUT_DIR))
        with WithInventory():
            with self.assertRaises(errors.PrivoraError):
                handler.build_privora_job(
                    {"mode": "references", "prompt": "x", "seed": 1,
                     "references": [{"type": "image", "role": "character"}]},
                    "gen-staging-failure",
                )
        self.assertEqual(
            set(os.listdir(handler.COMFY_INPUT_DIR)), before,
            "a failed staging run left its job directory behind",
        )

    def test_image_failure_after_one_successful_stage_is_cleaned(self):
        generation_id = "cleanup-image-failure"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [
                {"type": "image", "role": "character", "url": "https://x/1"},
                {"type": "image", "role": "style", "url": "https://x/2"},
            ],
        }
        calls = 0

        def stage(reference, job_dir):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise handler.ImageInputError("invalid second image")
            return self._write_stage(reference, "image", job_dir)

        with WithInventory(), mock.patch.object(handler, "_stage_image_reference", stage):
            with self.assertRaises(handler.ImageInputError):
                handler.build_privora_job(payload, generation_id)
        self.assertEqual(calls, 2)
        self._assert_job_removed(generation_id)

    def test_video_failure_after_a_successful_image_stage_is_cleaned(self):
        generation_id = "cleanup-video-failure"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [
                {"type": "image", "role": "character", "url": "https://x/image"},
                {"type": "video", "role": "motion", "url": "https://x/video"},
            ],
        }
        with (
            WithInventory(),
            mock.patch.object(
                handler, "_stage_image_reference",
                side_effect=lambda reference, job_dir: self._write_stage(
                    reference, "image", job_dir
                ),
            ),
            mock.patch.object(
                handler, "_stage_media_reference",
                side_effect=errors.PrivoraError(
                    errors.REFERENCE_PREPROCESSING_FAILED,
                    "The supplied video reference could not be read.", {"type": "video"},
                ),
            ),
        ):
            with self.assertRaises(errors.PrivoraError):
                handler.build_privora_job(payload, generation_id)
        self._assert_job_removed(generation_id)

    def test_audio_failure_after_a_successful_image_stage_is_cleaned(self):
        generation_id = "cleanup-audio-failure"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [
                {"type": "image", "role": "character", "url": "https://x/image"},
                {"type": "audio", "role": "music", "url": "https://x/audio"},
            ],
        }
        with (
            WithInventory(),
            mock.patch.object(
                handler, "_stage_image_reference",
                side_effect=lambda reference, job_dir: self._write_stage(
                    reference, "image", job_dir
                ),
            ),
            mock.patch.object(
                handler, "_stage_media_reference",
                side_effect=errors.PrivoraError(
                    errors.REFERENCE_PREPROCESSING_FAILED,
                    "The supplied audio reference could not be read.", {"type": "audio"},
                ),
            ),
        ):
            with self.assertRaises(errors.PrivoraError):
                handler.build_privora_job(payload, generation_id)
        self._assert_job_removed(generation_id)

    def test_invalid_later_reference_after_a_successful_stage_is_cleaned(self):
        generation_id = "cleanup-invalid-later"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [
                {"type": "image", "role": "character", "url": "https://x/first"},
                {"type": "image", "role": "style"},
            ],
        }
        with (
            WithInventory(),
            mock.patch.object(handler, "_fetch_asset_bytes", return_value=self.PLAINTEXT),
            mock.patch.object(handler, "_validate_image_bytes", return_value=".png"),
        ):
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(payload, generation_id)
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)
        self._assert_job_removed(generation_id)

    def test_fetch_timeout_after_a_successful_stage_is_cleaned(self):
        generation_id = "cleanup-fetch-timeout"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [
                {"type": "image", "role": "character", "url": "https://x/first"},
                {"type": "image", "role": "style", "url": "https://x/second"},
            ],
        }
        timeout = handler.requests.exceptions.Timeout("signed-url-canary")
        with (
            WithInventory(),
            mock.patch.object(handler, "_fetch_asset_bytes",
                              side_effect=[self.PLAINTEXT, timeout]),
            mock.patch.object(handler, "_validate_image_bytes", return_value=".png"),
        ):
            with self.assertRaises(handler.requests.exceptions.Timeout):
                handler.build_privora_job(payload, generation_id)
        self._assert_job_removed(generation_id)

    def test_post_staging_duration_validation_failure_is_cleaned(self):
        generation_id = "cleanup-validation-failure"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [
                {"type": "video", "role": "motion", "url": "https://x/video"},
            ],
        }

        def stage(reference, kind, job_dir):
            reference.duration_seconds = 20.0
            return self._write_stage(reference, kind, job_dir)

        with WithInventory(), mock.patch.object(handler, "_stage_media_reference", stage):
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(payload, generation_id)
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_DURATION)
        self._assert_job_removed(generation_id)

    def test_measured_soundtrack_duration_is_rejected_and_cleaned(self):
        generation_id = "cleanup-soundtrack-duration"
        payload = {
            "mode": "references", "prompt": "x",
            "references": [{
                "type": "video", "role": "motion", "url": "https://x/video",
                "soundtrack": {
                    "type": "audio", "role": "ambience", "url": "https://x/audio",
                },
            }],
        }

        def stage(reference, kind, job_dir):
            reference.duration_seconds = 8.0 if kind == "video" else 20.0
            return self._write_stage(reference, kind, job_dir)

        with WithInventory(), mock.patch.object(handler, "_stage_media_reference", stage):
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(payload, generation_id)
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_DURATION)
        self.assertEqual(caught.exception.details["type"], "soundtrack")
        self._assert_job_removed(generation_id)

    def test_a_reference_with_no_source_is_a_coded_client_error(self):
        with WithInventory():
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(
                    {"mode": "references", "prompt": "x", "seed": 1,
                     "references": [{"type": "image", "role": "character"}]},
                    "gen-no-source",
                )
        response = caught.exception.as_response()
        self.assertEqual(response["errorCode"], errors.INVALID_REFERENCE_TYPE)
        self.assertTrue(caught.exception.is_client_error)

    def test_a_reference_with_two_sources_is_refused(self):
        with WithInventory():
            with self.assertRaises(errors.PrivoraError) as caught:
                handler.build_privora_job(
                    {"mode": "references", "prompt": "x", "seed": 1,
                     "references": [{"type": "image", "role": "character",
                                     "url": "https://example.invalid/a.png",
                                     "data": "AAAA"}]},
                    "gen-two-sources",
                )
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)


class ReferenceFetchLimitTests(unittest.TestCase):
    """Video and audio references share the image fetcher; they must not share its cap.

    capabilities() advertises 256 MB of video and 64 MB of audio. Before this was
    parameterised, every reference went through the image path's 32 MB limit and an
    `Accept: image/*` header, so a supported reference could be refused by our own
    downloader - or by a strict origin answering 406.
    """

    def _capture(self, kind):
        seen = {}

        def fake_download(url, bearer_token=None, *, max_bytes, accept, what):
            seen.update(max_bytes=max_bytes, accept=accept, what=what)
            raise handler.ImageInputError("stop: the network is out of scope here")

        job_dir = tempfile.mkdtemp()
        saved = handler._download_image
        handler._download_image = fake_download
        try:
            reference = types.SimpleNamespace(
                url="https://example.invalid/a.bin", data_base64=None, token=None,
                type=kind, duration_seconds=None,
            )
            with self.assertRaises(handler.ImageInputError):
                if kind == "image":
                    handler._stage_image_reference(reference, job_dir)
                else:
                    handler._stage_media_reference(reference, kind, job_dir)
        finally:
            handler._download_image = saved
            shutil.rmtree(job_dir, ignore_errors=True)
        return seen

    def test_a_video_reference_uses_the_video_cap(self):
        seen = self._capture("video")
        self.assertEqual(seen["max_bytes"], media.MAX_VIDEO_BYTES)
        self.assertEqual(seen["accept"], "video/*, application/octet-stream")

    def test_an_audio_reference_uses_the_audio_cap(self):
        seen = self._capture("audio")
        self.assertEqual(seen["max_bytes"], media.MAX_AUDIO_BYTES)
        self.assertEqual(seen["accept"], "audio/*, application/octet-stream")

    def test_an_image_reference_keeps_the_image_cap(self):
        seen = self._capture("image")
        self.assertEqual(seen["max_bytes"], handler.MAX_IMAGE_BYTES)
        self.assertEqual(seen["accept"], "image/*")

    def test_a_fetch_timeout_does_not_echo_url_or_bearer_material(self):
        url_secret = "URL_SECRET_CANARY_51A9"
        bearer_secret = "BEARER_SECRET_CANARY_20F3"
        timeout = handler.requests.exceptions.Timeout(
            f"timed out fetching https://example.com/?token={url_secret}"
        )
        with (
            mock.patch.object(handler, "_assert_fetchable_url"),
            mock.patch.object(handler.requests, "get", side_effect=timeout),
        ):
            with self.assertRaises(handler.ImageInputError) as caught:
                handler._download_image(
                    f"https://example.com/?token={url_secret}",
                    bearer_token=bearer_secret,
                    what="reference video",
                )
        rendered = str(caught.exception)
        self.assertNotIn(url_secret, rendered)
        self.assertNotIn(bearer_secret, rendered)
        self.assertEqual(rendered, "Downloading reference video failed.")


class CapabilityBuildIdentityTests(unittest.TestCase):
    def test_capabilities_exposes_safe_immutable_build_identity(self):
        values = {
            "H3_BUILD_SOURCE_COMMIT": "abc123def456",
            "H3_BUILD_IMAGE_TAG": "multimodal-3",
            "H3_BUILD_ID": "98765-1",
            "COMFYUI_H3_COMMIT": "dec5d945",
        }
        with WithInventory(), mock.patch.dict(os.environ, values, clear=False):
            worker = handler.worker_capabilities()["worker"]
        self.assertEqual(worker["build"], {
            "sourceCommit": values["H3_BUILD_SOURCE_COMMIT"],
            "imageTag": values["H3_BUILD_IMAGE_TAG"],
            "buildId": values["H3_BUILD_ID"],
        })
        self.assertEqual(worker["comfyuiCommit"], "dec5d945")
        self.assertNotIn("path", str(worker).lower())


if __name__ == "__main__":
    unittest.main()
