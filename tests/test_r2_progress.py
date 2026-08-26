"""Tests for keyframe staging, R2-backed output upload and progress callbacks.

Run with:  python -m unittest discover -s tests -v

Three themes, all of which are really about failure behaviour:

  * Two independent keyframes must land on the right nodes, or be rejected - never
    silently put on the wrong end of the clip.
  * A failed upload must fail the job. Reporting COMPLETED with no playable video is the
    single worst outcome available here.
  * A progress callback must never be able to fail a generation that has already cost
    minutes of GPU time.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-r2-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-r2-output-"))
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


def png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def keyframe_workflow(first: bool = True, last: bool = True) -> dict:
    """A graph shaped like the Worker's first/last templates."""
    cond_inputs = {"clip": ["2", 0], "vae": ["3", 0], "prompt": "x"}
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs},
    }
    if first:
        workflow["10"] = {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}}
        cond_inputs["first_frame"] = ["10", 0]
    if last:
        workflow["11"] = {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}}
        cond_inputs["last_frame"] = ["11", 0]
    return workflow


class KeyframeStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir = handler.COMFY_INPUT_DIR
        os.makedirs(self.input_dir, exist_ok=True)
        for name in os.listdir(self.input_dir):
            os.remove(os.path.join(self.input_dir, name))
        self.png = png_bytes()

    def tearDown(self) -> None:
        for name in os.listdir(self.input_dir):
            os.remove(os.path.join(self.input_dir, name))

    def staged_names(self) -> list[str]:
        return sorted(os.listdir(self.input_dir))

    def test_text_only_stages_nothing(self):
        workflow = keyframe_workflow(first=False, last=False)
        self.assertEqual(handler.stage_input_assets({}, workflow), [])
        self.assertEqual(self.staged_names(), [])

    def test_first_frame_only(self):
        workflow = keyframe_workflow(first=True, last=False)
        staged = handler.stage_input_assets(
            {"assets": {"first_frame": {"base64": b64(self.png)}}}, workflow
        )
        self.assertEqual(len(staged), 1)
        self.assertNotEqual(workflow["10"]["inputs"]["image"], "placeholder.png")

    def test_last_frame_only(self):
        workflow = keyframe_workflow(first=False, last=True)
        staged = handler.stage_input_assets(
            {"assets": {"last_frame": {"base64": b64(self.png)}}}, workflow
        )
        self.assertEqual(len(staged), 1)
        self.assertNotEqual(workflow["11"]["inputs"]["image"], "placeholder.png")

    def test_first_and_last_land_on_their_own_nodes(self):
        """The core of the feature: two images, two loaders, no crossover."""
        workflow = keyframe_workflow()
        staged = handler.stage_input_assets(
            {
                "assets": {
                    "first_frame": {"base64": b64(self.png)},
                    "last_frame": {"base64": b64(self.png)},
                }
            },
            workflow,
        )

        self.assertEqual(len(staged), 2)
        first_name = workflow["10"]["inputs"]["image"]
        last_name = workflow["11"]["inputs"]["image"]
        self.assertNotEqual(first_name, "placeholder.png")
        self.assertNotEqual(last_name, "placeholder.png")
        self.assertNotEqual(first_name, last_name, "keyframes must not share one file")
        self.assertEqual(len(self.staged_names()), 2)

    def test_legacy_single_image_fields_still_work(self):
        workflow = keyframe_workflow(first=True, last=False)
        path = handler.stage_input_image({"image_base64": b64(self.png)}, workflow)
        self.assertIsNotNone(path)
        self.assertNotEqual(workflow["10"]["inputs"]["image"], "placeholder.png")

    def test_legacy_and_assets_together_is_rejected(self):
        workflow = keyframe_workflow()
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_assets(
                {
                    "image_base64": b64(self.png),
                    "assets": {"first_frame": {"base64": b64(self.png)}},
                },
                workflow,
            )

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(handler.ImageInputError) as ctx:
            handler.stage_input_assets({"assets": {"middle_frame": {"base64": "x"}}}, keyframe_workflow())
        self.assertIn("middle_frame", str(ctx.exception))

    def test_missing_loader_for_role_is_rejected(self):
        """Asking for a last frame on a first-frame-only graph must fail, not silently drop."""
        workflow = keyframe_workflow(first=True, last=False)
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_assets(
                {"assets": {"last_frame": {"base64": b64(self.png)}}}, workflow
            )

    def test_both_sources_on_one_asset_is_rejected(self):
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_assets(
                {"assets": {"first_frame": {"base64": b64(self.png), "url": "https://x/a.png"}}},
                keyframe_workflow(),
            )

    def test_a_failed_second_asset_leaves_no_files_behind(self):
        """Partial staging would orphan files in the input dir on every bad request."""
        workflow = keyframe_workflow()
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_assets(
                {
                    "assets": {
                        "first_frame": {"base64": b64(self.png)},
                        "last_frame": {"base64": "not-valid-base64!!"},
                    }
                },
                workflow,
            )
        self.assertEqual(self.staged_names(), [], "a rejected request must stage nothing")

    def test_invalid_image_bytes_are_rejected(self):
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_assets(
                {"assets": {"first_frame": {"base64": b64(b"not an image")}}},
                keyframe_workflow(),
            )

    def test_assets_must_be_an_object(self):
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_assets({"assets": ["first_frame"]}, keyframe_workflow())


class OutputUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._put = handler.requests.put
        self.tmp = tempfile.mkdtemp(prefix="h3-upload-")
        self.video = os.path.join(self.tmp, "v.mp4")
        with open(self.video, "wb") as fh:
            fh.write(b"\x00\x01" * 2048)

    def tearDown(self) -> None:
        handler.requests.put = self._put

    def fake_put(self, status=200, body=None, capture=None):
        # Mirrors the Worker's real acknowledgement: a 2xx *and* a JSON body carrying the
        # key. Both are required now, because a 2xx alone can come from an auth gateway.
        payload = body if body is not None else {"key": "outputs/j/video.mp4"}

        def _put(url, data=None, headers=None, timeout=None, allow_redirects=None):
            if capture is not None:
                capture["url"] = url
                capture["headers"] = headers or {}
                capture["allow_redirects"] = allow_redirects
                capture["body"] = data.read() if hasattr(data, "read") else data
            return types.SimpleNamespace(
                ok=status < 400,
                status_code=status,
                text="err",
                headers={},
                json=lambda: payload,
            )

        return _put

    def test_uploads_raw_bytes_not_base64(self):
        capture = {}
        handler.requests.put = self.fake_put(capture=capture, body={"key": "outputs/j/video.mp4"})

        store = handler.WorkerUploadStore("https://w/internal/jobs/j/output", "tok", 30)
        result = store.store(self.video, {"filename": "v.mp4"})

        self.assertEqual(capture["headers"]["Content-Type"], "video/mp4")
        self.assertEqual(capture["headers"]["Authorization"], "Bearer tok")
        self.assertEqual(capture["body"], b"\x00\x01" * 2048, "body must be the raw MP4")
        self.assertNotIn("data", result, "R2 path must not return base64")
        self.assertEqual(result["size"], 4096)
        self.assertEqual(result["key"], "outputs/j/video.mp4")

    def test_worker_supplied_key_is_echoed_not_invented(self):
        handler.requests.put = self.fake_put(body={"key": "outputs/u/j/video.mp4"})
        store = handler.WorkerUploadStore("https://w/o", None, 30)
        result = store.store(self.video, {"filename": "v.mp4"})
        self.assertEqual(result["key"], "outputs/u/j/video.mp4")

    def test_http_error_raises_so_the_job_fails(self):
        handler.requests.put = self.fake_put(status=403)
        store = handler.WorkerUploadStore("https://w/o", "tok", 30)
        with self.assertRaises(handler.WorkflowError):
            store.store(self.video, {"filename": "v.mp4"})

    def test_transport_error_raises_so_the_job_fails(self):
        def boom(*a, **k):
            raise OSError("connection reset")

        handler.requests.put = boom
        store = handler.WorkerUploadStore("https://w/o", "tok", 30)
        with self.assertRaises(handler.WorkflowError):
            store.store(self.video, {"filename": "v.mp4"})

    def test_upload_time_and_bytes_are_recorded(self):
        handler.requests.put = self.fake_put()
        store = handler.WorkerUploadStore("https://w/o", None, 30)
        store.store(self.video, {"filename": "v.mp4"})
        self.assertEqual(store.uploaded_bytes, 4096)
        self.assertGreaterEqual(store.upload_seconds, 0.0)


class OutputStoreSelectionTest(unittest.TestCase):
    def test_an_output_block_selects_the_worker_upload_path(self):
        store, upload = handler.build_output_store_for_job(
            {"output": {"url": "https://w/internal/jobs/j/output", "token": "t"}}
        )
        self.assertIsInstance(store, handler.WorkerUploadStore)
        self.assertIs(store, upload)

    def test_without_an_output_block_the_configured_store_is_used(self):
        store, upload = handler.build_output_store_for_job({})
        self.assertIsNone(upload, "no upload store means base64 compatibility is preserved")
        self.assertNotIsInstance(store, handler.WorkerUploadStore)


class ProgressReporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._post = handler.requests.post

    def tearDown(self) -> None:
        handler.requests.post = self._post

    def capture_posts(self, status=200):
        sent = []

        def _post(url, json=None, headers=None, timeout=None, allow_redirects=None):
            sent.append({"url": url, "json": json, "headers": headers or {}, "timeout": timeout})
            return types.SimpleNamespace(ok=status < 400, status_code=status, headers={})

        handler.requests.post = _post
        return sent

    def test_disabled_without_a_url(self):
        sent = self.capture_posts()
        reporter = handler.ProgressReporter("job-1")
        self.assertFalse(reporter.enabled)
        reporter.phase(handler.PHASE_SAMPLING)
        reporter.step(1, 20)
        self.assertEqual(sent, [], "no endpoint means no traffic at all")

    def test_phase_events_carry_job_and_token(self):
        sent = self.capture_posts()
        reporter = handler.ProgressReporter("job-1", "https://w/p", "tok")
        reporter.phase(handler.PHASE_DECODING, percent=90)

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["json"]["jobId"], "job-1")
        self.assertEqual(sent[0]["json"]["phase"], "decoding")
        self.assertEqual(sent[0]["json"]["percent"], 90)
        self.assertEqual(sent[0]["headers"]["Authorization"], "Bearer tok")

    def test_step_events_include_percent(self):
        sent = self.capture_posts()
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        reporter.step(7, 20)
        self.assertEqual(sent[0]["json"], {"jobId": "job-1", "phase": "sampling", "step": 7, "steps": 20, "percent": 35})

    def test_steps_are_coalesced_but_the_last_one_always_lands(self):
        sent = self.capture_posts()
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        for step in range(1, 21):
            reporter.step(step, 20)

        self.assertLess(len(sent), 20, "a fast sampler must not flood the Worker")
        self.assertEqual(sent[-1]["json"]["step"], 20, "the final step must never be dropped")
        self.assertEqual(sent[-1]["json"]["percent"], 100)

    def test_a_failing_callback_never_raises(self):
        def boom(*a, **k):
            raise OSError("cloudflare unreachable")

        handler.requests.post = boom
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        reporter.phase(handler.PHASE_SAMPLING)  # must not raise
        reporter.step(1, 20)
        self.assertEqual(reporter.sent, 0)
        self.assertEqual(reporter.failed, 2)

    def test_an_http_error_is_counted_not_raised(self):
        self.capture_posts(status=500)
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        reporter.phase(handler.PHASE_SAMPLING)
        self.assertEqual(reporter.failed, 1)
        self.assertEqual(reporter.sent, 0)

    def test_uses_a_short_timeout(self):
        sent = self.capture_posts()
        handler.ProgressReporter("job-1", "https://w/p").phase(handler.PHASE_SAMPLING)
        self.assertLessEqual(sent[0]["timeout"], 10, "callbacks must not stall a job")

    def test_no_raw_comfyui_text_is_forwarded(self):
        sent = self.capture_posts()
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        reporter.phase(handler.PHASE_SAMPLING)
        reporter.step(3, 20)
        for event in sent:
            self.assertEqual(
                set(event["json"]) - {"jobId", "phase", "step", "steps", "percent"},
                set(),
                "only the stable public vocabulary may be sent",
            )


class PerfFieldsTest(unittest.TestCase):
    def test_new_fields_appear_only_when_measured(self):
        timer = handler.JobTimer({})
        plain = timer.summary(job_index=1, status="ok")
        self.assertNotIn("output_upload=", plain)
        self.assertNotIn("output_bytes=", plain)

        timer.add_span("input_download", 1.2)
        timer.add_span("output_upload", 1.6)
        timer.output_bytes = 4654488
        timer.progress_callbacks = (21, 1)
        line = timer.summary(job_index=1, status="ok")

        self.assertIn("input_download=1.2s", line)
        self.assertIn("output_upload=1.6s", line)
        self.assertIn("output_bytes=4654488", line)
        self.assertIn("progress_callbacks=21/22", line)

    def test_callback_time_is_not_folded_into_sampling(self):
        timer = handler.JobTimer({})
        timer.add_span("progress_callback_time", 0.9)
        line = timer.summary(job_index=1, status="ok")
        self.assertIn("progress_time=0.9s", line)
        self.assertIn("sampling=n/a", line, "callback time must never be counted as sampling")


if __name__ == "__main__":
    unittest.main(verbosity=2)
