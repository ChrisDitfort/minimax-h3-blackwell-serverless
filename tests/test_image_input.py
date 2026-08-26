"""Tests for the optional first-frame image-to-video input path.

Run with:  python -m unittest discover -s tests -v

These exercise validation, node detection, staging and cleanup without needing a GPU,
ComfyUI or a RunPod account. `runpod` and `websocket` are stubbed so handler.py imports
on a plain machine.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import types
import unittest

_TEST_INPUT_DIR = tempfile.mkdtemp(prefix="h3-test-input-")

# handler.py reads its directories at import time, so point it at a scratch dir first.
os.environ["COMFY_INPUT_DIR"] = _TEST_INPUT_DIR
os.environ["COMFY_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="h3-test-output-")
os.environ.setdefault("H3_OUTPUT_MODE", "base64")

# Stub the two dependencies that only exist inside the container image.
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


def make_image(fmt: str = "PNG", size: tuple[int, int] = (64, 64)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 60, 200)).save(buffer, format=fmt)
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def text_to_video_workflow() -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va.safetensors"}},
        "7": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": "a cat", "length": 124},
        },
        "20": {"class_type": "SaveVideo", "inputs": {"video": ["19", 0]}},
    }


def image_to_video_workflow() -> dict:
    """Mirrors the bundled first-frame workflow: loader -> resize -> first_frame."""
    workflow = text_to_video_workflow()
    workflow["5"] = {"class_type": "PixaromaLoadImageMini", "inputs": {"image": "placeholder.png"}}
    workflow["6"] = {"class_type": "PixaromaLongestSide", "inputs": {"image": ["5", 0]}}
    workflow["7"]["inputs"]["first_frame"] = ["6", 0]
    return workflow


class ImageInputTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # handler.py resolves its directories once, at import. If another test module
        # imported it first, that module's scratch dir is the one staging actually uses,
        # so read it back off handler rather than assuming _TEST_INPUT_DIR won the race.
        self.input_dir = handler.COMFY_INPUT_DIR
        os.makedirs(self.input_dir, exist_ok=True)
        for name in os.listdir(self.input_dir):
            os.remove(os.path.join(self.input_dir, name))
        self._original_max_bytes = handler.MAX_IMAGE_BYTES

    def tearDown(self) -> None:
        handler.MAX_IMAGE_BYTES = self._original_max_bytes

    def staged_files(self) -> list[str]:
        return os.listdir(self.input_dir)


class TestTextToVideoUnchanged(ImageInputTestCase):
    def test_no_image_is_a_no_op(self):
        workflow = text_to_video_workflow()
        before = dict(workflow)

        self.assertIsNone(handler.stage_input_image({"workflow": workflow}, workflow))
        self.assertEqual(workflow, before, "text-to-video workflow must not be mutated")
        self.assertEqual(self.staged_files(), [], "no file should be written")

    def test_empty_values_are_treated_as_absent(self):
        workflow = text_to_video_workflow()
        job_input = {"workflow": workflow, "image_url": "", "image_base64": ""}
        self.assertIsNone(handler.stage_input_image(job_input, workflow))


class TestInvalidCombination(ImageInputTestCase):
    def test_both_inputs_rejected(self):
        workflow = image_to_video_workflow()
        job_input = {
            "workflow": workflow,
            "image_url": "https://example.com/a.png",
            "image_base64": b64(make_image()),
        }
        with self.assertRaises(handler.ImageInputError) as context:
            handler.stage_input_image(job_input, workflow)
        self.assertIn("not both", str(context.exception))
        self.assertEqual(self.staged_files(), [])


class TestBase64Input(ImageInputTestCase):
    def test_png_is_staged_and_wired(self):
        workflow = image_to_video_workflow()
        path = handler.stage_input_image(
            {"workflow": workflow, "image_base64": b64(make_image("PNG"))}, workflow
        )

        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))

        filename = os.path.basename(path)
        self.assertTrue(filename.startswith("h3-input-"))
        self.assertTrue(filename.endswith(".png"))
        # The loader node, not the resize node, receives the filename.
        self.assertEqual(workflow["5"]["inputs"]["image"], filename)
        self.assertEqual(self.staged_files(), [filename])

    def test_jpeg_and_webp_supported(self):
        for fmt, extension in (("JPEG", ".jpg"), ("WEBP", ".webp")):
            with self.subTest(fmt=fmt):
                workflow = image_to_video_workflow()
                path = handler.stage_input_image(
                    {"workflow": workflow, "image_base64": b64(make_image(fmt))}, workflow
                )
                self.assertTrue(path.endswith(extension))
                handler.cleanup_input_image(path)

    def test_data_uri_prefix_accepted(self):
        workflow = image_to_video_workflow()
        payload = "data:image/png;base64," + b64(make_image("PNG"))
        path = handler.stage_input_image({"workflow": workflow, "image_base64": payload}, workflow)
        self.assertTrue(os.path.isfile(path))

    def test_invalid_base64_rejected(self):
        workflow = image_to_video_workflow()
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_image({"workflow": workflow, "image_base64": "!!!not base64!!!"}, workflow)
        self.assertEqual(self.staged_files(), [])


class TestRejectedContent(ImageInputTestCase):
    def test_non_image_bytes_rejected(self):
        workflow = image_to_video_workflow()
        payload = b64(b"#!/bin/sh\necho pwned\n")
        with self.assertRaises(handler.ImageInputError) as context:
            handler.stage_input_image({"workflow": workflow, "image_base64": payload}, workflow)
        self.assertIn("Unsupported image format", str(context.exception))
        self.assertEqual(self.staged_files(), [], "rejected content must not be written")

    def test_unsupported_image_format_rejected(self):
        workflow = image_to_video_workflow()
        gif = make_image("GIF")
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_image({"workflow": workflow, "image_base64": b64(gif)}, workflow)
        self.assertEqual(self.staged_files(), [])

    def test_forged_magic_bytes_rejected(self):
        """PNG signature on non-image data must not pass: Pillow has to parse it too."""
        workflow = image_to_video_workflow()
        forged = b"\x89PNG\r\n\x1a\n" + b"totally not a png" * 10
        with self.assertRaises(handler.ImageInputError) as context:
            handler.stage_input_image({"workflow": workflow, "image_base64": b64(forged)}, workflow)
        self.assertIn("not a decodable image", str(context.exception))
        self.assertEqual(self.staged_files(), [])

    def test_oversized_image_rejected_without_temp_file(self):
        workflow = image_to_video_workflow()
        handler.MAX_IMAGE_BYTES = 512
        with self.assertRaises(handler.ImageInputError) as context:
            handler.stage_input_image(
                {"workflow": workflow, "image_base64": b64(make_image("PNG", (512, 512)))}, workflow
            )
        self.assertIn("limit", str(context.exception))
        self.assertEqual(self.staged_files(), [], "oversized input must leave nothing behind")


class TestSSRFGuards(ImageInputTestCase):
    def test_http_scheme_rejected(self):
        with self.assertRaises(handler.ImageInputError) as context:
            handler._assert_fetchable_url("http://example.com/a.png")
        self.assertIn("https", str(context.exception))

    def test_non_http_scheme_rejected(self):
        for url in ("file:///etc/passwd", "gopher://example.com/", "ftp://example.com/a.png"):
            with self.subTest(url=url), self.assertRaises(handler.ImageInputError):
                handler._assert_fetchable_url(url)

    def test_loopback_and_private_addresses_rejected(self):
        urls = [
            "https://127.0.0.1/a.png",
            "https://localhost/a.png",
            "https://[::1]/a.png",
            "https://10.0.0.5/a.png",
            "https://192.168.1.10/a.png",
            "https://172.16.4.4/a.png",
            "https://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
        ]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(handler.ImageInputError) as context:
                handler._assert_fetchable_url(url)
            self.assertIn("not allowed", str(context.exception).lower() + " not allowed")


class TestNodeDetection(ImageInputTestCase):
    def test_detects_loader_through_intermediate_node(self):
        """A decoy loader must not win over the one actually feeding first_frame."""
        workflow = image_to_video_workflow()
        workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": "decoy.png"}}

        self.assertEqual(handler.find_image_node(workflow, None), "5")

    def test_single_loader_is_used(self):
        workflow = text_to_video_workflow()
        workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": "only.png"}}
        self.assertEqual(handler.find_image_node(workflow, None), "9")

    def test_ambiguous_workflow_requires_explicit_node(self):
        workflow = text_to_video_workflow()
        workflow["8"] = {"class_type": "LoadImage", "inputs": {"image": "a.png"}}
        workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": "b.png"}}

        with self.assertRaises(handler.ImageInputError) as context:
            handler.find_image_node(workflow, None)
        self.assertIn("image_node_id", str(context.exception))

    def test_explicit_node_id_wins(self):
        workflow = image_to_video_workflow()
        workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": "decoy.png"}}
        self.assertEqual(handler.find_image_node(workflow, "9"), "9")
        self.assertEqual(handler.find_image_node(workflow, 9), "9", "int ids accepted")

    def test_explicit_node_id_must_exist_and_accept_images(self):
        workflow = image_to_video_workflow()
        with self.assertRaises(handler.ImageInputError):
            handler.find_image_node(workflow, "999")
        with self.assertRaises(handler.ImageInputError):
            handler.find_image_node(workflow, "1")  # UNETLoader has no image input

    def test_image_without_any_loader_is_rejected(self):
        workflow = text_to_video_workflow()
        with self.assertRaises(handler.ImageInputError) as context:
            handler.stage_input_image(
                {"workflow": workflow, "image_base64": b64(make_image())}, workflow
            )
        self.assertIn("no image-loader node", str(context.exception))
        self.assertEqual(self.staged_files(), [], "unusable workflow must not leave a file")


class TestShippedExamples(ImageInputTestCase):
    """The workflows in examples/ must behave the way the README documents."""

    EXAMPLES = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
    )

    def load(self, name: str) -> dict:
        import json

        with open(os.path.join(self.EXAMPLES, name), encoding="utf-8") as handle:
            return json.load(handle)

    def test_text_to_video_example_needs_no_image(self):
        workflow = self.load("fl2va-text-to-video.json")
        self.assertIsNone(handler.stage_input_image({"workflow": workflow}, workflow))
        self.assertEqual(self.staged_files(), [])

    def test_image_to_video_example_autodetects_the_load_image_node(self):
        workflow = self.load("fl2va-image-to-video.json")
        self.assertEqual(handler.find_image_node(workflow, None), "15")

        path = handler.stage_input_image(
            {"workflow": workflow, "image_base64": b64(make_image())}, workflow
        )
        self.assertEqual(workflow["15"]["inputs"]["image"], os.path.basename(path))
        self.assertNotEqual(workflow["15"]["inputs"]["image"], "placeholder.png")

    def test_text_to_video_example_rejects_an_image(self):
        """It has no loader node, so an image must be refused rather than misrouted."""
        workflow = self.load("fl2va-text-to-video.json")
        with self.assertRaises(handler.ImageInputError):
            handler.stage_input_image(
                {"workflow": workflow, "image_base64": b64(make_image())}, workflow
            )
        self.assertEqual(self.staged_files(), [])


class TestCleanup(ImageInputTestCase):
    def test_cleanup_removes_staged_file(self):
        workflow = image_to_video_workflow()
        path = handler.stage_input_image(
            {"workflow": workflow, "image_base64": b64(make_image())}, workflow
        )
        self.assertEqual(len(self.staged_files()), 1)

        handler.cleanup_input_image(path)
        self.assertEqual(self.staged_files(), [], "staged image must be removed")

    def test_cleanup_is_idempotent_and_none_safe(self):
        handler.cleanup_input_image(None)
        handler.cleanup_input_image(os.path.join(self.input_dir, "h3-input-missing.png"))

    def test_cleanup_refuses_files_it_did_not_create(self):
        """Shared inputs and model files must never be deleted by cleanup."""
        bystander = os.path.join(self.input_dir, "BallerinaBunny.png")
        with open(bystander, "wb") as handle:
            handle.write(make_image())

        handler.cleanup_input_image(bystander)
        self.assertTrue(os.path.isfile(bystander), "unrelated input must survive cleanup")

    def test_cleanup_refuses_paths_outside_the_input_dir(self):
        outside = os.path.join(tempfile.gettempdir(), "h3-input-elsewhere.png")
        with open(outside, "wb") as handle:
            handle.write(b"keep me")
        try:
            handler.cleanup_input_image(outside)
            self.assertTrue(os.path.isfile(outside), "files outside the input dir must survive")
        finally:
            os.remove(outside)


class TestFilenameSafety(ImageInputTestCase):
    def test_remote_filename_is_never_used(self):
        workflow = image_to_video_workflow()
        path = handler.stage_input_image(
            {"workflow": workflow, "image_base64": b64(make_image())}, workflow
        )
        filename = os.path.basename(path)
        self.assertNotIn("..", filename)
        self.assertNotIn("/", filename)
        self.assertTrue(filename.startswith("h3-input-"))

    def test_filenames_are_unique_per_job(self):
        names = set()
        for _ in range(5):
            workflow = image_to_video_workflow()
            path = handler.stage_input_image(
                {"workflow": workflow, "image_base64": b64(make_image())}, workflow
            )
            names.add(os.path.basename(path))
        self.assertEqual(len(names), 5, "concurrent jobs must not collide")

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are not meaningful on Windows")
    def test_staged_file_is_not_executable(self):
        import stat

        workflow = image_to_video_workflow()
        path = handler.stage_input_image(
            {"workflow": workflow, "image_base64": b64(make_image())}, workflow
        )
        mode = stat.S_IMODE(os.stat(path).st_mode)
        self.assertEqual(mode, 0o644, "staged image must be plain data, mode 0644")
        self.assertFalse(mode & 0o111, "uploaded content must never be executable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
