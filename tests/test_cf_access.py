"""Tests for Cloudflare Access service-token headers on RunPod -> Worker callbacks.

Run with:  python -m unittest discover -s tests -v

The Worker sits behind Cloudflare Access, so every callback needs two independent
credentials and both must be present: the Access service token decides whether the
request reaches the Worker at all, and the job-scoped bearer token decides what it is
allowed to do once it gets there. Sending only one of them fails in a way that looks like
a bug in the other, so each is asserted separately.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-cfa-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-cfa-output-"))
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


class CloudflareAccessHeaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._id = handler.CF_ACCESS_CLIENT_ID
        self._secret = handler.CF_ACCESS_CLIENT_SECRET
        self._post = handler.requests.post
        self._put = handler.requests.put

    def tearDown(self) -> None:
        handler.CF_ACCESS_CLIENT_ID = self._id
        handler.CF_ACCESS_CLIENT_SECRET = self._secret
        handler.requests.post = self._post
        handler.requests.put = self._put

    def test_no_headers_when_unconfigured(self):
        """A Worker not behind Access must not receive stray headers."""
        handler.CF_ACCESS_CLIENT_ID = ""
        handler.CF_ACCESS_CLIENT_SECRET = ""
        self.assertEqual(handler.cloudflare_access_headers(), {})

    def test_no_headers_when_only_half_configured(self):
        """Half a service token is not a credential; sending it would just confuse Access."""
        handler.CF_ACCESS_CLIENT_ID = "id-only"
        handler.CF_ACCESS_CLIENT_SECRET = ""
        self.assertEqual(handler.cloudflare_access_headers(), {})

        handler.CF_ACCESS_CLIENT_ID = ""
        handler.CF_ACCESS_CLIENT_SECRET = "secret-only"
        self.assertEqual(handler.cloudflare_access_headers(), {})

    def test_headers_when_configured(self):
        handler.CF_ACCESS_CLIENT_ID = "client-id"
        handler.CF_ACCESS_CLIENT_SECRET = "client-secret"
        self.assertEqual(
            handler.cloudflare_access_headers(),
            {"CF-Access-Client-Id": "client-id", "CF-Access-Client-Secret": "client-secret"},
        )

    def test_progress_callbacks_carry_both_credentials(self):
        handler.CF_ACCESS_CLIENT_ID = "client-id"
        handler.CF_ACCESS_CLIENT_SECRET = "client-secret"
        captured = {}

        def _post(url, json=None, headers=None, timeout=None, allow_redirects=None):
            captured.update(headers or {})
            return types.SimpleNamespace(ok=True, status_code=200, headers={})

        handler.requests.post = _post
        handler.ProgressReporter("job-1", "https://w/p", "tok").phase(handler.PHASE_SAMPLING)

        self.assertEqual(captured.get("CF-Access-Client-Id"), "client-id")
        self.assertEqual(captured.get("CF-Access-Client-Secret"), "client-secret")
        self.assertEqual(captured.get("Authorization"), "Bearer tok")

    def test_output_upload_carries_both_credentials(self):
        handler.CF_ACCESS_CLIENT_ID = "client-id"
        handler.CF_ACCESS_CLIENT_SECRET = "client-secret"
        captured = {}

        tmp = tempfile.mkdtemp(prefix="h3-access-")
        video = os.path.join(tmp, "v.mp4")
        with open(video, "wb") as fh:
            fh.write(b"x" * 16)

        def _put(url, data=None, headers=None, timeout=None, allow_redirects=None):
            captured.update(headers or {})
            return types.SimpleNamespace(
                ok=True,
                status_code=200,
                text="",
                headers={},
                json=lambda: {"key": "outputs/j/video.mp4"},
            )

        handler.requests.put = _put
        handler.WorkerUploadStore("https://w/o", "tok", 30).store(video, {"filename": "v.mp4"})

        self.assertEqual(captured.get("CF-Access-Client-Id"), "client-id")
        self.assertEqual(captured.get("CF-Access-Client-Secret"), "client-secret")
        self.assertEqual(captured.get("Authorization"), "Bearer tok")
        self.assertEqual(captured.get("Content-Type"), "video/mp4")

    def test_access_failure_still_cannot_fail_a_generation(self):
        """Access rejecting a callback is a delivery problem, never a generation error."""
        handler.CF_ACCESS_CLIENT_ID = "client-id"
        handler.CF_ACCESS_CLIENT_SECRET = "wrong-secret"

        def _post(url, json=None, headers=None, timeout=None, allow_redirects=None):
            return types.SimpleNamespace(
                ok=True, status_code=302, headers={"Location": "https://x/login"}
            )

        handler.requests.post = _post
        reporter = handler.ProgressReporter("job-1", "https://w/p", "tok")
        reporter.phase(handler.PHASE_SAMPLING)  # must not raise
        self.assertEqual(reporter.failed, 1)


class UnresolvedSecretReferenceTest(unittest.TestCase):
    """RunPod passes {{ RUNPOD_SECRET_x }} through literally when no such secret exists.

    The variable then looks set, so the handler would happily send a template string as a
    credential and Cloudflare Access would reject it - indistinguishable, from inside the
    container, from a real token being refused. These pin down that it is named instead.
    """

    def setUp(self) -> None:
        self._id = handler.CF_ACCESS_CLIENT_ID
        self._secret = handler.CF_ACCESS_CLIENT_SECRET

    def tearDown(self) -> None:
        handler.CF_ACCESS_CLIENT_ID = self._id
        handler.CF_ACCESS_CLIENT_SECRET = self._secret

    def test_detects_the_placeholder(self):
        self.assertTrue(
            handler._is_unresolved_secret_reference("{{ RUNPOD_SECRET_CLOUDFLARE_ACCESS_KEY_ID }}")
        )
        self.assertTrue(handler._is_unresolved_secret_reference("  {{ RUNPOD_SECRET_X }}  "))

    def test_does_not_flag_a_real_token(self):
        self.assertFalse(handler._is_unresolved_secret_reference("abcdef1234567890.access"))
        self.assertFalse(handler._is_unresolved_secret_reference(""))
        self.assertFalse(handler._is_unresolved_secret_reference("{{ not closed"))

    def test_no_headers_are_sent_for_an_unresolved_reference(self):
        """Sending a template string as a credential is worse than sending nothing."""
        handler.CF_ACCESS_CLIENT_ID = "{{ RUNPOD_SECRET_CLOUDFLARE_ACCESS_KEY_ID }}"
        handler.CF_ACCESS_CLIENT_SECRET = "{{ RUNPOD_SECRET_CLOUDFLARE_SECRET_ACCESS_KEY }}"
        self.assertEqual(handler.cloudflare_access_headers(), {})

    def test_the_error_names_the_unresolved_reference(self):
        handler.CF_ACCESS_CLIENT_ID = "{{ RUNPOD_SECRET_CLOUDFLARE_ACCESS_KEY_ID }}"
        handler.CF_ACCESS_CLIENT_SECRET = "{{ RUNPOD_SECRET_CLOUDFLARE_SECRET_ACCESS_KEY }}"

        message = handler._describe_non_2xx(
            types.SimpleNamespace(
                status_code=302,
                headers={"Location": "https://x.cloudflareaccess.com/cdn-cgi/access/login/y"},
                text="",
            )
        )
        self.assertIn("UNRESOLVED RunPod secret reference", message)
        self.assertIn("Settings -> Secrets", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
