"""Regression tests: a job must never claim a video it does not have.

Run with:  python -m unittest discover -s tests -v

These exist because that exact failure reached production. Cloudflare Access answers an
unauthenticated call with a 302 to its login page; `requests` follows redirects by default
and `Response.ok` is merely `status_code < 400`, so both the redirect and the resulting
HTML login page read as success. A job reported COMPLETED with a video URL, `output_upload
=0.1s` for 1.58 MB, and `progress_callbacks=26/26` - while the R2 bucket stayed empty.

Two independent guards now have to fail before that can happen again: the status code must
be a real 2xx, and the body must be the Worker's own acknowledgement.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-truth-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-truth-output-"))
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

ACCESS_LOGIN = "https://chrisditfort.cloudflareaccess.com/cdn-cgi/access/login/worker.dev"


def response(status, *, text="", json_body=None, headers=None, raise_json=False):
    def _json():
        if raise_json or json_body is None:
            raise ValueError("not json")
        return json_body

    return types.SimpleNamespace(
        status_code=status,
        ok=status < 400,  # what requests actually does - the trap being guarded against
        text=text,
        headers=headers or {},
        json=_json,
    )


class StatusCodeSemanticsTest(unittest.TestCase):
    def test_a_302_is_not_success(self):
        """requests.Response.ok would say True here. It must not be trusted."""
        redirect = response(302, headers={"Location": ACCESS_LOGIN})
        self.assertTrue(redirect.ok, "precondition: requests would call this ok")
        self.assertFalse(handler._is_2xx(redirect), "our check must reject it")

    def test_2xx_range(self):
        for status in (200, 201, 204, 299):
            self.assertTrue(handler._is_2xx(response(status)), status)
        for status in (301, 302, 303, 307, 308, 400, 401, 403, 500):
            self.assertFalse(handler._is_2xx(response(status)), status)

    def test_access_redirect_is_named_in_the_error(self):
        message = handler._describe_non_2xx(response(302, headers={"Location": ACCESS_LOGIN}))
        self.assertIn("Cloudflare Access", message)
        self.assertIn(".access", message, "the error should say what a valid client id looks like")

    def test_other_redirects_are_described_without_blaming_access(self):
        message = handler._describe_non_2xx(
            response(307, headers={"Location": "https://elsewhere.example/x"})
        )
        self.assertNotIn("Cloudflare Access", message)
        self.assertIn("elsewhere.example", message)


class UploadMustNotLieTest(unittest.TestCase):
    def setUp(self) -> None:
        self._put = handler.requests.put
        tmp = tempfile.mkdtemp(prefix="h3-truth-")
        self.video = os.path.join(tmp, "v.mp4")
        with open(self.video, "wb") as fh:
            fh.write(b"\x00" * 4096)

    def tearDown(self) -> None:
        handler.requests.put = self._put

    def put_returning(self, resp):
        captured = {}

        def _put(url, data=None, headers=None, timeout=None, allow_redirects=None):
            captured["allow_redirects"] = allow_redirects
            return resp

        handler.requests.put = _put
        return captured

    def store(self):
        return handler.WorkerUploadStore("https://w/internal/jobs/j/output", "tok", 30)

    def test_access_redirect_fails_the_job(self):
        """The exact production failure: 302 to the Access login page."""
        self.put_returning(response(302, headers={"Location": ACCESS_LOGIN}))
        store = self.store()
        with self.assertRaises(handler.WorkflowError) as ctx:
            store.store(self.video, {"filename": "v.mp4"})
        self.assertIn("Cloudflare Access", str(ctx.exception))
        self.assertEqual(store.uploaded_bytes, 0, "nothing was uploaded, so count nothing")

    def test_redirects_are_not_followed(self):
        captured = self.put_returning(response(302, headers={"Location": ACCESS_LOGIN}))
        with self.assertRaises(handler.WorkflowError):
            self.store().store(self.video, {"filename": "v.mp4"})
        self.assertIs(
            captured["allow_redirects"],
            False,
            "following a redirect re-issues the PUT as a GET and drops the video body",
        )

    def test_a_200_html_page_fails_the_job(self):
        """A login page answering 200 must not count as a stored video."""
        self.put_returning(response(200, text="<html>Sign in</html>", raise_json=True))
        store = self.store()
        with self.assertRaises(handler.WorkflowError) as ctx:
            store.store(self.video, {"filename": "v.mp4"})
        self.assertIn("did not reach R2", str(ctx.exception))
        self.assertEqual(store.uploaded_bytes, 0)

    def test_a_200_json_without_a_key_fails_the_job(self):
        self.put_returning(response(200, json_body={"ok": True}))
        with self.assertRaises(handler.WorkflowError) as ctx:
            self.store().store(self.video, {"filename": "v.mp4"})
        self.assertIn("acknowledgement", str(ctx.exception))

    def test_the_workers_real_acknowledgement_succeeds(self):
        self.put_returning(
            response(201, json_body={"key": "outputs/j/video.mp4", "url": "/jobs/j/video"})
        )
        store = self.store()
        result = store.store(self.video, {"filename": "v.mp4"})

        self.assertEqual(result["key"], "outputs/j/video.mp4")
        self.assertEqual(result["url"], "/jobs/j/video")
        self.assertEqual(result["size"], 4096)
        self.assertNotIn("data", result, "the R2 path must never return base64")
        self.assertEqual(store.uploaded_bytes, 4096)


class ProgressMustNotOvercountTest(unittest.TestCase):
    def setUp(self) -> None:
        self._post = handler.requests.post

    def tearDown(self) -> None:
        handler.requests.post = self._post

    def post_returning(self, resp):
        captured = {}

        def _post(url, json=None, headers=None, timeout=None, allow_redirects=None):
            captured["allow_redirects"] = allow_redirects
            return resp

        handler.requests.post = _post
        return captured

    def test_an_access_redirect_counts_as_failed_not_sent(self):
        """progress_callbacks=26/26 while nothing arrived is what made this invisible."""
        self.post_returning(response(302, headers={"Location": ACCESS_LOGIN}))
        reporter = handler.ProgressReporter("job-1", "https://w/p", "tok")
        reporter.phase(handler.PHASE_SAMPLING)

        self.assertEqual(reporter.sent, 0, "a redirect is not a delivery")
        self.assertEqual(reporter.failed, 1)

    def test_redirects_are_not_followed(self):
        captured = self.post_returning(response(302, headers={"Location": ACCESS_LOGIN}))
        handler.ProgressReporter("job-1", "https://w/p").phase(handler.PHASE_SAMPLING)
        self.assertIs(captured["allow_redirects"], False)

    def test_the_rejection_is_logged_once_not_per_event(self):
        self.post_returning(response(302, headers={"Location": ACCESS_LOGIN}))
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        for step in range(1, 21):
            reporter.step(step, 20)
        reporter.phase(handler.PHASE_DECODING)

        self.assertEqual(reporter.sent, 0)
        self.assertTrue(reporter._warned, "one clear warning, not twenty identical ones")

    def test_a_real_2xx_still_counts_as_sent(self):
        self.post_returning(response(200))
        reporter = handler.ProgressReporter("job-1", "https://w/p")
        reporter.phase(handler.PHASE_SAMPLING)
        self.assertEqual(reporter.sent, 1)
        self.assertEqual(reporter.failed, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
