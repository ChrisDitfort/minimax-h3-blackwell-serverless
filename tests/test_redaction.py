"""Regression tests for secret redaction and safe logging.

Run with:  python -m pytest tests/test_redaction.py

Redaction is the control that has to work when everything else already went wrong: an
unexpected exception, a debug dump, an upstream error echoed back. So these tests are
written as "this exact string must not appear in the output", not as "the function was
called".

The second half is just as important and easier to get wrong: redaction that eats useful
fields gets switched off. `keyId` is meant to be visible, and every `key` in this codebase
that is not inside a crypto block is an R2 object key that a support question needs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-redact-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-redact-output-"))
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

import artifacts  # noqa: E402
import handler  # noqa: E402

KEY = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")

# A real public key, so the confidential path can be exercised as far as the log line.
def _public_key_b64():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    spki = private.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return base64.urlsafe_b64encode(spki).decode().rstrip("=")


PUBLIC_KEY = _public_key_b64()
PASSPHRASE = "correct horse battery staple"
ACCESS_SECRET = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def rendered(value) -> str:
    return json.dumps(artifacts.redact(value), default=str)


class TestSecretsAreRemoved(unittest.TestCase):
    def test_an_encryption_key_is_redacted(self):
        payload = {"encryption": {"algorithm": "AES-256-GCM", "key": KEY, "keyId": "kid-7"}}
        out = rendered(payload)
        self.assertNotIn(KEY, out)
        self.assertIn("[redacted]", out)

    def test_a_passphrase_is_redacted_wherever_it_appears(self):
        for shape in (
            {"passphrase": PASSPHRASE},
            {"user": {"password": PASSPHRASE}},
            {"a": {"b": {"c": {"passphrase": PASSPHRASE}}}},
            {"list": [{"passphrase": PASSPHRASE}]},
        ):
            with self.subTest(shape=list(shape)[0]):
                self.assertNotIn(PASSPHRASE, rendered(shape))

    def test_authorization_headers_are_redacted(self):
        headers = {
            "Authorization": "Bearer eyJhbGciOi.super.secret",
            "Content-Type": "video/mp4",
        }
        out = rendered(headers)
        self.assertNotIn("super.secret", out)
        self.assertIn("video/mp4", out, "ordinary headers must survive")

    def test_cloudflare_access_credentials_are_redacted(self):
        headers = {
            "CF-Access-Client-Id": "abc123.access",
            "CF-Access-Client-Secret": ACCESS_SECRET,
        }
        out = rendered(headers)
        self.assertNotIn(ACCESS_SECRET, out)
        self.assertNotIn("abc123.access", out, "the client id identifies the service token")

    def test_wrapped_keys_and_data_keys_are_redacted(self):
        payload = {"encryption": {"wrapped_key": "AAAA", "dek": "BBBB", "key_id": "kid"}}
        out = rendered(payload)
        self.assertNotIn('"AAAA"', out)
        self.assertNotIn('"BBBB"', out)
        self.assertIn("kid", out)

    def test_an_ephemeral_key_object_cannot_be_serialized_into_a_log(self):
        raw = secrets.token_bytes(32)
        payload = {"protector": {"key": artifacts.EphemeralKey(raw)}}
        out = rendered(payload)
        self.assertNotIn(raw.hex(), out)
        self.assertNotIn(base64.b64encode(raw).decode(), out)

    def test_deep_nesting_is_bounded_rather_than_recursing_forever(self):
        deep = current = {}
        for _ in range(40):
            current["next"] = {}
            current = current["next"]
        current["passphrase"] = PASSPHRASE
        out = rendered(deep)
        self.assertNotIn(PASSPHRASE, out)
        self.assertIn("nesting too deep", out)


class TestUsefulFieldsSurvive(unittest.TestCase):
    """Redaction that destroys diagnostics is redaction that gets turned off."""

    def test_key_id_is_never_redacted(self):
        for field in ("keyId", "key_id"):
            with self.subTest(field):
                out = rendered({"encryption": {field: "customer-key-3", "key": KEY}})
                self.assertIn("customer-key-3", out)
                self.assertNotIn(KEY, out)

    def test_an_r2_object_key_is_not_a_secret(self):
        payload = {"video": {"key": "outputs/job-1/artifact.enc", "size": 1024}}
        out = rendered(payload)
        self.assertIn("outputs/job-1/artifact.enc", out)

    def test_an_explicit_r2_key_field_survives(self):
        out = rendered({"r2_key": "outputs/job-1/video.mp4", "storage_key": "x/y"})
        self.assertIn("outputs/job-1/video.mp4", out)
        self.assertIn("x/y", out)

    def test_structure_is_preserved(self):
        payload = {"encryption": {"algorithm": "AES-256-GCM", "key": KEY}, "width": 1024}
        out = artifacts.redact(payload)
        self.assertEqual(out["width"], 1024)
        self.assertEqual(out["encryption"]["algorithm"], "AES-256-GCM")
        self.assertEqual(out["encryption"]["key"], "[redacted]")

    def test_redaction_does_not_mutate_its_input(self):
        payload = {"encryption": {"key": KEY}}
        artifacts.redact(payload)
        self.assertEqual(payload["encryption"]["key"], KEY)


class TestBothImplementationsAgree(unittest.TestCase):
    """The Worker and the handler must redact the same names.

    A field scrubbed in Python and printed in JavaScript is not partially protected; it is
    unprotected, in the half of the system nobody was looking at. This test exists because
    the two lists had already drifted once.
    """

    @staticmethod
    def worker_set(name: str) -> set[str]:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "worker.js"), encoding="utf-8") as handle:
            source = handle.read()

        start = source.index(f"const {name} = new Set([")
        body = source[source.index("[", start) + 1 : source.index("]);", start)]
        # Strip line comments first, then take every string literal. Handles both the
        # one-per-line and the all-on-one-line declaration styles.
        body = re.sub(r"//[^\n]*", "", body)
        return set(re.findall(r'"([^"]+)"', body))

    def test_secret_field_lists_are_identical(self):
        self.assertEqual(self.worker_set("SECRET_FIELDS"), set(artifacts.SECRET_FIELDS))

    def test_never_redact_lists_are_identical(self):
        self.assertEqual(self.worker_set("NEVER_REDACT"), set(artifacts.NEVER_REDACT))

    def test_crypto_parent_lists_are_identical(self):
        self.assertEqual(self.worker_set("CRYPTO_PARENTS"), set(artifacts.CRYPTO_PARENTS))


class TestPromptHandling(unittest.TestCase):
    def test_the_digest_is_stable_and_short(self):
        a = artifacts.prompt_digest("a cinematic ocean scene")
        b = artifacts.prompt_digest("a cinematic ocean scene")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)
        self.assertNotEqual(a, artifacts.prompt_digest("a cinematic desert scene"))

    def test_the_digest_does_not_contain_the_prompt(self):
        prompt = "a very specific and identifying prompt"
        self.assertNotIn("specific", artifacts.prompt_digest(prompt))


class TestHandlerLogging(unittest.TestCase):
    """What the handler actually prints for a confidential job."""

    def setUp(self):
        self.lines = []
        self._log = handler.log
        handler.log = self.lines.append

    def tearDown(self):
        handler.log = self._log

    def test_the_generation_line_names_the_mode_but_not_the_key(self):
        result = handler.handler(
            {
                "id": "job-log-1",
                "input": {
                    "workflow": {"1": {"class_type": "X", "inputs": {}}},
                    "privacy": {"mode": "confidential"},
                    "encryption": {"version": 2, "publicKey": PUBLIC_KEY},
                },
            }
        )
        # It fails, deliberately - there is no upload endpoint - but the log line is emitted
        # before that check, which is exactly the window a key would leak in.
        self.assertIn("error", result)

        joined = "\n".join(self.lines)
        self.assertIn("privacy_mode=confidential", joined)
        self.assertIn("encryption=AES-256-GCM", joined)
        self.assertNotIn(KEY, joined)
        self.assertNotIn(PASSPHRASE, joined)

    def test_no_log_line_contains_key_material_on_a_validation_failure(self):
        handler.handler(
            {
                "id": "job-log-2",
                "input": {
                    "workflow": {"1": {"class_type": "X", "inputs": {}}},
                    "privacy": {"mode": "confidential"},
                    "encryption": {"key": KEY[:20]},
                },
            }
        )
        joined = "\n".join(self.lines)
        self.assertNotIn(KEY[:20], joined, "even a rejected key must not be echoed")

    def test_an_error_message_describes_shape_not_content(self):
        with self.assertRaises(artifacts.ArtifactError) as caught:
            artifacts.decode_key(base64.urlsafe_b64encode(b"x" * 16).decode())
        message = str(caught.exception)
        self.assertIn("16", message)
        self.assertNotIn("eHh4", message, "the rejected material must not be quoted back")


if __name__ == "__main__":
    unittest.main()
