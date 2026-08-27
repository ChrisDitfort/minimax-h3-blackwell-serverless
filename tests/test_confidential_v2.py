"""Confidential Generation v2: hybrid encryption, and what the worker never learns.

Run with:  python -m pytest tests/test_confidential_v2.py

v1 put a symmetric key in the job payload, so a retained copy of that payload plus the
stored object was enough to decrypt the video. v2 removes that, and the tests here exist to
prove the removal rather than assert it.

The one that matters most is TestSecurityInvariant: it gives an attacker everything the
platform ever holds - the job input, the job output, the stored bytes, the metadata, the
public key, the wrapped file key - and shows that none of it decrypts anything.
"""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-v2-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-v2-output-"))
os.environ.setdefault("COMFY_TEMP_DIR", tempfile.mkdtemp(prefix="h3-v2-temp-"))
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

MP4_SIGNATURE = b"\x00\x00\x00\x18ftypmp42"


def sample_video(size: int = 4096) -> bytes:
    return MP4_SIGNATURE + b"MOOV" + bytes(i % 251 for i in range(size))


def write_output(name: str, data: bytes | None = None) -> str:
    path = os.path.join(handler.COMFY_OUTPUT_DIR, name)
    with open(path, "wb") as handle:
        handle.write(data if data is not None else sample_video())
    return path


def history_for(name: str) -> dict:
    return {"outputs": {"9": {"images": [{"filename": name, "subfolder": "", "type": "output"}]}}}


#: One RSA-3072 pair for the module. Keygen is ~200 ms; a per-test pair would dominate the
#: runtime. Reuse is a test-speed decision only - production draws a fresh file key per
#: artefact, and the key pair belongs to the user rather than the platform.
_KEYPAIR = None


def keypair():
    global _KEYPAIR
    if _KEYPAIR is None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        spki = private.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        _KEYPAIR = (private, spki, base64.urlsafe_b64encode(spki).decode().rstrip("="))
    return _KEYPAIR


def public_key():
    return artifacts.PublicEncryptionKey(keypair()[1])


def protector(**kwargs):
    return artifacts.ConfidentialV2Protector(public_key(), **kwargs)


class CapturingStore(handler.OutputStore):
    """Whatever arrives here is what persistent storage would have received."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def store(self, path, entry, protected=None):
        with open(path, "rb") as handle:
            body = handle.read()
        self.calls.append({"path": path, "entry": entry, "protected": protected, "body": body})
        if self.fail:
            raise handler.WorkflowError("simulated upload failure")
        return {"filename": entry["filename"], "size": len(body)}

    @property
    def body(self) -> bytes:
        return self.calls[0]["body"]


# ======================================================================================
# The security invariant
# ======================================================================================


class TestSecurityInvariant(unittest.TestCase):
    """No value crossing into the job payload may decrypt the artefact afterwards."""

    def test_platform_data_alone_cannot_decrypt_a_v2_artefact(self):
        private, spki, spki_b64 = keypair()
        plaintext = sample_video(8192)
        path = write_output("invariant.mp4", plaintext)

        store = CapturingStore()
        handler.collect_outputs(
            history_for("invariant.mp4"),
            store=store,
            protector=protector(),
            generation_id="gen-invariant",
        )
        stored = store.body

        # Everything an attacker could hold after the job finishes.
        parsed = artifacts.parse_container_prefix(stored)
        job_input = json.dumps(
            {
                "workflow": {"1": {"class_type": "X"}},
                "privacy": {"mode": "confidential"},
                "encryption": {
                    "version": 2,
                    "algorithm": "AES-256-GCM",
                    "keyWrapAlgorithm": "RSA-OAEP-256",
                    "publicKey": spki_b64,
                    "keyId": public_key().key_id,
                },
            }
        )
        job_output = json.dumps({"images": [{"filename": "v.mp4"}], "privacyMode": "confidential"})
        platform_data = {
            "job_input": job_input.encode(),
            "job_output": job_output.encode(),
            "stored_object": stored,
            "object_metadata": json.dumps(
                {"privacyMode": "confidential", "cryptoVersion": "2", "keyId": public_key().key_id}
            ).encode(),
            "public_key": spki,
            "wrapped_file_key": parsed["key_wrap"]["wrappedFileKey"].encode(),
            "container_header": parsed["header_bytes"],
        }

        # 1. None of it contains the private key or the file key.
        private_pkcs8 = self._private_bytes(private)
        for name, value in platform_data.items():
            with self.subTest(f"{name} holds no private key"):
                self.assertNotIn(private_pkcs8, value)

        # 2. None of it decrypts the artefact, whether used raw or hashed to 32 bytes.
        import hashlib

        for name, value in platform_data.items():
            for label, candidate in (
                ("raw", value[:32] if len(value) >= 32 else None),
                ("sha256", hashlib.sha256(value).digest()),
            ):
                if candidate is None:
                    continue
                with self.subTest(f"{name} ({label}) must not decrypt"):
                    with self.assertRaises(artifacts.ContainerError):
                        artifacts.decrypt_container(stored, candidate)

        # 3. The public key cannot decrypt. It has no private exponent to try.
        self.assertFalse(hasattr(public_key(), "unwrap"))

        # 4. The private key, which never crossed the boundary, does.
        recovered, header = artifacts.decrypt_v2_container(stored, private)
        self.assertEqual(recovered, plaintext)
        self.assertEqual(header["artifactId"], "gen-invariant")
        self.assertFalse(os.path.exists(path))

    @staticmethod
    def _private_bytes(private) -> bytes:
        from cryptography.hazmat.primitives import serialization

        return private.private_bytes(
            serialization.Encoding.DER,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )


# ======================================================================================
# Hybrid encryption
# ======================================================================================


class TestHybridEncryption(unittest.TestCase):
    def setUp(self):
        self.plaintext = sample_video()
        self.private = keypair()[0]

    def seal(self, generation_id="gen"):
        path = write_output(f"{secrets.token_hex(6)}.mp4", self.plaintext)
        protected = protector().protect(
            artifacts.GeneratedArtifact(
                path=path, mime_type="video/mp4", generation_id=generation_id
            )
        )
        self.addCleanup(protected.discard)
        with open(protected.path, "rb") as handle:
            return handle.read(), protected

    def test_the_public_key_wraps_and_the_private_key_unwraps(self):
        file_key = artifacts.EphemeralKey(secrets.token_bytes(32))
        expected = file_key.bytes
        wrapped = public_key().wrap(file_key)

        self.assertNotIn(expected, wrapped)
        self.assertEqual(len(wrapped), 384, "a 3072-bit RSA wrap is 384 bytes")

        recovered = artifacts.unwrap_file_key(wrapped, self.private)
        self.assertEqual(recovered.bytes, expected)

    def test_a_full_round_trip(self):
        container, _ = self.seal()
        recovered, header = artifacts.decrypt_v2_container(container, self.private)
        self.assertEqual(recovered, self.plaintext)
        self.assertEqual(header["v"], 2)
        self.assertEqual(header["kw"]["alg"], "RSA-OAEP-256")

    def test_ciphertext_differs_from_plaintext(self):
        container, _ = self.seal()
        self.assertNotIn(self.plaintext, container)
        self.assertFalse(container.startswith(MP4_SIGNATURE))

    def test_a_different_private_key_cannot_unwrap(self):
        from cryptography.hazmat.primitives.asymmetric import rsa

        container, _ = self.seal()
        stranger = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_v2_container(container, stranger)

    def test_modified_ciphertext_fails_authentication(self):
        container, _ = self.seal()
        tampered = bytearray(container)
        tampered[-40] ^= 0x01
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_v2_container(bytes(tampered), self.private)

    def test_a_modified_tag_fails(self):
        container, _ = self.seal()
        tampered = bytearray(container)
        tampered[-1] ^= 0x80
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_v2_container(bytes(tampered), self.private)

    def test_a_modified_wrapped_key_fails_safely(self):
        """OAEP is authenticated padding: a flipped bit is a decode failure, not a new key."""
        container, _ = self.seal()
        parsed = artifacts.parse_container_prefix(container)
        at = container.index(parsed["key_wrap"]["wrappedFileKey"].encode()) + 8

        tampered = bytearray(container)
        tampered[at] = ord("A") if tampered[at] != ord("A") else ord("B")

        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_v2_container(bytes(tampered), self.private)

    def test_swapping_in_a_key_wrapped_to_another_public_key_breaks_the_video(self):
        """The reason the wrapped key lives inside the authenticated header.

        An attacker who substitutes a file key wrapped to a key pair they control cannot
        then read the video: the header is the AEAD's associated data, so the swap
        invalidates the tag over the ciphertext.
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        container, _ = self.seal()
        parsed = artifacts.parse_container_prefix(container)

        attacker = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        attacker_public = artifacts.PublicEncryptionKey(
            attacker.public_key().public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
        their_key = artifacts.EphemeralKey(secrets.token_bytes(32))
        their_wrap = base64.urlsafe_b64encode(attacker_public.wrap(their_key)).decode().rstrip("=")

        # Same length, so the framing stays valid and only authenticated bytes change.
        self.assertEqual(len(their_wrap), len(parsed["key_wrap"]["wrappedFileKey"]))
        forged = container.replace(
            parsed["key_wrap"]["wrappedFileKey"].encode(), their_wrap.encode()
        )

        # The attacker can unwrap their own key from the forged container...
        reparsed = artifacts.parse_container_prefix(forged)
        self.assertEqual(reparsed["key_wrap"]["wrappedFileKey"], their_wrap)
        # ...and it does not open the video.
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_v2_container(forged, attacker)

    def test_truncation_fails(self):
        container, _ = self.seal()
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_v2_container(container[:-8], self.private)


# ======================================================================================
# The per-video file key
# ======================================================================================


class TestFileKeyLifecycle(unittest.TestCase):
    def test_every_artefact_gets_a_fresh_file_key_and_nonce(self):
        """Retries included. A deterministic object path must not imply a deterministic key.

        Nonce reuse under one key is the single fatal misuse of GCM, and the same generation
        being encrypted twice - a RunPod retry - is exactly when a careless implementation
        would reuse one.
        """
        private = keypair()[0]
        plaintext = sample_video(1024)

        wrapped_keys, nonces, file_keys, ciphertexts = set(), set(), set(), set()
        for _ in range(16):
            path = write_output("retry.mp4", plaintext)
            protected = protector().protect(
                artifacts.GeneratedArtifact(
                    # The same generation id every time, as a retry would have.
                    path=path, mime_type="video/mp4", generation_id="same-generation"
                )
            )
            with open(protected.path, "rb") as handle:
                container = handle.read()
            protected.discard()

            parsed = artifacts.parse_container_prefix(container)
            wrapped_keys.add(parsed["key_wrap"]["wrappedFileKey"])
            nonces.add(container[parsed["nonce_offset"] : parsed["ciphertext_offset"]])
            ciphertexts.add(container[parsed["ciphertext_offset"] :])
            padded = parsed["key_wrap"]["wrappedFileKey"] + "=="
            file_keys.add(
                artifacts.unwrap_file_key(base64.urlsafe_b64decode(padded), private).bytes
            )

        self.assertEqual(len(file_keys), 16, "a file key was reused across artefacts")
        self.assertEqual(len(nonces), 16, "a nonce was reused")
        self.assertEqual(len(wrapped_keys), 16)
        self.assertEqual(len(ciphertexts), 16, "identical ciphertext means identical key material")

    def test_the_plaintext_file_key_is_never_in_the_container(self):
        private = keypair()[0]
        path = write_output("fek.mp4")
        protected = protector().protect(
            artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
        )
        try:
            with open(protected.path, "rb") as handle:
                container = handle.read()
            parsed = artifacts.parse_container_prefix(container)
            padded = parsed["key_wrap"]["wrappedFileKey"] + "=="
            file_key = artifacts.unwrap_file_key(base64.urlsafe_b64decode(padded), private).bytes

            self.assertNotIn(file_key, container, "the file key must appear only wrapped")
            self.assertNotIn(file_key.hex().encode(), container)
            self.assertNotIn(base64.b64encode(file_key), container)
        finally:
            protected.discard()

    def test_the_plaintext_file_key_is_never_in_the_returned_metadata(self):
        private = keypair()[0]
        path = write_output("fek-meta.mp4")
        protected = protector().protect(
            artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
        )
        try:
            with open(protected.path, "rb") as handle:
                parsed = artifacts.parse_container_prefix(handle.read(8192))
            padded = parsed["key_wrap"]["wrappedFileKey"] + "=="
            file_key = artifacts.unwrap_file_key(base64.urlsafe_b64decode(padded), private).bytes

            serialized = json.dumps(protected.metadata)
            self.assertNotIn(file_key.hex(), serialized)
            self.assertNotIn(base64.b64encode(file_key).decode(), serialized)
            # What is there is public: an algorithm name and the key's derived id.
            self.assertEqual(protected.metadata["keyWrapAlgorithm"], "RSA-OAEP-256")
            self.assertEqual(protected.metadata["cryptoVersion"], 2)
        finally:
            protected.discard()

    def test_the_protector_holds_no_decryption_capable_secret(self):
        """destroy() is a no-op here precisely because there is nothing to destroy."""
        instance = protector()
        instance.destroy()
        self.assertFalse(hasattr(instance, "_key"))


# ======================================================================================
# Failing closed
# ======================================================================================


class TestFailureBehaviour(unittest.TestCase):
    def test_a_wrapping_failure_uploads_nothing_and_keeps_no_ciphertext(self):
        class BrokenKey(artifacts.PublicEncryptionKey):
            def wrap(self, file_key):
                raise RuntimeError("HSM unavailable")

        broken = BrokenKey(keypair()[1])
        path = write_output("wrapfail.mp4")
        store = CapturingStore()

        with self.assertRaisesRegex(handler.WorkflowError, "Wrapping the file key"):
            handler.collect_outputs(
                history_for("wrapfail.mp4"),
                store=store,
                protector=artifacts.ConfidentialV2Protector(broken),
            )

        self.assertEqual(store.calls, [], "nothing may be uploaded when wrapping fails")
        self.assertTrue(
            os.path.exists(path),
            "wrapping happens before encryption, so the plaintext is still the caller's",
        )
        os.remove(path)

    def test_an_upload_failure_deletes_plaintext_and_ciphertext(self):
        path = write_output("uploadfail.mp4")
        store = CapturingStore(fail=True)

        with self.assertRaises(handler.WorkflowError):
            handler.collect_outputs(
                history_for("uploadfail.mp4"), store=store, protector=protector()
            )

        self.assertFalse(os.path.exists(path), "plaintext must be gone even when upload fails")
        self.assertFalse(os.path.exists(store.calls[0]["path"]), "ciphertext must be gone too")

    def test_keep_outputs_cannot_preserve_confidential_plaintext(self):
        path = write_output("keep-v2.mp4")
        previous = os.environ.get("H3_KEEP_OUTPUTS")
        os.environ["H3_KEEP_OUTPUTS"] = "1"
        try:
            handler.collect_outputs(
                history_for("keep-v2.mp4"), store=CapturingStore(), protector=protector()
            )
            self.assertFalse(os.path.exists(path))
        finally:
            if previous is None:
                os.environ.pop("H3_KEEP_OUTPUTS", None)
            else:
                os.environ["H3_KEEP_OUTPUTS"] = previous

    def test_plaintext_is_gone_after_a_successful_encryption(self):
        path = write_output("success.mp4")
        store = CapturingStore()
        handler.collect_outputs(history_for("success.mp4"), store=store, protector=protector())
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(store.calls[0]["path"]))


# ======================================================================================
# Public keys
# ======================================================================================


class TestPublicKeys(unittest.TestCase):
    def test_the_key_id_is_the_truncated_sha256_of_the_spki(self):
        import hashlib

        spki = keypair()[1]
        expected = base64.urlsafe_b64encode(hashlib.sha256(spki).digest()[:16]).decode().rstrip("=")
        self.assertEqual(artifacts.public_key_id(spki), expected)
        self.assertEqual(public_key().key_id, expected)

    def test_decoding_accepts_both_base64_alphabets(self):
        spki = keypair()[1]
        for encoded in (
            base64.urlsafe_b64encode(spki).decode().rstrip("="),
            base64.b64encode(spki).decode(),
        ):
            with self.subTest(encoded[:12]):
                self.assertEqual(artifacts.decode_public_key(encoded).spki, spki)

    def test_a_missing_key_is_rejected(self):
        for value in ("", "   ", None):
            with self.subTest(repr(value)):
                with self.assertRaisesRegex(artifacts.ArtifactError, "missing"):
                    artifacts.decode_public_key(value)

    def test_the_repr_does_not_leak_the_encoding(self):
        rendered = repr(public_key())
        self.assertIn("RSA-OAEP-256", rendered)
        self.assertIn("3072-bit", rendered)
        self.assertNotIn(keypair()[2][:40], rendered)


# ======================================================================================
# Reading v1 still works
# ======================================================================================


class TestV1Compatibility(unittest.TestCase):
    """Existing artefacts must not become unreadable because a new version shipped."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vectors", "confidential_container.json"
        )
        with open(path, encoding="utf-8") as handle:
            cls.vector = json.load(handle)

    def test_a_v1_container_still_parses(self):
        parsed = artifacts.parse_container_prefix(bytes.fromhex(self.vector["container_hex"]))
        self.assertEqual(parsed["version"], artifacts.CONTAINER_V1_SYMMETRIC)
        self.assertIsNone(parsed["key_wrap"], "v1 wraps nothing")

    def test_a_v1_container_still_decrypts(self):
        plaintext, header = artifacts.decrypt_container(
            bytes.fromhex(self.vector["container_hex"]), bytes.fromhex(self.vector["key_hex"])
        )
        self.assertEqual(plaintext.hex(), self.vector["plaintext_hex"])
        self.assertEqual(header["v"], 1)

    def test_the_v1_reader_refuses_a_v2_container(self):
        """Different key material entirely; a clear message beats a confusing failure."""
        path = write_output("v1-reader.mp4")
        protected = protector().protect(
            artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
        )
        try:
            with open(protected.path, "rb") as handle:
                container = handle.read()
            with self.assertRaisesRegex(artifacts.ContainerError, "Not a hybrid container|v1"):
                artifacts.decrypt_v2_container(bytes.fromhex(self.vector["container_hex"]), keypair()[0])
            # And the v2 container is not openable with a raw symmetric key.
            with self.assertRaises(artifacts.ContainerError):
                artifacts.decrypt_container(container, secrets.token_bytes(32))
        finally:
            protected.discard()

    def test_only_v2_is_writable(self):
        self.assertEqual(artifacts.FORMAT_VERSION, artifacts.CONTAINER_V2_HYBRID)
        self.assertIn(artifacts.CONTAINER_V1_SYMMETRIC, artifacts.SUPPORTED_CONTAINER_VERSIONS)

    def test_writing_an_unknown_version_is_refused(self):
        with self.assertRaisesRegex(artifacts.ContainerError, "unknown container version"):
            artifacts.encrypt_stream(
                io.BytesIO(b"x"),
                io.BytesIO(),
                key=artifacts.EphemeralKey(secrets.token_bytes(32)),
                header={"v": 9},
            )


if __name__ == "__main__":
    unittest.main()


# ======================================================================================
# The RunPod job output
# ======================================================================================


class TestJobOutput(unittest.TestCase):
    """What the worker hands back to RunPod, driven through the real handler().

    RunPod retains job results as well as job inputs, so the output JSON is a second place
    a decryption-capable secret could leak into permanent storage. It must not.
    """

    def run_confidential_job(self, plaintext: bytes):
        """One full pass through handler(), with ComfyUI and the upload endpoint stubbed."""
        private, spki, spki_b64 = keypair()
        name = f"output-{secrets.token_hex(4)}.mp4"
        write_output(name, plaintext)

        uploaded = {}

        class FakeResponse:
            status_code = 201
            text = '{"key":"outputs/gen-out/artifact.enc"}'

            @staticmethod
            def json():
                return {"key": "outputs/gen-out/artifact.enc", "url": "/jobs/gen-out/artifact"}

        def fake_put(url, data=None, headers=None, **kwargs):
            uploaded["body"] = data.read()
            uploaded["headers"] = dict(headers or {})
            return FakeResponse()

        originals = (
            handler.start_comfyui,
            handler.queue_prompt,
            handler.await_execution,
            handler.requests.put,
        )
        handler.start_comfyui = lambda: None
        handler.queue_prompt = lambda workflow, client_id: "prompt-1"
        handler.await_execution = lambda *a, **k: history_for(name)
        handler.requests.put = fake_put
        try:
            result = handler.handler(
                {
                    "id": "runpod-job-1",
                    "input": {
                        "workflow": {"1": {"class_type": "SaveVideo", "inputs": {}}},
                        "privacy": {"mode": "confidential"},
                        "encryption": {
                            "version": 2,
                            "algorithm": "AES-256-GCM",
                            "keyWrapAlgorithm": "RSA-OAEP-256",
                            "publicKey": spki_b64,
                        },
                        "output": {"url": "https://worker.example/internal/jobs/gen-out/output",
                                   "token": "job-token", "jobId": "gen-out"},
                    },
                }
            )
        finally:
            (
                handler.start_comfyui,
                handler.queue_prompt,
                handler.await_execution,
                handler.requests.put,
            ) = originals

        return result, uploaded, private

    def test_the_job_output_carries_no_decryption_capable_secret(self):
        plaintext = sample_video(2048)
        result, uploaded, private = self.run_confidential_job(plaintext)

        self.assertNotIn("error", result, result.get("error"))
        serialized = json.dumps(result)

        # What the artefact was actually encrypted with, recovered from the upload.
        parsed = artifacts.parse_container_prefix(uploaded["body"])
        padded = parsed["key_wrap"]["wrappedFileKey"] + "=="
        file_key = artifacts.unwrap_file_key(base64.urlsafe_b64decode(padded), private).bytes

        self.assertNotIn(file_key.hex(), serialized, "the file key must not be in the output")
        self.assertNotIn(base64.b64encode(file_key).decode(), serialized)
        self.assertNotIn(
            base64.urlsafe_b64encode(file_key).decode().rstrip("="), serialized
        )
        for forbidden in ("privateKey", "passphrase", "kek", "fileEncryptionKey", "aesKey"):
            self.assertNotIn(forbidden, serialized)

        # No base64 of the plaintext either - the inline delivery path must not have run.
        self.assertNotIn("data", result["images"][0])
        self.assertNotIn(base64.b64encode(plaintext).decode()[:64], serialized)

    def test_the_job_output_describes_the_artefact_in_public_terms(self):
        result, _, _ = self.run_confidential_job(sample_video(2048))

        self.assertEqual(result["privacyMode"], "confidential")
        artifact = result["images"][0]["artifact"]
        self.assertEqual(artifact["cryptoVersion"], 2)
        self.assertEqual(artifact["algorithm"], "AES-256-GCM")
        self.assertEqual(artifact["keyWrapAlgorithm"], "RSA-OAEP-256")
        self.assertEqual(artifact["keyId"], public_key().key_id)
        self.assertEqual(artifact["contentType"], "application/octet-stream")
        self.assertEqual(artifact["originalContentType"], "video/mp4")
        self.assertEqual(result["images"][0]["key"], "outputs/gen-out/artifact.enc")

    def test_what_is_uploaded_is_a_v2_container_for_this_generation(self):
        plaintext = sample_video(2048)
        _, uploaded, private = self.run_confidential_job(plaintext)

        self.assertEqual(uploaded["headers"]["Content-Type"], "application/octet-stream")
        self.assertNotIn(plaintext, uploaded["body"])

        recovered, header = artifacts.decrypt_v2_container(uploaded["body"], private)
        self.assertEqual(recovered, plaintext)
        # The generation id comes from the output block, which is what the storage key is
        # derived from - so the Worker's header check will agree.
        self.assertEqual(header["artifactId"], "gen-out")

    def test_the_plaintext_is_gone_when_the_job_returns(self):
        result, _, _ = self.run_confidential_job(sample_video(2048))
        name = result["images"][0]["filename"]
        self.assertFalse(os.path.exists(os.path.join(handler.COMFY_OUTPUT_DIR, name)))
