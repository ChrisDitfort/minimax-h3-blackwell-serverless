"""Confidential Generation: container format, encryption, protection and cleanup.

Run with:  python -m pytest tests/test_confidential_artifacts.py

The canonical test is TestConfidentialPipeline. It runs a known plaintext video through the
whole artefact path, captures the bytes that would have been handed to persistent storage,
and proves three things about them:

    stored_bytes != plaintext_bytes
    decrypt(stored_bytes, client_key) == plaintext_bytes
    decrypt(stored_bytes, other_key) fails

Everything else in this file exists to make sure that stays true under failure: a wrong
key, a modified byte, an encryption that raises, an upload that raises, a retry.
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

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-conf-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-conf-output-"))
os.environ.setdefault("COMFY_TEMP_DIR", tempfile.mkdtemp(prefix="h3-conf-temp-"))
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

VECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors",
                           "confidential_container.json")

# The first bytes of a real MP4: an ISO base media file starts with a 'ftyp' box. Used to
# prove that what reached storage is not one.
MP4_SIGNATURE = b"\x00\x00\x00\x18ftypmp42"


def sample_video(size: int = 4096) -> bytes:
    """A plaintext payload that is recognisably an MP4 and recognisably not random."""
    return MP4_SIGNATURE + b"MOOV" + bytes(i % 251 for i in range(size))


def write_output(name: str = "MiniMaxH3_00001.mp4", data: bytes | None = None) -> str:
    """Put a file where ComfyUI would have put it, and return its path."""
    path = os.path.join(handler.COMFY_OUTPUT_DIR, name)
    with open(path, "wb") as handle:
        handle.write(data if data is not None else sample_video())
    return path


def history_for(name: str = "MiniMaxH3_00001.mp4") -> dict:
    return {"outputs": {"9": {"images": [{"filename": name, "subfolder": "", "type": "output"}]}}}


class CapturingStore(handler.OutputStore):
    """Stands in for persistent storage and keeps whatever bytes it was given.

    This is the seam that matters: whatever arrives here is what R2 would have received.
    """

    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def store(self, path, entry, protected=None):
        with open(path, "rb") as handle:
            body = handle.read()
        self.calls.append({"path": path, "entry": entry, "protected": protected, "body": body})
        if self.fail:
            raise handler.WorkflowError("simulated upload failure")
        return {
            "filename": entry["filename"],
            "size": len(body),
            "contentType": protected.content_type if protected else "video/mp4",
        }

    @property
    def body(self) -> bytes:
        return self.calls[0]["body"]


def confidential_protector(key: bytes | None = None, **kwargs):
    return artifacts.ConfidentialProtector(
        artifacts.EphemeralKey(key or secrets.token_bytes(32)), **kwargs
    )


# ======================================================================================
# The canonical Confidential Generation integration test
# ======================================================================================


class TestConfidentialPipeline(unittest.TestCase):
    """Known plaintext in, ciphertext out, and only the client's key opens it."""

    def setUp(self):
        self.plaintext = sample_video(8192)
        self.client_key = secrets.token_bytes(32)
        self.path = write_output(data=self.plaintext)

    def test_stored_bytes_are_not_the_plaintext_and_only_the_client_key_opens_them(self):
        store = CapturingStore()
        protector = confidential_protector(self.client_key)

        results = handler.collect_outputs(
            history_for(), store=store, protector=protector, generation_id="gen-1"
        )

        stored = store.body

        # 1. Persistent storage did not receive the video.
        self.assertNotEqual(stored, self.plaintext)
        self.assertNotIn(self.plaintext, stored)
        self.assertFalse(
            stored.startswith(MP4_SIGNATURE),
            "the stored object must not begin with an MP4 signature",
        )
        self.assertTrue(stored.startswith(artifacts.MAGIC), "stored bytes must be a container")

        # 2. The client's key recovers exactly what the model produced.
        recovered, header = artifacts.decrypt_container(stored, self.client_key)
        self.assertEqual(recovered, self.plaintext)
        self.assertEqual(header["artifactId"], "gen-1")
        self.assertEqual(header["contentType"], "video/mp4")

        # 3. Any other key does not.
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_container(stored, secrets.token_bytes(32))

        # And nothing in the returned metadata is key material.
        self.assertNotIn(base64.b64encode(self.client_key).decode(), json.dumps(results))
        self.assertNotIn(self.client_key.hex(), json.dumps(results))

    def test_plaintext_does_not_survive_the_pipeline(self):
        store = CapturingStore()
        handler.collect_outputs(
            history_for(), store=store, protector=confidential_protector(self.client_key),
            generation_id="gen-1",
        )
        self.assertFalse(os.path.exists(self.path), "the plaintext MP4 must be gone")
        self.assertFalse(
            os.path.exists(store.calls[0]["path"]), "the ciphertext scratch file must be gone too"
        )

    def test_standard_mode_is_untouched(self):
        store = CapturingStore()
        results = handler.collect_outputs(
            history_for(), store=store, protector=artifacts.PassthroughProtector()
        )
        self.assertEqual(store.body, self.plaintext, "standard mode must not transform the video")
        self.assertEqual(store.calls[0]["protected"].content_type, "video/mp4")
        self.assertFalse(store.calls[0]["protected"].encrypted)
        self.assertEqual(results[0]["filename"], "MiniMaxH3_00001.mp4")


# ======================================================================================
# Container format
# ======================================================================================


class TestContainerFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(VECTOR_PATH, encoding="utf-8") as handle:
            cls.vector = json.load(handle)

    def test_layout_matches_the_documented_offsets(self):
        container = bytes.fromhex(self.vector["container_hex"])
        self.assertEqual(container[:4], b"CGEN")
        self.assertEqual(container[4], artifacts.FORMAT_VERSION)
        self.assertEqual(container[5], artifacts.SUITE_AES_256_GCM)
        self.assertEqual(int.from_bytes(container[6:8], "big"), self.vector["header_length"])

    def test_reproduces_the_conformance_vector_byte_for_byte(self):
        """The producer must still emit exactly these bytes.

        This is what catches a change to the canonical header serialization - a key order,
        a space - that would otherwise only surface as an authentication failure on a
        user's video months later.
        """
        destination = io.BytesIO()
        artifacts.encrypt_stream(
            io.BytesIO(bytes.fromhex(self.vector["plaintext_hex"])),
            destination,
            key=artifacts.EphemeralKey(bytes.fromhex(self.vector["key_hex"])),
            header=self.vector["header"],
            nonce=bytes.fromhex(self.vector["nonce_hex"]),
        )
        self.assertEqual(destination.getvalue().hex(), self.vector["container_hex"])

    def test_the_vector_decrypts_to_its_stated_plaintext(self):
        plaintext, header = artifacts.decrypt_container(
            bytes.fromhex(self.vector["container_hex"]), bytes.fromhex(self.vector["key_hex"])
        )
        self.assertEqual(plaintext.hex(), self.vector["plaintext_hex"])
        self.assertEqual(header, self.vector["header"])

    def test_parse_rejects_things_that_are_not_containers(self):
        cases = {
            "an MP4": sample_video(),
            "empty": b"",
            "too short": b"CGEN",
            "wrong magic": b"XXXX" + b"\x01\x01\x00\x10" + b"{}" * 8,
        }
        for name, data in cases.items():
            with self.subTest(name):
                with self.assertRaises(artifacts.ContainerError):
                    artifacts.parse_container_prefix(data)

    def test_parse_rejects_unsupported_versions_and_suites(self):
        good = bytes.fromhex(self.vector["container_hex"])

        future_version = bytearray(good)
        future_version[4] = 2
        with self.assertRaisesRegex(artifacts.ContainerError, "version"):
            artifacts.parse_container_prefix(bytes(future_version))

        unknown_suite = bytearray(good)
        unknown_suite[5] = 9
        with self.assertRaisesRegex(artifacts.ContainerError, "suite"):
            artifacts.parse_container_prefix(bytes(unknown_suite))

    def test_parse_rejects_a_truncated_header(self):
        good = bytes.fromhex(self.vector["container_hex"])
        with self.assertRaisesRegex(artifacts.ContainerError, "[Tt]runcated"):
            artifacts.parse_container_prefix(good[:20])

    def test_parse_rejects_an_absurd_header_length(self):
        broken = bytearray(bytes.fromhex(self.vector["container_hex"]))
        broken[6:8] = (artifacts.MAX_HEADER_BYTES + 1).to_bytes(2, "big")
        with self.assertRaisesRegex(artifacts.ContainerError, "header length"):
            artifacts.parse_container_prefix(bytes(broken))

    def test_header_serialization_is_canonical(self):
        a = artifacts.canonical_header({"b": 1, "a": 2})
        b = artifacts.canonical_header({"a": 2, "b": 1})
        self.assertEqual(a, b, "key order must not change the bytes")
        self.assertEqual(a, b'{"a":2,"b":1}')

    def test_an_oversized_header_is_refused_rather_than_written(self):
        with self.assertRaisesRegex(artifacts.ContainerError, "over the"):
            artifacts.canonical_header({"junk": "x" * (artifacts.MAX_HEADER_BYTES + 10)})


# ======================================================================================
# Encryption properties
# ======================================================================================


class TestEncryption(unittest.TestCase):
    def setUp(self):
        self.key = secrets.token_bytes(32)
        self.plaintext = sample_video()

    def seal(self, key=None, nonce=None, generation_id="gen"):
        destination = io.BytesIO()
        artifacts.encrypt_stream(
            io.BytesIO(self.plaintext),
            destination,
            key=artifacts.EphemeralKey(key or self.key),
            header=artifacts.build_header(
                generation_id=generation_id,
                content_type="video/mp4",
                plaintext_bytes=len(self.plaintext),
            ),
            nonce=nonce,
        )
        return destination.getvalue()

    def test_ciphertext_differs_from_plaintext(self):
        container = self.seal()
        self.assertNotIn(self.plaintext, container)
        self.assertNotEqual(container, self.plaintext)

    def test_the_correct_key_decrypts(self):
        self.assertEqual(artifacts.decrypt_container(self.seal(), self.key)[0], self.plaintext)

    def test_a_wrong_key_fails_authentication(self):
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_container(self.seal(), secrets.token_bytes(32))

    def test_modified_ciphertext_fails_authentication(self):
        container = bytearray(self.seal())
        # A byte in the middle of the payload, well past the header.
        container[-64] ^= 0x01
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_container(bytes(container), self.key)

    def test_a_modified_tag_fails(self):
        container = bytearray(self.seal())
        container[-1] ^= 0x80
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_container(bytes(container), self.key)

    def test_a_modified_header_fails_because_it_is_authenticated(self):
        """The header is the AEAD's associated data, so it cannot be rewritten silently."""
        container = self.seal(generation_id="job-a")
        parsed = artifacts.parse_container_prefix(container)
        forged_header = parsed["header_bytes"].replace(b"job-a", b"job-b")
        self.assertEqual(len(forged_header), len(parsed["header_bytes"]), "same length swap")

        forged = bytearray(container)
        forged[artifacts.PREAMBLE_BYTES : artifacts.PREAMBLE_BYTES + len(forged_header)] = forged_header

        # The header now reads job-b, and parsing happily says so...
        self.assertEqual(artifacts.parse_container_prefix(bytes(forged))["header"]["artifactId"], "job-b")
        # ...but the ciphertext refuses to open under it.
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_container(bytes(forged), self.key)

    def test_truncation_fails(self):
        with self.assertRaises(artifacts.ContainerError):
            artifacts.decrypt_container(self.seal()[:-8], self.key)

    def test_every_encryption_uses_a_fresh_nonce(self):
        """Same key, same plaintext, many times - as a RunPod retry would.

        Nonce reuse under one key is the single fatal misuse of GCM, and a deterministic
        object path makes 'reuse the nonce because the generation id is the same' an easy
        mistake to make. It is not made here: the nonce is drawn per call, never derived.
        """
        nonces = set()
        ciphertexts = set()
        for _ in range(32):
            container = self.seal()
            parsed = artifacts.parse_container_prefix(container)
            nonces.add(container[parsed["nonce_offset"] : parsed["ciphertext_offset"]])
            ciphertexts.add(container[parsed["ciphertext_offset"] :])

        self.assertEqual(len(nonces), 32, "a nonce was reused")
        self.assertEqual(len(ciphertexts), 32, "identical ciphertext means identical nonce")

    def test_ciphertext_is_the_same_length_as_the_plaintext(self):
        container = self.seal()
        parsed = artifacts.parse_container_prefix(container)
        payload = container[parsed["ciphertext_offset"] :]
        self.assertEqual(len(payload) - artifacts.TAG_BYTES, len(self.plaintext))

    def test_streaming_matches_a_single_pass(self):
        """Chunk size must not change the output - the tag is over the whole stream."""
        small = io.BytesIO()
        artifacts.encrypt_stream(
            io.BytesIO(self.plaintext),
            small,
            key=artifacts.EphemeralKey(self.key),
            header=artifacts.build_header(
                generation_id="gen", content_type="video/mp4",
                plaintext_bytes=len(self.plaintext), created_at="2026-01-01T00:00:00Z",
            ),
            nonce=b"\x01" * 12,
            chunk_bytes=17,
        )
        large = io.BytesIO()
        artifacts.encrypt_stream(
            io.BytesIO(self.plaintext),
            large,
            key=artifacts.EphemeralKey(self.key),
            header=artifacts.build_header(
                generation_id="gen", content_type="video/mp4",
                plaintext_bytes=len(self.plaintext), created_at="2026-01-01T00:00:00Z",
            ),
            nonce=b"\x01" * 12,
            chunk_bytes=1 << 20,
        )
        self.assertEqual(small.getvalue(), large.getvalue())


# ======================================================================================
# Privacy modes and request validation
# ======================================================================================


class TestPrivacyModes(unittest.TestCase):
    def test_implemented_modes_resolve(self):
        for name in ("standard", "confidential"):
            self.assertTrue(artifacts.privacy_mode(name).implemented)

    def test_default_is_standard(self):
        self.assertEqual(artifacts.privacy_mode(None).name, "standard")
        self.assertEqual(artifacts.privacy_mode("").name, "standard")

    def test_declared_but_unimplemented_modes_are_refused_with_a_reason(self):
        for name in ("private", "ephemeral"):
            with self.subTest(name):
                with self.assertRaisesRegex(artifacts.ArtifactError, "not available yet"):
                    artifacts.privacy_mode(name)
                self.assertTrue(artifacts.PRIVACY_MODES[name].reason)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "Unknown privacyMode"):
            artifacts.privacy_mode("super-secret")

    def test_the_registry_agrees_with_the_worker(self):
        """Both implementations must know the same four modes, with the same semantics."""
        with open(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker.js"),
            encoding="utf-8",
        ) as handle:
            worker_source = handle.read()
        for name, spec in artifacts.PRIVACY_MODES.items():
            with self.subTest(name):
                self.assertIn(f"  {name}: Object.freeze({{", worker_source)
                self.assertIn(f"implemented: {str(spec.implemented).lower()}", worker_source)


class TestRequestValidation(unittest.TestCase):
    def key_b64url(self, size=32):
        return base64.urlsafe_b64encode(secrets.token_bytes(size)).decode().rstrip("=")

    def test_a_valid_256_bit_key_is_accepted(self):
        protector = artifacts.build_protector("confidential", {"key": self.key_b64url()})
        self.assertTrue(protector.encrypts)

    def test_standard_base64_is_accepted_too(self):
        raw = secrets.token_bytes(32)
        protector = artifacts.build_protector(
            "confidential", {"key": base64.b64encode(raw).decode()}
        )
        self.assertTrue(protector.encrypts)

    def test_a_wrong_size_key_is_rejected(self):
        for size in (16, 24, 31, 33, 64):
            with self.subTest(size=size):
                with self.assertRaisesRegex(artifacts.ArtifactError, "32 bytes"):
                    artifacts.build_protector("confidential", {"key": self.key_b64url(size)})

    def test_malformed_base64_is_rejected(self):
        for bad in ("not base64!", "***", "a b c"):
            with self.subTest(bad):
                with self.assertRaisesRegex(artifacts.ArtifactError, "base64url"):
                    artifacts.build_protector("confidential", {"key": bad})

    def test_a_missing_encryption_block_is_rejected(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "requires an encryption block"):
            artifacts.build_protector("confidential", None)

    def test_a_missing_key_is_rejected(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "missing"):
            artifacts.build_protector("confidential", {"algorithm": "AES-256-GCM"})

    def test_an_unsupported_algorithm_is_rejected(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "Unsupported encryption algorithm"):
            artifacts.build_protector(
                "confidential", {"algorithm": "AES-128-CBC", "key": self.key_b64url()}
            )

    def test_standard_mode_with_an_encryption_block_is_rejected(self):
        """Silently ignoring it would leave the caller believing they were encrypted."""
        with self.assertRaisesRegex(artifacts.ArtifactError, "does not encrypt"):
            artifacts.build_protector("standard", {"key": self.key_b64url()})

    def test_a_non_object_kdf_is_rejected(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "kdf must be an object"):
            artifacts.build_protector(
                "confidential", {"key": self.key_b64url(), "kdf": "argon2id"}
            )

    def test_kdf_metadata_reaches_the_container(self):
        kdf = {"name": "argon2id", "salt": "c2FsdA", "parameters": {"memorySize": 65536}}
        protector = artifacts.build_protector(
            "confidential", {"key": self.key_b64url(), "kdf": kdf, "keyId": "kid-1"}
        )
        path = write_output("kdf.mp4")
        protected = protector.protect(
            artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
        )
        try:
            with open(protected.path, "rb") as handle:
                header = artifacts.parse_container_prefix(handle.read(4096))["header"]
            self.assertEqual(header["kdf"], kdf)
            self.assertEqual(header["keyId"], "kid-1")
        finally:
            protected.discard()


# ======================================================================================
# Key handling
# ======================================================================================


class TestEphemeralKey(unittest.TestCase):
    def test_the_key_refuses_to_render_itself(self):
        raw = secrets.token_bytes(32)
        key = artifacts.EphemeralKey(raw)
        for rendered in (repr(key), str(key), f"{key}", f"{key!s}", f"{key!r}", "{}".format(key)):
            self.assertIn("redacted", rendered)
            self.assertNotIn(raw.hex(), rendered)
            self.assertNotIn(base64.b64encode(raw).decode(), rendered)

    def test_a_key_inside_a_logged_structure_is_redacted(self):
        raw = secrets.token_bytes(32)
        payload = {"encryption": {"key": base64.b64encode(raw).decode(), "algorithm": "AES-256-GCM"}}
        cleaned = json.dumps(artifacts.redact(payload))
        self.assertNotIn(base64.b64encode(raw).decode(), cleaned)
        self.assertIn("AES-256-GCM", cleaned, "non-secret context must survive")

    def test_destroy_zeroes_the_buffer_and_blocks_reuse(self):
        key = artifacts.EphemeralKey(secrets.token_bytes(32))
        key.destroy()
        self.assertTrue(key.destroyed)
        with self.assertRaisesRegex(artifacts.ArtifactError, "destroyed"):
            _ = key.bytes

    def test_destroy_is_idempotent(self):
        key = artifacts.EphemeralKey(secrets.token_bytes(32))
        key.destroy()
        key.destroy()

    def test_a_wrong_length_key_cannot_be_constructed(self):
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.EphemeralKey(b"short")

    def test_the_protector_destroys_its_key(self):
        key = artifacts.EphemeralKey(secrets.token_bytes(32))
        protector = artifacts.ConfidentialProtector(key)
        protector.destroy()
        self.assertTrue(key.destroyed)


# ======================================================================================
# Cleanup, on every path
# ======================================================================================


class TestCleanup(unittest.TestCase):
    def test_plaintext_is_deleted_after_encryption(self):
        path = write_output("cleanup-ok.mp4")
        protector = confidential_protector()
        protected = protector.protect(
            artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
        )
        try:
            self.assertFalse(os.path.exists(path))
            self.assertTrue(protected.plaintext_removed)
            self.assertTrue(os.path.isfile(protected.path))
        finally:
            protected.discard()

    def test_discard_removes_the_ciphertext_scratch_directory(self):
        path = write_output("cleanup-discard.mp4")
        protected = confidential_protector().protect(
            artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
        )
        scratch = protected.scratch_dir
        protected.discard()
        self.assertFalse(os.path.exists(scratch))
        protected.discard()  # idempotent

    def test_a_failed_encryption_leaves_no_ciphertext_behind(self):
        path = write_output("cleanup-fail.mp4")

        class Exploding(artifacts.ConfidentialProtector):
            seen = []

            def protect(self, artifact):
                # Fail after the scratch directory exists but before the container does.
                scratch = artifacts.make_scratch_dir("cg-enc-")
                Exploding.seen.append(scratch)
                import shutil

                shutil.rmtree(scratch, ignore_errors=True)
                raise artifacts.ArtifactError("encryption exploded")

        store = CapturingStore()
        with self.assertRaises(artifacts.ArtifactError):
            handler.collect_outputs(
                history_for("cleanup-fail.mp4"),
                store=store,
                protector=Exploding(artifacts.EphemeralKey(secrets.token_bytes(32))),
            )

        self.assertEqual(store.calls, [], "nothing may be uploaded when encryption fails")
        for scratch in Exploding.seen:
            self.assertFalse(os.path.exists(scratch))

    def test_a_failed_upload_still_deletes_plaintext_and_ciphertext(self):
        path = write_output("cleanup-upload.mp4")
        store = CapturingStore(fail=True)

        with self.assertRaises(handler.WorkflowError):
            handler.collect_outputs(
                history_for("cleanup-upload.mp4"), store=store, protector=confidential_protector()
            )

        self.assertFalse(os.path.exists(path), "plaintext must be gone even when upload fails")
        self.assertFalse(
            os.path.exists(store.calls[0]["path"]), "the ciphertext temp file must be gone too"
        )

    def test_keep_outputs_never_preserves_confidential_plaintext(self):
        """H3_KEEP_OUTPUTS is a debugging convenience. It is not a licence to keep plaintext."""
        path = write_output("keep-outputs.mp4")
        previous = os.environ.get("H3_KEEP_OUTPUTS")
        os.environ["H3_KEEP_OUTPUTS"] = "1"
        try:
            handler.collect_outputs(
                history_for("keep-outputs.mp4"),
                store=CapturingStore(),
                protector=confidential_protector(),
            )
            self.assertFalse(os.path.exists(path))
        finally:
            if previous is None:
                os.environ.pop("H3_KEEP_OUTPUTS", None)
            else:
                os.environ["H3_KEEP_OUTPUTS"] = previous

    def test_keep_outputs_still_works_for_standard_generations(self):
        path = write_output("keep-standard.mp4")
        previous = os.environ.get("H3_KEEP_OUTPUTS")
        os.environ["H3_KEEP_OUTPUTS"] = "1"
        try:
            handler.collect_outputs(
                history_for("keep-standard.mp4"),
                store=CapturingStore(),
                protector=artifacts.PassthroughProtector(),
            )
            self.assertTrue(os.path.exists(path), "standard behaviour must not change")
        finally:
            os.remove(path)
            if previous is None:
                os.environ.pop("H3_KEEP_OUTPUTS", None)
            else:
                os.environ["H3_KEEP_OUTPUTS"] = previous

    def test_the_scratch_directory_is_private(self):
        scratch = artifacts.make_scratch_dir("cg-perm-")
        try:
            if os.name != "nt":
                import stat

                mode = stat.S_IMODE(os.stat(scratch).st_mode)
                self.assertEqual(mode, 0o700, f"expected 0700, got {oct(mode)}")
            self.assertNotEqual(
                os.path.basename(scratch), "cg-perm-", "the name must be unpredictable"
            )
        finally:
            os.rmdir(scratch)

    def test_an_artefact_outside_our_own_directories_is_refused_not_shredded(self):
        """Confidential mode deletes the plaintext, so it must not accept a stray path.

        `path` comes from ComfyUI's history entry. That is trustworthy in normal operation
        and is exactly the wrong thing to trust when the next step is an unconditional
        delete, so the escape is refused before anything is encrypted.
        """
        outside = os.path.join(tempfile.mkdtemp(prefix="not-ours-"), "elsewhere.mp4")
        with open(outside, "wb") as handle:
            handle.write(sample_video())

        history = {
            "outputs": {
                "9": {
                    "images": [
                        {
                            # Escapes COMFY_OUTPUT_DIR via the subfolder.
                            "filename": "elsewhere.mp4",
                            "subfolder": os.path.relpath(
                                os.path.dirname(outside), handler.COMFY_OUTPUT_DIR
                            ),
                            "type": "output",
                        }
                    ]
                }
            }
        }

        store = CapturingStore()
        with self.assertRaisesRegex(handler.WorkflowError, "resolved outside"):
            handler.collect_outputs(history, store=store, protector=confidential_protector())

        self.assertEqual(store.calls, [], "nothing may be uploaded")
        self.assertTrue(os.path.exists(outside), "a file we do not own must not be deleted")
        os.remove(outside)

    def test_shred_removes_a_file_and_tolerates_a_missing_one(self):
        path = write_output("shred.mp4")
        self.assertTrue(artifacts.shred(path))
        self.assertFalse(os.path.exists(path))
        self.assertTrue(artifacts.shred(path), "shredding a missing file is not an error")


# ======================================================================================
# Handler wiring
# ======================================================================================


class TestHandlerWiring(unittest.TestCase):
    def key_b64url(self):
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")

    def test_no_privacy_block_means_standard(self):
        protector = handler.build_protector_for_job({"workflow": {}})
        self.assertFalse(protector.encrypts)
        self.assertEqual(protector.mode, "standard")

    def test_a_confidential_job_produces_an_encrypting_protector(self):
        protector = handler.build_protector_for_job(
            {
                "privacy": {"mode": "confidential"},
                "encryption": {"algorithm": "AES-256-GCM", "key": self.key_b64url()},
            }
        )
        self.assertTrue(protector.encrypts)

    def test_a_bad_confidential_job_is_a_workflow_error_not_a_crash(self):
        with self.assertRaises(handler.WorkflowError):
            handler.build_protector_for_job(
                {"privacy": {"mode": "confidential"}, "encryption": {"key": "nope!"}}
            )

    def test_confidential_without_an_upload_endpoint_fails_before_any_gpu_work(self):
        """Fail closed: there is no inline fallback for a confidential artefact."""
        result = handler.handler(
            {
                "id": "job-1",
                "input": {
                    "workflow": {"1": {"class_type": "X", "inputs": {}}},
                    "privacy": {"mode": "confidential"},
                    "encryption": {"key": self.key_b64url()},
                },
            }
        )
        self.assertIn("error", result)
        self.assertIn("output upload endpoint", result["error"])

    def test_an_unknown_privacy_mode_is_rejected_by_the_handler_too(self):
        result = handler.handler(
            {
                "id": "job-2",
                "input": {"workflow": {"1": {}}, "privacy": {"mode": "invisible"}},
            }
        )
        self.assertIn("error", result)
        self.assertIn("Unknown privacyMode", result["error"])

    def test_the_upload_store_labels_ciphertext_correctly(self):
        """A confidential body must never be announced to the Worker as video/mp4."""
        sent = {}

        class FakeResponse:
            status_code = 201
            text = '{"key":"outputs/x/artifact.enc"}'

            @staticmethod
            def json():
                return {"key": "outputs/x/artifact.enc", "encrypted": True}

        def fake_put(url, data=None, headers=None, **kwargs):
            sent.update(headers)
            return FakeResponse()

        original = handler.requests.put
        handler.requests.put = fake_put
        try:
            path = write_output("upload-label.mp4")
            protected = confidential_protector().protect(
                artifacts.GeneratedArtifact(path=path, mime_type="video/mp4", generation_id="g")
            )
            store = handler.WorkerUploadStore("https://w/o", "tok", 30)
            result = store.store(protected.path, {"filename": "v.mp4"}, protected)
            protected.discard()
        finally:
            handler.requests.put = original

        self.assertEqual(sent["Content-Type"], "application/octet-stream")
        self.assertEqual(result["privacyMode"], "confidential")
        self.assertTrue(result["encrypted"])
        self.assertEqual(result["artifact"]["algorithm"], "AES-256-GCM")
        self.assertNotIn("key", result["artifact"])

    def test_the_generation_id_comes_from_the_block_the_storage_key_is_derived_from(self):
        """The container header must name the same job the upload endpoint will check."""
        self.assertEqual(
            handler.build_protector_for_job({"privacy": {"mode": "standard"}}).mode, "standard"
        )

        captured = {}

        class Recording(artifacts.PassthroughProtector):
            def protect(self, artifact):
                captured["id"] = artifact.generation_id
                return super().protect(artifact)

        write_output("genid.mp4")
        handler.collect_outputs(
            history_for("genid.mp4"),
            store=CapturingStore(),
            protector=Recording(),
            generation_id="worker-job-id",
        )
        self.assertEqual(captured["id"], "worker-job-id")

    def test_the_perf_line_quantifies_what_encryption_cost(self):
        """The point of measuring is being able to answer 'how much does this cost?'."""
        timer = handler.JobTimer({})
        timer.privacy_mode = "confidential"
        timer.add_span("encryption", 0.084)
        line = timer.summary(job_index=1, status="ok")

        self.assertIn("privacy=confidential", line)
        self.assertIn("encryption_ms=84", line)

    def test_the_perf_line_is_unchanged_for_standard_jobs(self):
        line = handler.JobTimer({}).summary(job_index=1, status="ok")
        self.assertNotIn("privacy=", line)
        self.assertNotIn("encryption_ms=", line)

    def test_standard_uploads_are_unchanged(self):
        sent = {}

        class FakeResponse:
            status_code = 201
            text = '{"key":"outputs/x/video.mp4"}'

            @staticmethod
            def json():
                return {"key": "outputs/x/video.mp4"}

        def fake_put(url, data=None, headers=None, **kwargs):
            sent.update(headers)
            return FakeResponse()

        original = handler.requests.put
        handler.requests.put = fake_put
        try:
            path = write_output("upload-standard.mp4")
            store = handler.WorkerUploadStore("https://w/o", "tok", 30)
            result = store.store(path, {"filename": "v.mp4"})
        finally:
            handler.requests.put = original
            if os.path.exists(path):
                os.remove(path)

        self.assertEqual(sent["Content-Type"], "video/mp4")
        self.assertNotIn("artifact", result)


if __name__ == "__main__":
    unittest.main()
