"""Model-independent protection of generated artefacts.

This module knows nothing about MiniMax H3, ComfyUI, R2 or browsers. It knows how to take
a file that a generative model just produced and turn it into something safe to hand to
persistent storage. That separation is the point: when a second model is added (WAN, LTX,
an image model, an audio model), it reuses this file unchanged.

Three stages, three types:

    GeneratedArtifact   plaintext, on local disk, inside the trusted inference boundary
          |
          |  ArtifactProtector.protect()
          v
    ProtectedArtifact   the bytes that are allowed to leave - ciphertext in confidential
          |             mode, the same plaintext in standard mode
          |  OutputStore.store()
          v
    StoredArtifact      what persistent storage acknowledged (key, size, url)

Only the middle stage lives here. Storage belongs to the caller, because where the bytes
go is a deployment decision and encryption is not.

HYBRID ENCRYPTION (crypto v2)
----------------------------
The worker never receives anything that can decrypt what it produces. It is handed an
*encryption-only public key*, and for each artefact it:

    1. draws a fresh random 256-bit file-encryption key (FEK)
    2. encrypts the media with AES-256-GCM under that key
    3. wraps the FEK to the public key with RSA-OAEP-256
    4. destroys the FEK and deletes the plaintext

The wrapped FEK travels inside the container. The unwrapped one exists only in this
process, for the duration of one encryption. So a complete copy of the job payload, the job
result and the stored object is not enough to recover the media: that needs the private
key, which lives in the caller's browser and never crosses the boundary.

Crypto v1 - where the caller derived a symmetric key from a passphrase and *sent it* -
remains readable so existing artefacts keep working, and is no longer creatable.

WHAT THIS IS NOT
----------------
This is not zero-knowledge anything. The model needs the plaintext prompt, and it produces
plaintext frames. Both exist, in the clear, inside the inference process while it runs, as
does the file key for the moment it is in use. The property this module provides is
narrower and true: *persistent storage receives ciphertext only*, and nothing that reaches
this process can decrypt it afterwards.

THE CONTAINER FORMAT
--------------------
Encrypted artefacts are self-describing, so that a file recovered on its own - from a
backup, a download folder, an object listing - can still be identified and decrypted by
whoever holds the private key. Layout, all integers big-endian::

    offset          size          field
    0               4             MAGIC = b"CGEN"
    4               1             CONTAINER VERSION (1 = symmetric, 2 = hybrid)
    5               1             SUITE (1 = AES-256-GCM, 96-bit nonce, 128-bit tag)
    6               2             HEADER_LEN (uint16)
    8               HEADER_LEN    HEADER, canonical UTF-8 JSON - also the AEAD's
                                  additional authenticated data
    8 + HEADER_LEN  12            NONCE
    20 + HEADER_LEN ...           CIPHERTEXT followed immediately by the 16-byte TAG

A v2 header carries the key-wrapping block::

    "kw": {"alg": "RSA-OAEP-256", "wrappedFileKey": "<base64url>", "keyId": "<derived>"}

Three deliberate choices in that layout:

* **The header is the AAD.** Nobody can rewrite the recorded algorithm, content type,
  artefact id *or the wrapped file key* without the file key: doing so breaks
  authentication. That last one matters most - an attacker cannot substitute a file key
  wrapped to their own public key, because the swap invalidates the tag over the video.
  The container is bound to one key pair and one generation, not merely accompanied by a
  claim about them.

* **The tag trails the ciphertext.** That is exactly the byte range WebCrypto's
  ``crypto.subtle.decrypt`` expects for AES-GCM, so a browser decrypts by slicing from the
  end of the nonce to end-of-file with no reassembly. It also means encryption is a single
  forward pass with no seeking, which is what allows a large file to be encrypted in
  constant memory.

* **No KDF metadata in a v2 container.** The passphrase KDF protects the caller's *private
  key*, which is account-scoped and never sent here. Repeating it inside every video would
  imply the platform had a role in deriving it, and it does not.

The format is versioned at two levels - the container version for the framing, SUITE for
the cipher - so a future XChaCha20-Poly1305 or chunked-streaming variant can be added
without invalidating a single existing artefact. See ``docs/confidential-generation.md``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, BinaryIO, Callable

# --------------------------------------------------------------------------------------
# Container constants
# --------------------------------------------------------------------------------------

MAGIC = b"CGEN"

#: Container versions this build can *read*. v1 is the original symmetric design, where the
#: caller derived the file key from a passphrase and sent it to the worker. v2 is hybrid:
#: the worker draws a random file key and wraps it to a public key it was given, so nothing
#: capable of decrypting the artefact ever travels to the worker.
CONTAINER_V1_SYMMETRIC = 1
CONTAINER_V2_HYBRID = 2
SUPPORTED_CONTAINER_VERSIONS = frozenset({CONTAINER_V1_SYMMETRIC, CONTAINER_V2_HYBRID})

#: The only version this build will *write*. Reading v1 keeps existing artefacts openable;
#: writing it would keep reintroducing the very property v2 exists to remove.
FORMAT_VERSION = CONTAINER_V2_HYBRID

SUITE_AES_256_GCM = 1
SUITE_NAMES = {SUITE_AES_256_GCM: "AES-256-GCM"}
SUITE_IDS = {name: suite for suite, name in SUITE_NAMES.items()}

#: Key-wrapping algorithms. RSA-OAEP with SHA-256 and MGF1-SHA256, empty label - the
#: default for both `cryptography` and WebCrypto, which is why they interoperate with no
#: parameters to agree on. See docs/confidential-generation.md for why this and not ECIES.
KEY_WRAP_RSA_OAEP_256 = "RSA-OAEP-256"
SUPPORTED_KEY_WRAP_ALGORITHMS = frozenset({KEY_WRAP_RSA_OAEP_256})

#: 3072-bit RSA is the NIST 128-bit-security level and is what "modern RSA" means. Larger
#: is accepted; smaller is not. A floor rather than a fixed size means a client that wants
#: 4096 simply generates 4096 and nothing here has to change.
MIN_RSA_MODULUS_BITS = 3072
MAX_RSA_MODULUS_BITS = 8192

KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16

# The fixed part of the header: magic, version, suite, header length.
PREAMBLE_BYTES = 8

# A header is metadata, not a payload. The cap stops a malformed or hostile file from
# making a parser allocate. Real headers are a few hundred bytes.
MAX_HEADER_BYTES = 8192

# Enough of a file to always contain the preamble plus any legal header and the nonce.
MAX_PREFIX_BYTES = PREAMBLE_BYTES + MAX_HEADER_BYTES + NONCE_BYTES

DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024

ENCRYPTED_CONTENT_TYPE = "application/octet-stream"
ENCRYPTED_EXTENSION = ".enc"


class ArtifactError(RuntimeError):
    """Anything that makes an artefact unsafe or impossible to protect."""


class ContainerError(ArtifactError):
    """A byte sequence is not a well-formed confidential container."""


# --------------------------------------------------------------------------------------
# Privacy modes
#
# One registry, mirrored in worker.js. Nothing anywhere else compares privacy-mode strings:
# code asks this table what a mode *does* instead of what it is called, which is what keeps
# a future mode from needing edits scattered through the request path.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyModeSpec:
    name: str
    implemented: bool
    encrypts: bool
    #: True when the mode forbids logging prompt text (only length and digest).
    restricts_prompt_logging: bool
    description: str
    reason: str = ""


PRIVACY_MODES: dict[str, PrivacyModeSpec] = {
    "standard": PrivacyModeSpec(
        name="standard",
        implemented=True,
        encrypts=False,
        restricts_prompt_logging=False,
        description=(
            "Current behaviour, unchanged. The MP4 is stored as-is and streamed back "
            "normally."
        ),
    ),
    "confidential": PrivacyModeSpec(
        name="confidential",
        implemented=True,
        encrypts=True,
        restricts_prompt_logging=True,
        description=(
            "The artefact is encrypted inside the inference environment with a key the "
            "caller derived. Persistent storage receives ciphertext only; the browser "
            "decrypts for playback."
        ),
    ),
    # Declared, not implemented - the same pattern worker.js already uses for the Ref2VA
    # and 2K modes. Naming them here is what makes the abstraction real rather than
    # aspirational: the shape they would take is fixed, and a request for one gets an
    # honest 501 instead of a silent downgrade to something weaker.
    "private": PrivacyModeSpec(
        name="private",
        implemented=False,
        encrypts=False,
        restricts_prompt_logging=True,
        description=(
            "Unencrypted storage with restricted logging and a short default retention."
        ),
        reason=(
            "Needs a retention enforcer to be meaningful. Today's retention is a "
            "prefix-wide R2 lifecycle rule, which cannot express a per-job lifetime, so "
            "'private' would promise a shorter life than the platform can deliver."
        ),
    ),
    "ephemeral": PrivacyModeSpec(
        name="ephemeral",
        implemented=False,
        encrypts=True,
        restricts_prompt_logging=True,
        description=(
            "Confidential, plus the server-side copy is deleted as soon as delivery "
            "succeeds."
        ),
        reason=(
            "Needs a reliable definition of 'delivery succeeded'. A ranged or aborted GET "
            "is not a delivery, and deleting on the first byte read would destroy the "
            "artefact mid-download."
        ),
    ),
}

DEFAULT_PRIVACY_MODE = "standard"


def privacy_mode(name: str | None) -> PrivacyModeSpec:
    """Resolve a mode name, or raise. ``None``/empty means the default."""
    key = (name or DEFAULT_PRIVACY_MODE).strip().lower()
    spec = PRIVACY_MODES.get(key)
    if spec is None:
        supported = ", ".join(sorted(PRIVACY_MODES))
        raise ArtifactError(f"Unknown privacyMode {key!r}. Supported: {supported}")
    if not spec.implemented:
        raise ArtifactError(f"privacyMode {key!r} is not available yet. {spec.reason}")
    return spec


# --------------------------------------------------------------------------------------
# Key material
# --------------------------------------------------------------------------------------


class EphemeralKey:
    """A request-scoped symmetric key that refuses to render itself.

    The single most common way a key ends up in a log is not a deliberate ``log(key)`` -
    it is an f-string interpolating a dict, or a traceback rendering a local, or a debug
    dump of a request. Wrapping the bytes in a type whose ``__repr__`` and ``__str__`` are
    redacted removes that whole class of accident: the key can still be used, but it cannot
    be printed by accident.

    ``destroy()`` zeroes the backing ``bytearray``. Be clear about what that is worth:
    CPython may already have copied these bytes elsewhere (an intermediate ``bytes``, a
    freed buffer, the OS page cache, a swap file), and the interpreter offers no way to find
    or erase those copies. This narrows the window; it does not close it. See "Secrets in
    memory" in docs/confidential-generation.md.
    """

    __slots__ = ("_buffer", "_destroyed")

    def __init__(self, raw: bytes | bytearray) -> None:
        if len(raw) != KEY_BYTES:
            raise ArtifactError(
                f"Encryption key must be exactly {KEY_BYTES} bytes, got {len(raw)}."
            )
        self._buffer = bytearray(raw)
        self._destroyed = False

    @property
    def bytes(self) -> bytes:
        if self._destroyed:
            raise ArtifactError("Encryption key has already been destroyed.")
        return bytes(self._buffer)

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def destroy(self) -> None:
        for index in range(len(self._buffer)):
            self._buffer[index] = 0
        self._destroyed = True

    def __enter__(self) -> "EphemeralKey":
        return self

    def __exit__(self, *_exc) -> None:
        self.destroy()

    def __repr__(self) -> str:
        return f"<EphemeralKey {KEY_BYTES} bytes, redacted>"

    __str__ = __repr__

    def __format__(self, _spec: str) -> str:
        return repr(self)


def decode_key(value: str) -> EphemeralKey:
    """Decode a base64url (or base64) encoded 256-bit key.

    Padding is optional and both alphabets are accepted, because a browser's
    ``base64url`` and a curl user's ``base64`` should not be a support ticket. Every
    failure message describes the *shape* of what arrived, never its content.
    """
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError("Encryption key is missing.")

    text = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-+/]+={0,2}", text):
        raise ArtifactError("Encryption key is not valid base64url.")

    normalized = text.replace("-", "+").replace("_", "/")
    normalized += "=" * (-len(normalized) % 4)
    try:
        raw = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ArtifactError(f"Encryption key is not valid base64url: {error}") from error

    if len(raw) != KEY_BYTES:
        raise ArtifactError(
            f"Encryption key must decode to {KEY_BYTES} bytes (256 bits), got {len(raw)}."
        )
    return EphemeralKey(raw)


# --------------------------------------------------------------------------------------
# Public keys and key wrapping (crypto v2)
#
# The whole point of v2 lives in this section: the only key material the inference worker
# ever receives is a public key, which can encrypt and cannot decrypt. Everything that
# could open an artefact - the passphrase, the key-encryption key it derives, the private
# key - stays in the browser and never enters a job payload.
# --------------------------------------------------------------------------------------


class PublicEncryptionKey:
    """An encryption-only public key, loaded from its SPKI DER encoding.

    SPKI is what WebCrypto's ``exportKey("spki")`` produces and what `cryptography`'s
    ``load_der_public_key`` consumes, so no format has to be agreed on beyond "the standard
    one". The key id is *derived* from those exact bytes rather than supplied: a caller
    cannot label a key as something it is not, and two parties independently computing the
    id always agree.
    """

    __slots__ = ("_key", "spki", "key_id", "algorithm", "modulus_bits")

    def __init__(self, spki: bytes) -> None:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives.serialization import load_der_public_key

        try:
            key = load_der_public_key(spki)
        except Exception as error:
            raise ArtifactError(f"Public key is not a valid SPKI DER encoding: {error}") from error

        if not isinstance(key, rsa.RSAPublicKey):
            raise ArtifactError(
                f"Unsupported public key type {type(key).__name__}. This build wraps with "
                f"{KEY_WRAP_RSA_OAEP_256}, which needs an RSA key."
            )
        if key.key_size < MIN_RSA_MODULUS_BITS:
            raise ArtifactError(
                f"RSA public key is {key.key_size} bits; the minimum is "
                f"{MIN_RSA_MODULUS_BITS}."
            )
        if key.key_size > MAX_RSA_MODULUS_BITS:
            raise ArtifactError(
                f"RSA public key is {key.key_size} bits; the maximum is "
                f"{MAX_RSA_MODULUS_BITS}."
            )

        self._key = key
        self.spki = bytes(spki)
        self.modulus_bits = key.key_size
        self.algorithm = KEY_WRAP_RSA_OAEP_256
        self.key_id = public_key_id(self.spki)

    def wrap(self, file_key: "EphemeralKey") -> bytes:
        """Wrap a file-encryption key to this public key.

        One library call, deliberately. The alternative designs - ECIES, HPKE - would have
        this method compose an ephemeral keypair, a key agreement, a KDF and a second AEAD,
        all inside the most security-critical code in the system. RSA-OAEP does it in one
        primitive that both `cryptography` and WebCrypto implement natively.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        return self._key.encrypt(
            file_key.bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def __repr__(self) -> str:
        return f"<PublicEncryptionKey {self.algorithm} {self.modulus_bits}-bit {self.key_id}>"


def public_key_id(spki: bytes) -> str:
    """A stable, verifiable name for a public key: the truncated SHA-256 of its SPKI.

    Derived rather than chosen, so a key id can never name a key it does not belong to.
    Both the browser and the worker compute it the same way from the same bytes.
    """
    digest = hashlib.sha256(spki).digest()
    return base64.urlsafe_b64encode(digest[:16]).decode("ascii").rstrip("=")


def decode_public_key(value: str) -> PublicEncryptionKey:
    """Decode a base64url (or base64) SPKI public key.

    Failure messages describe the shape of what arrived. A public key is not secret, but
    quoting caller-supplied bytes back into an error message is a habit worth not having.
    """
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError("Public encryption key is missing.")

    text = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-+/\s]+={0,2}", text):
        raise ArtifactError("Public encryption key is not valid base64url.")

    normalized = re.sub(r"\s+", "", text).replace("-", "+").replace("_", "/")
    normalized += "=" * (-len(normalized) % 4)
    try:
        spki = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ArtifactError(f"Public encryption key is not valid base64url: {error}") from error

    return PublicEncryptionKey(spki)


# --------------------------------------------------------------------------------------
# Artefact types
# --------------------------------------------------------------------------------------


@dataclass
class GeneratedArtifact:
    """A plaintext file a model has just produced, inside the trusted boundary."""

    path: str
    mime_type: str
    generation_id: str
    filename: str | None = None

    @property
    def name(self) -> str:
        return self.filename or os.path.basename(self.path)

    @property
    def size(self) -> int:
        return os.path.getsize(self.path)


@dataclass
class ProtectedArtifact:
    """The bytes that are allowed to leave the trusted boundary.

    In standard mode this *is* the generated artefact - same path, same bytes, nothing
    added. In confidential mode it is a container file, and ``plaintext_removed`` is True
    because the plaintext was deleted the moment encryption finished.
    """

    path: str
    content_type: str
    encrypted: bool
    generation_id: str
    filename: str
    size: int
    #: Non-secret description of the protection, safe to store and to return over the API.
    metadata: dict[str, Any] = field(default_factory=dict)
    plaintext_removed: bool = False
    encryption_seconds: float = 0.0
    #: Directory to remove once the bytes have been handed to storage. None in standard
    #: mode, where the artefact lives in a directory the caller already manages.
    scratch_dir: str | None = None

    def discard(self) -> None:
        """Remove everything this protector created. Safe to call more than once."""
        if self.scratch_dir:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
            self.scratch_dir = None


@dataclass
class StoredArtifact:
    """What persistent storage acknowledged. Built by the caller, not by this module."""

    storage_key: str
    encrypted: bool
    content_type: str
    size: int
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Filesystem hygiene
# --------------------------------------------------------------------------------------


def make_scratch_dir(prefix: str = "cg-") -> str:
    """A private directory for transient plaintext-derived data.

    ``mkdtemp`` already creates with mode 0700 and an unpredictable name; the explicit
    chmod is there so the guarantee survives an inherited umask on an unusual base image.
    """
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:  # pragma: no cover - Windows and some overlay filesystems
        pass
    return path


#: Overwriting before unlinking is cheap for a five-second clip and occasionally useful on
#: a plain ext4 volume. It is worth close to nothing on overlayfs, copy-on-write
#: filesystems or SSDs with wear levelling, which is where these containers actually run.
#: Set H3_OVERWRITE_PLAINTEXT=0 to skip it.
OVERWRITE_BEFORE_DELETE = os.environ.get("H3_OVERWRITE_PLAINTEXT", "1") == "1"


def shred(path: str, *, on_warning: Callable[[str], None] | None = None) -> bool:
    """Delete a file, best effort, overwriting it first when that is likely to help.

    Returns True if the file is gone afterwards. Never raises: this runs in ``finally``
    blocks, where the interesting failure is the one already in flight.
    """
    try:
        if not os.path.isfile(path):
            return True
        if OVERWRITE_BEFORE_DELETE:
            try:
                size = os.path.getsize(path)
                with open(path, "r+b", buffering=0) as handle:
                    remaining = size
                    while remaining > 0:
                        block = min(remaining, 1024 * 1024)
                        handle.write(secrets.token_bytes(block))
                        remaining -= block
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as error:
                if on_warning:
                    on_warning(f"could not overwrite before deleting: {error}")
        os.remove(path)
        return True
    except OSError as error:
        if on_warning:
            on_warning(f"could not delete temporary file: {error}")
        return not os.path.exists(path)


# --------------------------------------------------------------------------------------
# Container encode / decode
# --------------------------------------------------------------------------------------


def canonical_header(header: dict[str, Any]) -> bytes:
    """Serialize a header deterministically.

    Sorted keys and no whitespace, so the same inputs always produce the same bytes. That
    matters because the header *is* the AEAD's additional data: a re-serialization that
    differed by one space would fail authentication.
    """
    encoded = json.dumps(
        header, separators=(",", ":"), sort_keys=True, ensure_ascii=True
    ).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ContainerError(
            f"Container header is {len(encoded)} bytes, over the {MAX_HEADER_BYTES} limit."
        )
    return encoded


def build_v2_header(
    *,
    generation_id: str,
    content_type: str,
    plaintext_bytes: int,
    wrapped_file_key: bytes,
    key_id: str,
    key_wrap_algorithm: str = KEY_WRAP_RSA_OAEP_256,
    privacy_mode: str = "confidential",
    created_at: str | None = None,
    suite: int = SUITE_AES_256_GCM,
) -> dict[str, Any]:
    """The non-secret metadata carried inside a hybrid (v2) container.

    Everything here is authenticated: the header is the AEAD's associated data, so none of
    it can be altered without the file key, which nobody has. That is what makes putting
    the *wrapped file key in the header* the right place for it - an attacker cannot
    substitute a file key wrapped to their own public key, because doing so invalidates the
    tag over the video. The container is therefore bound to one key pair and one
    generation, not merely accompanied by a claim about them.

    What is authenticated, exactly: container version, cipher suite name, generation id,
    plaintext media type and length, creation time, privacy mode, and the entire
    key-wrapping block (algorithm, wrapped file key, key id).
    """
    return {
        "v": CONTAINER_V2_HYBRID,
        "alg": SUITE_NAMES[suite],
        "artifactId": str(generation_id),
        "contentType": str(content_type),
        "plaintextBytes": int(plaintext_bytes),
        "privacyMode": privacy_mode,
        "createdAt": created_at
        or datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "kw": {
            "alg": key_wrap_algorithm,
            "wrappedFileKey": base64.urlsafe_b64encode(wrapped_file_key).decode("ascii").rstrip("="),
            "keyId": key_id,
        },
    }


def build_header(
    *,
    generation_id: str,
    content_type: str,
    plaintext_bytes: int,
    kdf: dict[str, Any] | None = None,
    key_id: str | None = None,
    created_at: str | None = None,
    suite: int = SUITE_AES_256_GCM,
) -> dict[str, Any]:
    """Assemble the non-secret metadata carried inside a **v1** container.

    Kept so that artefacts written before the hybrid design remain readable and their
    conformance vector still reproduces. Nothing calls this to write new artefacts - see
    build_v2_header.

    Everything here is public by construction. ``plaintextBytes`` leaks nothing the file
    size does not already: AES-GCM is length preserving, so the ciphertext is exactly as
    long as the plaintext.
    """
    header: dict[str, Any] = {
        "v": CONTAINER_V1_SYMMETRIC,
        "alg": SUITE_NAMES[suite],
        "artifactId": str(generation_id),
        "contentType": str(content_type),
        "plaintextBytes": int(plaintext_bytes),
        "createdAt": created_at
        or datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    if kdf:
        header["kdf"] = kdf
    if key_id:
        header["keyId"] = str(key_id)
    return header


def parse_container_prefix(data: bytes) -> dict[str, Any]:
    """Parse the framing and header from the first bytes of a container.

    Accepts any prefix long enough to cover the header; returns the parsed header plus the
    offsets a reader needs. Raises ``ContainerError`` for anything that is not a container,
    which is what makes "are these bytes really ciphertext?" a cheap question to ask.
    """
    if len(data) < PREAMBLE_BYTES:
        raise ContainerError(
            f"Too short to be an encrypted artefact: {len(data)} bytes, need at least "
            f"{PREAMBLE_BYTES}."
        )
    if bytes(data[:4]) != MAGIC:
        raise ContainerError("Missing container magic - these bytes are not ciphertext.")

    version = data[4]
    suite = data[5]
    if version not in SUPPORTED_CONTAINER_VERSIONS:
        raise ContainerError(
            f"Unsupported container version {version}; this build understands "
            f"{sorted(SUPPORTED_CONTAINER_VERSIONS)}."
        )
    if suite not in SUITE_NAMES:
        raise ContainerError(f"Unsupported cipher suite {suite}.")

    header_len = int.from_bytes(data[6:8], "big")
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise ContainerError(f"Illegal header length {header_len}.")

    header_end = PREAMBLE_BYTES + header_len
    if len(data) < header_end:
        raise ContainerError(f"Truncated header: need {header_end} bytes, have {len(data)}.")

    try:
        header = json.loads(bytes(data[PREAMBLE_BYTES:header_end]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContainerError(f"Container header is not valid JSON: {error}") from error

    if not isinstance(header, dict):
        raise ContainerError("Container header must be a JSON object.")

    # Version-specific header requirements. Checked here rather than at each call site so
    # that "this parsed" always means "this is structurally a container of that version",
    # and nothing downstream has to re-derive what a v2 header must contain.
    if version == CONTAINER_V2_HYBRID:
        wrap = header.get("kw")
        if not isinstance(wrap, dict):
            raise ContainerError("A v2 container must carry a 'kw' key-wrapping block.")
        if wrap.get("alg") not in SUPPORTED_KEY_WRAP_ALGORITHMS:
            raise ContainerError(f"Unsupported key-wrap algorithm {wrap.get('alg')!r}.")
        if not isinstance(wrap.get("wrappedFileKey"), str) or not wrap["wrappedFileKey"]:
            raise ContainerError("A v2 container must carry a wrapped file key.")
        if not isinstance(wrap.get("keyId"), str) or not wrap["keyId"]:
            raise ContainerError("A v2 container must name the key it was wrapped to.")

    return {
        "version": version,
        "suite": suite,
        "algorithm": SUITE_NAMES[suite],
        "header": header,
        "header_bytes": bytes(data[PREAMBLE_BYTES:header_end]),
        "header_length": header_len,
        "nonce_offset": header_end,
        "ciphertext_offset": header_end + NONCE_BYTES,
        "key_wrap": header.get("kw") if version == CONTAINER_V2_HYBRID else None,
    }


def encrypt_stream(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    key: EphemeralKey,
    header: dict[str, Any],
    nonce: bytes | None = None,
    suite: int = SUITE_AES_256_GCM,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Write one container from ``source`` to ``destination``.

    Streamed in fixed-size chunks, so peak memory is the chunk size rather than the video
    size. Returns non-secret facts about what was written.

    ``nonce`` exists for the format conformance vector only. Production never passes it:
    every call must draw a fresh nonce, and reusing one under the same key is the single
    catastrophic misuse of GCM.
    """
    if suite != SUITE_AES_256_GCM:
        raise ContainerError(f"Unsupported cipher suite {suite}.")

    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if nonce is None:
        nonce = secrets.token_bytes(NONCE_BYTES)
    elif len(nonce) != NONCE_BYTES:
        raise ContainerError(f"Nonce must be {NONCE_BYTES} bytes, got {len(nonce)}.")

    # The framing version comes from the header, not a module constant: one function
    # writes both container versions, and the two must never disagree about which one this
    # is - the version byte is what a reader dispatches on.
    version = int(header.get("v", CONTAINER_V1_SYMMETRIC))
    if version not in SUPPORTED_CONTAINER_VERSIONS:
        raise ContainerError(f"Refusing to write unknown container version {version}.")

    header_bytes = canonical_header(header)

    destination.write(MAGIC)
    destination.write(bytes([version, suite]))
    destination.write(len(header_bytes).to_bytes(2, "big"))
    destination.write(header_bytes)
    destination.write(nonce)

    encryptor = Cipher(algorithms.AES(key.bytes), modes.GCM(nonce)).encryptor()
    # Must precede any update(): the header authenticates as associated data, not payload.
    encryptor.authenticate_additional_data(header_bytes)

    plaintext_bytes = 0
    ciphertext_bytes = 0
    while True:
        chunk = source.read(chunk_bytes)
        if not chunk:
            break
        plaintext_bytes += len(chunk)
        sealed = encryptor.update(chunk)
        if sealed:
            destination.write(sealed)
            ciphertext_bytes += len(sealed)

    sealed = encryptor.finalize()
    if sealed:
        destination.write(sealed)
        ciphertext_bytes += len(sealed)

    destination.write(encryptor.tag)

    return {
        "algorithm": SUITE_NAMES[suite],
        "version": version,
        "suite": suite,
        "plaintext_bytes": plaintext_bytes,
        "ciphertext_bytes": ciphertext_bytes,
        "container_bytes": (
            PREAMBLE_BYTES + len(header_bytes) + NONCE_BYTES + ciphertext_bytes + TAG_BYTES
        ),
        "header": json.loads(header_bytes.decode("utf-8")),
    }


def decrypt_container(data: bytes, key: bytes | EphemeralKey) -> tuple[bytes, dict[str, Any]]:
    """Decrypt a whole container held in memory. Returns ``(plaintext, header)``.

    The reference implementation of the format, and what the tests check the browser module
    against. Production upload paths never call it - the worker only ever encrypts. Raises
    ``ContainerError`` on a wrong key, a modified header, modified ciphertext or a modified
    tag; the message deliberately does not distinguish between them.
    """
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    raw = key.bytes if isinstance(key, EphemeralKey) else key
    if len(raw) != KEY_BYTES:
        raise ArtifactError(f"Key must be {KEY_BYTES} bytes, got {len(raw)}.")

    parsed = parse_container_prefix(data)
    start = parsed["ciphertext_offset"]
    if len(data) < start + TAG_BYTES:
        raise ContainerError("Container is truncated: no room for an authentication tag.")

    nonce = bytes(data[parsed["nonce_offset"] : start])
    body = bytes(data[start : len(data) - TAG_BYTES])
    tag = bytes(data[len(data) - TAG_BYTES :])

    decryptor = Cipher(algorithms.AES(raw), modes.GCM(nonce, tag)).decryptor()
    decryptor.authenticate_additional_data(parsed["header_bytes"])
    try:
        plaintext = decryptor.update(body) + decryptor.finalize()
    except InvalidTag as error:
        raise ContainerError(
            "Unable to decrypt this artefact. The key is wrong or the data was modified."
        ) from error

    return plaintext, parsed["header"]


# --------------------------------------------------------------------------------------
# Protectors
# --------------------------------------------------------------------------------------


class ArtifactProtector:
    """Turns a GeneratedArtifact into a ProtectedArtifact."""

    encrypts = False
    mode = DEFAULT_PRIVACY_MODE

    def __init__(self) -> None:
        #: Cumulative wall clock spent protecting artefacts, for the caller's own timing
        #: line. Zero for modes that do no work.
        self.total_seconds = 0.0

    def protect(self, artifact: GeneratedArtifact) -> ProtectedArtifact:
        raise NotImplementedError

    def destroy(self) -> None:
        """Release any key material. Always safe, always idempotent."""


class PassthroughProtector(ArtifactProtector):
    """Standard mode: hand the artefact on untouched.

    Deliberately a real object rather than a ``None`` check at the call site, so the
    plaintext path and the ciphertext path run through identical code and cannot drift.
    """

    encrypts = False
    mode = "standard"

    def __init__(self) -> None:
        super().__init__()

    def protect(self, artifact: GeneratedArtifact) -> ProtectedArtifact:
        return ProtectedArtifact(
            path=artifact.path,
            content_type=artifact.mime_type,
            encrypted=False,
            generation_id=artifact.generation_id,
            filename=artifact.name,
            size=artifact.size,
            metadata={"privacyMode": self.mode, "encrypted": False},
        )


class ConfidentialProtector(ArtifactProtector):
    """Confidential mode: encrypt with the caller's key, then destroy the plaintext.

    The key arrives with the request and leaves with it. It is not written to disk, not
    included in the returned metadata, not logged, and not retained after ``protect()``
    returns unless the caller keeps its own reference.
    """

    encrypts = True
    mode = "confidential"

    def __init__(
        self,
        key: EphemeralKey,
        *,
        kdf: dict[str, Any] | None = None,
        key_id: str | None = None,
        on_warning: Callable[[str], None] | None = None,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        super().__init__()
        if not isinstance(key, EphemeralKey):
            raise ArtifactError("ConfidentialProtector needs an EphemeralKey.")
        self._key = key
        self.kdf = kdf
        self.key_id = key_id
        self.chunk_bytes = chunk_bytes
        self._on_warning = on_warning

    def _warn(self, message: str) -> None:
        if self._on_warning:
            self._on_warning(message)

    def destroy(self) -> None:
        self._key.destroy()

    def protect(self, artifact: GeneratedArtifact) -> ProtectedArtifact:
        began = time.monotonic()
        plaintext_size = artifact.size

        header = build_header(
            generation_id=artifact.generation_id,
            content_type=artifact.mime_type,
            plaintext_bytes=plaintext_size,
            kdf=self.kdf,
            key_id=self.key_id,
        )

        scratch = make_scratch_dir("cg-enc-")
        # A fresh name every time. Two runs of the same generation - a RunPod retry, say -
        # must never collide on a path, and a predictable name is one more thing a
        # co-tenant process could race.
        target = os.path.join(scratch, f"{uuid.uuid4().hex}{ENCRYPTED_EXTENSION}")

        try:
            with open(artifact.path, "rb") as source, open(target, "wb") as destination:
                facts = encrypt_stream(
                    source,
                    destination,
                    key=self._key,
                    header=header,
                    chunk_bytes=self.chunk_bytes,
                )
        except Exception:
            # Fail closed. Nothing half-encrypted is allowed to survive, and the caller
            # gets an exception rather than a plaintext artefact it might upload. The
            # plaintext is left for the caller's own cleanup, which runs in a finally.
            shutil.rmtree(scratch, ignore_errors=True)
            raise

        # Only now, with a complete container on disk, is the plaintext expendable.
        removed = shred(artifact.path, on_warning=self._warn)
        if not removed:
            self._warn(
                f"plaintext for {artifact.name} could not be deleted; it will go when the "
                "container is destroyed"
            )

        elapsed = time.monotonic() - began
        self.total_seconds += elapsed
        return ProtectedArtifact(
            path=target,
            content_type=ENCRYPTED_CONTENT_TYPE,
            encrypted=True,
            generation_id=artifact.generation_id,
            filename=artifact.name + ENCRYPTED_EXTENSION,
            size=facts["container_bytes"],
            plaintext_removed=removed,
            encryption_seconds=elapsed,
            scratch_dir=scratch,
            metadata={
                "privacyMode": self.mode,
                "encrypted": True,
                "algorithm": facts["algorithm"],
                "encryptionVersion": facts["version"],
                "contentType": ENCRYPTED_CONTENT_TYPE,
                "originalContentType": artifact.mime_type,
                "plaintextBytes": plaintext_size,
                # Repeated from the container so a client can prompt for the passphrase
                # before it downloads anything. Both copies are non-secret, and the one
                # inside the container is the authenticated one.
                **({"kdf": self.kdf} if self.kdf else {}),
                **({"keyId": self.key_id} if self.key_id else {}),
            },
        )


class ConfidentialV2Protector(ArtifactProtector):
    """Hybrid mode: a random file key per artefact, wrapped to the caller's public key.

    The difference from v1 is the whole reason this class exists. v1 was handed a symmetric
    key that could decrypt the result, which meant that key travelled through the job queue
    and lived in whatever the queue retains. Here the worker is handed a key that can only
    encrypt. It generates the file-encryption key itself, uses it once, wraps it to the
    public key and drops it. Nothing that leaves this process can open the artefact.

    Lifecycle of the file-encryption key, in full:

        secrets.token_bytes(32) -> AES-256-GCM encrypt -> RSA-OAEP wrap -> destroy()

    It is never logged, never written to disk, never returned, and never persisted in any
    form other than the wrapped one inside the container's authenticated header.
    """

    encrypts = True
    mode = "confidential"
    crypto_version = CONTAINER_V2_HYBRID

    def __init__(
        self,
        public_key: PublicEncryptionKey,
        *,
        on_warning: Callable[[str], None] | None = None,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        super().__init__()
        if not isinstance(public_key, PublicEncryptionKey):
            raise ArtifactError("ConfidentialV2Protector needs a PublicEncryptionKey.")
        self.public_key = public_key
        self.key_id = public_key.key_id
        self.chunk_bytes = chunk_bytes
        self._on_warning = on_warning

    def _warn(self, message: str) -> None:
        if self._on_warning:
            self._on_warning(message)

    def destroy(self) -> None:
        """Nothing to release: this protector never holds a decryption-capable secret."""

    def protect(self, artifact: GeneratedArtifact) -> ProtectedArtifact:
        began = time.monotonic()
        plaintext_size = artifact.size

        # Fresh for every artefact, and every retry of every artefact. Never derived from
        # the generation id, the seed, the job id or a timestamp - a deterministic file key
        # or nonce under a reused key is the one catastrophic misuse of GCM, and a
        # deterministic object path makes that an easy mistake to reach for.
        file_key = EphemeralKey(secrets.token_bytes(KEY_BYTES))

        try:
            # Wrap before encrypting: the wrapped key belongs in the header, and the header
            # is the associated data, so it has to exist before the first byte is sealed.
            # It also means a wrapping failure costs nothing - no ciphertext was written
            # and the plaintext is still where the caller left it.
            try:
                wrapped = self.public_key.wrap(file_key)
            except ArtifactError:
                raise
            except Exception as error:
                raise ArtifactError(
                    f"Wrapping the file key to public key {self.key_id} failed: "
                    f"{type(error).__name__}"
                ) from error

            header = build_v2_header(
                generation_id=artifact.generation_id,
                content_type=artifact.mime_type,
                plaintext_bytes=plaintext_size,
                wrapped_file_key=wrapped,
                key_id=self.key_id,
                key_wrap_algorithm=self.public_key.algorithm,
                privacy_mode=self.mode,
            )

            scratch = make_scratch_dir("cg-enc-")
            target = os.path.join(scratch, f"{uuid.uuid4().hex}{ENCRYPTED_EXTENSION}")

            try:
                with open(artifact.path, "rb") as source, open(target, "wb") as destination:
                    facts = encrypt_stream(
                        source,
                        destination,
                        key=file_key,
                        header=header,
                        chunk_bytes=self.chunk_bytes,
                    )
            except Exception:
                # Fail closed: nothing half-encrypted survives, and the caller gets an
                # exception rather than a plaintext artefact it might upload.
                shutil.rmtree(scratch, ignore_errors=True)
                raise
        finally:
            # The file key has done its only two jobs. Releasing it here means it is gone
            # before the upload even starts, on the failure paths as well as the happy one.
            file_key.destroy()

        removed = shred(artifact.path, on_warning=self._warn)
        if not removed:
            self._warn(
                f"plaintext for {artifact.name} could not be deleted; it will go when the "
                "container is destroyed"
            )

        elapsed = time.monotonic() - began
        self.total_seconds += elapsed
        return ProtectedArtifact(
            path=target,
            content_type=ENCRYPTED_CONTENT_TYPE,
            encrypted=True,
            generation_id=artifact.generation_id,
            filename=artifact.name + ENCRYPTED_EXTENSION,
            size=facts["container_bytes"],
            plaintext_removed=removed,
            encryption_seconds=elapsed,
            scratch_dir=scratch,
            metadata={
                "privacyMode": self.mode,
                "encrypted": True,
                "cryptoVersion": CONTAINER_V2_HYBRID,
                "algorithm": facts["algorithm"],
                "encryptionVersion": CONTAINER_V2_HYBRID,
                "keyWrapAlgorithm": self.public_key.algorithm,
                "keyId": self.key_id,
                "contentType": ENCRYPTED_CONTENT_TYPE,
                "originalContentType": artifact.mime_type,
                "plaintextBytes": plaintext_size,
            },
        )


def unwrap_file_key(wrapped: bytes, private_key) -> "EphemeralKey":
    """Recover a file key from its wrapped form. Reference implementation, for tests.

    Production never calls this: the worker only ever wraps, and unwrapping happens in the
    browser with a private key this codebase never sees.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    try:
        raw = private_key.decrypt(
            wrapped,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    except Exception as error:
        raise ContainerError(
            "Unable to unwrap the file key. The private key is wrong or the wrapped key "
            "was modified."
        ) from error
    return EphemeralKey(raw)


def decrypt_v2_container(data: bytes, private_key) -> tuple[bytes, dict[str, Any]]:
    """Unwrap the file key with a private key, then decrypt. Returns ``(plaintext, header)``.

    The reference implementation of the v2 read path and what the browser module is checked
    against. Every failure - wrong private key, modified wrapped key, modified header,
    modified ciphertext, modified tag - raises the same ``ContainerError``.
    """
    parsed = parse_container_prefix(data)
    if parsed["version"] != CONTAINER_V2_HYBRID:
        raise ContainerError(
            f"Not a hybrid container: version {parsed['version']}. Use decrypt_container "
            "for v1 artefacts."
        )

    wrap = parsed["key_wrap"]
    padded = wrap["wrappedFileKey"] + "=" * (-len(wrap["wrappedFileKey"]) % 4)
    try:
        wrapped = base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError) as error:
        raise ContainerError(f"Wrapped file key is not valid base64url: {error}") from error

    file_key = unwrap_file_key(wrapped, private_key)
    try:
        return decrypt_container(data, file_key)
    finally:
        file_key.destroy()


def build_protector(
    mode_name: str | None,
    encryption: dict[str, Any] | None = None,
    *,
    on_warning: Callable[[str], None] | None = None,
) -> ArtifactProtector:
    """Resolve a privacy mode plus its encryption block into a protector.

    Fails closed in both directions: a confidential request without usable key material is
    an error, and a standard request that carries key material is an error too. Silently
    ignoring an ``encryption`` block would leave the caller believing their artefact was
    encrypted when it was not, which is worse than a rejection.
    """
    spec = privacy_mode(mode_name)
    encryption = encryption or {}

    if not spec.encrypts:
        if encryption:
            raise ArtifactError(
                f"privacyMode {spec.name!r} does not encrypt, but an encryption block was "
                "supplied. Use privacyMode 'confidential' or drop the block."
            )
        return PassthroughProtector()

    if not encryption:
        raise ArtifactError("privacyMode 'confidential' requires an encryption block.")

    algorithm = str(encryption.get("algorithm") or SUITE_NAMES[SUITE_AES_256_GCM]).strip()
    if algorithm not in SUITE_IDS:
        raise ArtifactError(
            f"Unsupported encryption algorithm {algorithm!r}. Supported: "
            f"{', '.join(sorted(SUITE_IDS))}."
        )

    version = int(encryption.get("version") or CONTAINER_V2_HYBRID)

    # A symmetric key in the request is the exact property v2 exists to remove, so it is
    # refused outright rather than honoured. Old artefacts stay readable; new ones cannot
    # be created this way. Naming the replacement in the message is the difference between
    # a migration and an outage.
    if version == CONTAINER_V1_SYMMETRIC or encryption.get("key"):
        raise ArtifactError(
            "Confidential generation v1 (a symmetric key in the request) is no longer "
            "accepted: that key would travel through the job queue and could later decrypt "
            "the stored video. Send encryption.version 2 with a publicKey instead. "
            "Existing v1 artefacts remain decryptable."
        )

    if version != CONTAINER_V2_HYBRID:
        raise ArtifactError(
            f"Unsupported encryption.version {version}. This build creates version "
            f"{CONTAINER_V2_HYBRID}."
        )

    wrap_algorithm = str(
        encryption.get("keyWrapAlgorithm")
        or encryption.get("key_wrap_algorithm")
        or KEY_WRAP_RSA_OAEP_256
    ).strip()
    if wrap_algorithm not in SUPPORTED_KEY_WRAP_ALGORITHMS:
        raise ArtifactError(
            f"Unsupported keyWrapAlgorithm {wrap_algorithm!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_KEY_WRAP_ALGORITHMS))}."
        )

    public_key = decode_public_key(
        encryption.get("publicKey") or encryption.get("public_key") or ""
    )

    # A supplied key id has to agree with the key it claims to name. The id is derived from
    # the key's own bytes, so disagreement means the caller is confused about which key
    # they hold - and an artefact filed under the wrong id is one the client will not think
    # to try its private key against.
    claimed = encryption.get("keyId") or encryption.get("key_id")
    if claimed and str(claimed) != public_key.key_id:
        raise ArtifactError(
            f"encryption.keyId does not match the supplied public key (expected "
            f"{public_key.key_id})."
        )

    return ConfidentialV2Protector(public_key, on_warning=on_warning)


# --------------------------------------------------------------------------------------
# Redaction
#
# Used for logs and for anything echoed back from an upstream service. The rule is
# structural, not a regex over a serialized blob: walk the value and decide per field.
# --------------------------------------------------------------------------------------

#: Field names whose value is secret wherever it appears.
#:
#: Kept identical to SECRET_FIELDS in worker.js - tests/test_redaction.py asserts that,
#: because a name redacted on one side of the system and not the other is exactly the kind
#: of gap nobody notices until it is in a log.
#:
#: The Access *client id* is in here alongside the secret. On its own it is closer to a
#: username than a credential, but it identifies which service token is in play, and the
#: one place this codebase deliberately prints it (log_callback_configuration) prints a
#: truncated form on purpose. A generic structural dump should not undo that.
SECRET_FIELDS = frozenset(
    {
        "encryptionkey",
        "encryption_key",
        "passphrase",
        "password",
        "secret",
        "authorization",
        "cf-access-client-id",
        "cf_access_client_id",
        "cloudflare_access_client_id",
        "cf-access-client-secret",
        "cf_access_client_secret",
        "cloudflare_access_client_secret",
        "x-api-key",
        "api_key",
        "apikey",
        "token",
        "job_token_secret",
        "runpod_api_key",
        "dek",
        "data_key",
        "datakey",
        "wrapped_key",
        "wrappedkey",
        "h3_key_wrap_key",
        # Crypto v2. A private key, the key that protects it, and the per-video file key
        # are the three values whose exposure would undo the whole design.
        "privatekey",
        "private_key",
        "privateencryptionkey",
        "private_encryption_key",
        "encryptedprivatekey",
        "encrypted_private_key",
        "keyencryptionkey",
        "key_encryption_key",
        "kek",
        "fileencryptionkey",
        "file_encryption_key",
        "fek",
        "decryptionkey",
        "decryption_key",
        "derivedkey",
        "derived_key",
        "aeskey",
        "aes_key",
        "symmetrickey",
        "symmetric_key",
        "secretkey",
        "secret_key",
    }
)

#: ``key`` is only a secret in a cryptographic context. Everywhere else in this codebase it
#: is an R2 object key, which is not sensitive and is exactly what a support question needs.
CRYPTO_PARENTS = frozenset({"encryption", "crypto", "kdf", "confidential", "privacy"})

#: Never redacted: an opaque client-chosen label whose whole purpose is to be visible.
#: Never redacted. Each is public by construction, and scrubbing them would destroy the
#: fields an operator needs to answer "which key was this encrypted to?" without helping
#: anyone decrypt anything. A public key encrypts and cannot decrypt; a wrapped file key is
#: useless without the private key; a KDF salt is not a secret.
NEVER_REDACT = frozenset(
    {
        "keyid",
        "key_id",
        "r2_key",
        "storage_key",
        "object_key",
        "publickey",
        "public_key",
        "publickeyalgorithm",
        "public_key_algorithm",
        "keywrapalgorithm",
        "key_wrap_algorithm",
        "wrappedfilekey",
        "wrapped_file_key",
        "cryptoversion",
        "crypto_version",
        "salt",
    }
)

REDACTED = "[redacted]"


def redact(value: Any, *, parent: str = "", depth: int = 0) -> Any:
    """Return a copy of ``value`` with secret fields replaced by ``[redacted]``.

    Structure is preserved, so a redacted payload is still worth reading.
    """
    if depth > 12:
        return "[omitted: nesting too deep]"

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for name, child in value.items():
            lowered = str(name).strip().lower()
            if lowered in NEVER_REDACT:
                out[name] = child
            elif lowered in SECRET_FIELDS or (
                lowered == "key" and parent.strip().lower() in CRYPTO_PARENTS
            ):
                out[name] = REDACTED
            else:
                out[name] = redact(child, parent=str(name), depth=depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        return [redact(item, parent=parent, depth=depth + 1) for item in value]

    if isinstance(value, EphemeralKey):
        return REDACTED

    return value


def prompt_digest(prompt: str) -> str:
    """A correlation handle for a prompt, for logs that must not contain the prompt.

    This is *not* a way to keep a prompt secret. Prompts are low entropy; anyone holding a
    candidate list can confirm a match by hashing it. It exists so two log lines about the
    same generation can be tied together without writing the text down.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
