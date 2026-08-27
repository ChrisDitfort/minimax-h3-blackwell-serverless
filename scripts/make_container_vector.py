#!/usr/bin/env python3
"""Regenerate the confidential-container conformance vector.

    python scripts/make_container_vector.py > tests/vectors/confidential_container.json

The vector pins the exact bytes of one container built from fixed inputs. Three separate
implementations read it:

    artifacts.py                        the producer - must reproduce these bytes exactly
    worker.js::parseContainerPrefix     must parse the framing and header
    client/confidential-generation.js   must decrypt it back to the plaintext

That is the only thing standing between the three and a silent format drift. A change to
the layout, the header's canonical serialization or the AAD binding will fail at least one
of those suites, which is the point: the failure should happen in CI, not the first time a
user cannot open a video they paid for.

Everything here is fixed on purpose - key, nonce, timestamp - so the output is byte-stable.
Nothing in this file resembles production: production draws both key and nonce from a
CSPRNG for every artefact, and reusing a nonce under one key is the single fatal misuse of
AES-GCM.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import artifacts  # noqa: E402

# Fixed inputs. Chosen to be obviously synthetic and easy to type into another language.
NONCE = bytes(range(100, 112))
PLAINTEXT = b"\x00\x00\x00\x18ftypmp42" + b"conformance vector payload" * 4
GENERATION_ID = "11111111-2222-3333-4444-555555555555"
CREATED_AT = "2026-01-01T00:00:00Z"

# The container's key is *derived*, not invented, so that one vector proves the entire user
# journey across all three implementations: this passphrase, through this KDF, produces the
# key that opens this container. A browser that derives a different key fails the same test
# that a handler emitting different bytes fails.
PASSPHRASE = "correct horse battery staple"
KDF_SALT = b"conformance-salt"
KDF_ITERATIONS = 600000
KDF = {
    "name": "pbkdf2-sha256",
    "salt": base64.urlsafe_b64encode(KDF_SALT).decode().rstrip("="),
    "parameters": {"iterations": KDF_ITERATIONS},
}
KEY = hashlib.pbkdf2_hmac("sha256", PASSPHRASE.encode("utf-8"), KDF_SALT, KDF_ITERATIONS, 32)


#: A fixed RSA-3072 key pair for the v2 vector. Generated once and pinned here so the
#: vector is byte-stable across regenerations; a fresh pair each time would change the
#: wrapped key (RSA-OAEP is randomised) and make every regeneration look like a change.
#: Obviously synthetic, obviously not production - it is committed in a test fixture.
V2_PRIVATE_KEY_PEM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "vectors", "v2_test_key.pem"
)


def load_or_create_v2_key():
    """The v2 vector's key pair, created on first run and reused thereafter."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    path = os.path.normpath(V2_PRIVATE_KEY_PEM_PATH)
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return serialization.load_pem_private_key(handle.read(), password=None)

    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    with open(path, "wb") as handle:
        handle.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    return key


def build_v2_vector() -> dict:
    """One hybrid container, with everything needed to open it from either language."""
    from cryptography.hazmat.primitives import serialization

    private = load_or_create_v2_key()
    spki = private.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pkcs8 = private.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = artifacts.PublicEncryptionKey(spki)

    # The wrapped key is randomised by OAEP and the nonce is random, so this container is
    # not byte-stable - only its *structure* and its decryptability are. That is the right
    # thing to pin for v2: a fixed ciphertext would require a fixed OAEP seed, which no
    # sane API exposes and which nothing should ever do in production.
    file_key = artifacts.EphemeralKey(bytes(range(32)))
    wrapped = public.wrap(file_key)
    header = artifacts.build_v2_header(
        generation_id=GENERATION_ID,
        content_type="video/mp4",
        plaintext_bytes=len(PLAINTEXT),
        wrapped_file_key=wrapped,
        key_id=public.key_id,
        created_at=CREATED_AT,
    )

    destination = io.BytesIO()
    artifacts.encrypt_stream(
        io.BytesIO(PLAINTEXT), destination, key=file_key, header=header, nonce=NONCE
    )
    container = destination.getvalue()

    recovered, _ = artifacts.decrypt_v2_container(container, private)
    assert recovered == PLAINTEXT, "v2 vector does not round-trip"

    return {
        "container_hex": container.hex(),
        "container_bytes": len(container),
        "generation_id": GENERATION_ID,
        "header": header,
        "key_id": public.key_id,
        "public_key_spki_b64": base64.urlsafe_b64encode(spki).decode().rstrip("="),
        "private_key_pkcs8_b64": base64.urlsafe_b64encode(pkcs8).decode().rstrip("="),
        "file_key_hex": file_key.bytes.hex(),
        "plaintext_hex": PLAINTEXT.hex(),
        "key_wrap_algorithm": artifacts.KEY_WRAP_RSA_OAEP_256,
        "modulus_bits": public.modulus_bits,
    }


def main() -> int:
    # Force LF regardless of platform. .gitattributes pins *.json to LF, so a CRLF write on
    # Windows would show as a diff on every regeneration and hide real changes.
    sys.stdout.reconfigure(newline="\n")

    header = artifacts.build_header(
        generation_id=GENERATION_ID,
        content_type="video/mp4",
        plaintext_bytes=len(PLAINTEXT),
        kdf=KDF,
        key_id="demo-key-1",
        created_at=CREATED_AT,
    )

    destination = io.BytesIO()
    facts = artifacts.encrypt_stream(
        io.BytesIO(PLAINTEXT),
        destination,
        key=artifacts.EphemeralKey(KEY),
        header=header,
        nonce=NONCE,
    )
    container = destination.getvalue()

    # Prove the vector is self-consistent before publishing it.
    recovered, parsed_header = artifacts.decrypt_container(container, KEY)
    assert recovered == PLAINTEXT, "vector does not round-trip"
    assert parsed_header == header, "header does not survive the round trip"

    print(
        json.dumps(
            {
                "_comment": (
                    "Conformance vector for the CGEN confidential container. Regenerate "
                    "with scripts/make_container_vector.py. The key and nonce are fixed "
                    "for reproducibility and must never appear in production."
                ),
                "key_hex": KEY.hex(),
                "kdf": {
                    **KDF,
                    "passphrase": PASSPHRASE,
                    "salt_utf8": KDF_SALT.decode("ascii"),
                    "derived_key_hex": KEY.hex(),
                },
                "nonce_hex": NONCE.hex(),
                "plaintext_hex": PLAINTEXT.hex(),
                "generation_id": GENERATION_ID,
                "created_at": CREATED_AT,
                "header": header,
                "header_length": len(artifacts.canonical_header(header)),
                "container_hex": container.hex(),
                "container_bytes": len(container),
                "offsets": {
                    "preamble": artifacts.PREAMBLE_BYTES,
                    "nonce": artifacts.PREAMBLE_BYTES + len(artifacts.canonical_header(header)),
                    "ciphertext": (
                        artifacts.PREAMBLE_BYTES
                        + len(artifacts.canonical_header(header))
                        + artifacts.NONCE_BYTES
                    ),
                },
                "container_base64": base64.b64encode(container).decode("ascii"),
                # The hybrid vector. Its ciphertext is not byte-stable (OAEP is
                # randomised), so what is pinned is the structure and the fact that this
                # private key opens it from any language.
                "v2": build_v2_vector(),
                "facts": {
                    "algorithm": facts["algorithm"],
                    "version": facts["version"],
                    "suite": facts["suite"],
                    "plaintext_bytes": facts["plaintext_bytes"],
                    "ciphertext_bytes": facts["ciphertext_bytes"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
