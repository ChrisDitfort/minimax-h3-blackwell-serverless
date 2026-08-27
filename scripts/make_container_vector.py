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
