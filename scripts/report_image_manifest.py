#!/usr/bin/env python3
"""Summarize OCI metadata fetched by crane without downloading image layers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_ENTRYPOINT = ["python3", "/opt/serverless/bootstrap.py"]
EXPECTED_ENVIRONMENT = {
    "H3_MODEL_REPO": "CDitfort/privora-minimax-h3-models",
    "H3_MODEL_REVISION": "ecb69a4211d74b5798398021003bccde02d63757",
    "H3_MODEL_MANIFEST_VERSION": "multimodal-4-hf-cache-v1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "COMFYUI_H3_COMMIT": "dec5d945",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--platform-digest", required=True)
    parser.add_argument("--registry-ref", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if not OCI_DIGEST_RE.fullmatch(args.digest):
        parser.error("--digest must be an immutable sha256 OCI digest")
    if not OCI_DIGEST_RE.fullmatch(args.platform_digest):
        parser.error("--platform-digest must be an immutable sha256 OCI digest")
    if not SOURCE_COMMIT_RE.fullmatch(args.source_commit):
        parser.error("--source-commit must be a full lowercase Git SHA")
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        print("Image metadata verification failed: manifest has no layers")
        return 1
    compressed_bytes = sum(int(layer["size"]) for layer in layers)
    runtime = config.get("config") or {}
    entrypoint = runtime.get("Entrypoint") or []
    command = runtime.get("Cmd") or []
    environment = dict(
        item.split("=", 1)
        for item in (runtime.get("Env") or [])
        if isinstance(item, str) and "=" in item
    )
    labels = runtime.get("Labels") or {}
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        print("Image metadata verification failed: expected linux/amd64")
        return 1
    if entrypoint != EXPECTED_ENTRYPOINT or command:
        print(
            "Image metadata verification failed: unexpected entrypoint/cmd "
            f"entrypoint={entrypoint!r} cmd={command!r}"
        )
        return 1
    wrong_environment = {
        key: environment.get(key)
        for key, expected in EXPECTED_ENVIRONMENT.items()
        if environment.get(key) != expected
    }
    expected_labels = {
        "org.opencontainers.image.source":
            "https://github.com/ChrisDitfort/minimax-h3-blackwell-serverless",
        "org.opencontainers.image.revision": args.source_commit,
        "org.opencontainers.image.version": args.image_tag,
        "ai.privora.build.id": args.build_id,
        "ai.privora.model.repository": EXPECTED_ENVIRONMENT["H3_MODEL_REPO"],
        "ai.privora.model.revision": EXPECTED_ENVIRONMENT["H3_MODEL_REVISION"],
        "ai.privora.model.manifest": EXPECTED_ENVIRONMENT["H3_MODEL_MANIFEST_VERSION"],
    }
    wrong_labels = {
        key: labels.get(key)
        for key, expected in expected_labels.items()
        if labels.get(key) != expected
    }
    if wrong_environment or wrong_labels:
        print(
            "Image metadata verification failed: release identity mismatch "
            f"environment={wrong_environment!r} labels={wrong_labels!r}"
        )
        return 1
    result = {
        "registryRef": args.registry_ref,
        "digest": args.digest,
        "platformDigest": args.platform_digest,
        "compressedBytes": compressed_bytes,
        "layerCount": len(layers),
        "os": config.get("os"),
        "architecture": config.get("architecture"),
        "entrypoint": entrypoint,
        "cmd": command,
        "modelRepository": environment["H3_MODEL_REPO"],
        "modelRevision": environment["H3_MODEL_REVISION"],
        "modelManifestVersion": environment["H3_MODEL_MANIFEST_VERSION"],
    }
    print(json.dumps(result, sort_keys=True))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"image_digest={args.digest}\n")
            handle.write(f"platform_digest={args.platform_digest}\n")
            handle.write(f"compressed_bytes={compressed_bytes}\n")
            handle.write(f"image_architecture={config.get('architecture') or 'unknown'}\n")
            handle.write(f"image_os={config.get('os') or 'unknown'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
