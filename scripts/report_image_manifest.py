#!/usr/bin/env python3
"""Summarize OCI metadata fetched by crane without downloading image layers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_ENTRYPOINT = ["python3", "/opt/serverless/bootstrap.py"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--platform-digest", required=True)
    parser.add_argument("--registry-ref", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if not OCI_DIGEST_RE.fullmatch(args.digest):
        parser.error("--digest must be an immutable sha256 OCI digest")
    if not OCI_DIGEST_RE.fullmatch(args.platform_digest):
        parser.error("--platform-digest must be an immutable sha256 OCI digest")
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
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        print("Image metadata verification failed: expected linux/amd64")
        return 1
    if entrypoint != EXPECTED_ENTRYPOINT or command:
        print(
            "Image metadata verification failed: unexpected entrypoint/cmd "
            f"entrypoint={entrypoint!r} cmd={command!r}"
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
