#!/usr/bin/env python3
"""Write the metadata-only release record for a pushed slim GHCR image."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


IMAGE_REPOSITORY = "ghcr.io/chrisditfort/privora-h3-runpod-worker"
MODEL_REPOSITORY = "CDitfort/privora-minimax-h3-models"
MODEL_REVISION = "ecb69a4211d74b5798398021003bccde02d63757"
MODEL_MANIFEST = "multimodal-4-hf-cache-v1"
COMFYUI_REVISION = "dec5d9450a5290bcf63430409ea41018e67f41c3"
SOURCE_BRANCH = "hf-cached-models-slim"
BASE_IMAGE = "ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-blackwell"
BASE_IMAGE_DIGEST = "sha256:4fdcd50e8e5f54f8329933c66e2eac17680cbac82d43c1a74d00465e9413a3e1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
BUILD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def release_document(
    *, source_commit: str, image_tag: str, image_digest: str, platform_digest: str, build_id: str,
    compressed_bytes: int,
) -> dict:
    if not SHA_RE.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if not TAG_RE.fullmatch(image_tag) or image_tag in {"latest", "multimodal-4", "code"}:
        raise ValueError("image tag is invalid or reserved")
    if image_tag.startswith("staging-"):
        raise ValueError("legacy staging tags are reserved")
    if not DIGEST_RE.fullmatch(image_digest):
        raise ValueError("image digest must be sha256:<64 lowercase hex>")
    if not DIGEST_RE.fullmatch(platform_digest):
        raise ValueError("platform image digest must be sha256:<64 lowercase hex>")
    if not BUILD_ID_RE.fullmatch(build_id):
        raise ValueError("build id contains unsupported characters")
    if compressed_bytes <= 0:
        raise ValueError("compressed image size must be positive")
    return {
        "schemaVersion": 1,
        "sourceBranch": SOURCE_BRANCH,
        "sourceCommit": source_commit,
        "buildId": build_id,
        "image": f"{IMAGE_REPOSITORY}:{image_tag}",
        "sourceCommitTag": f"{IMAGE_REPOSITORY}:sha-{source_commit}",
        "imageDigest": image_digest,
        "platformImageDigest": platform_digest,
        "platform": "linux/amd64",
        "compressedBytes": compressed_bytes,
        "entrypoint": ["python3", "/opt/serverless/bootstrap.py"],
        "baseImage": BASE_IMAGE,
        "baseImageDigest": BASE_IMAGE_DIGEST,
        "comfyuiRevision": COMFYUI_REVISION,
        "modelRelease": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "manifest": MODEL_MANIFEST,
        },
        "modelWeightsInImage": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--platform-digest", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--compressed-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        document = release_document(
            source_commit=args.source_commit,
            image_tag=args.image_tag,
            image_digest=args.image_digest,
            platform_digest=args.platform_digest,
            build_id=args.build_id,
            compressed_bytes=args.compressed_bytes,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Release metadata failed: {error}")
        return 1
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
