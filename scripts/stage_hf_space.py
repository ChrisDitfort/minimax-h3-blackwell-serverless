#!/usr/bin/env python3
"""Create the minimal, allowlisted source tree published to the private Docker Space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath


EXPECTED_SPACE_REPOSITORY = "CDitfort/privora-h3-runpod-worker"
MAX_SOURCE_BYTES = 10 * 1024 * 1024
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BUILD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

STATIC_MAPPINGS = {
    "Dockerfile.hf-space": "Dockerfile",
    ".dockerignore.hf-space": ".dockerignore",
    "space/README.md": "README.md",
    "handler.py": "handler.py",
    "artifacts.py": "artifacts.py",
    "bootstrap.py": "bootstrap.py",
    "cached_models.py": "cached_models.py",
    "models/h3-cache-manifest.json": "models/h3-cache-manifest.json",
    "comfy_custom_nodes/h3_parallel_boot.py": "comfy_custom_nodes/h3_parallel_boot.py",
    "scripts/verify_ref2va_contract.py": "scripts/verify_ref2va_contract.py",
    "scripts/guard_no_model_weights.py": "scripts/guard_no_model_weights.py",
}
PACKAGE_DIRECTORIES = ("privora", "h3_parallel")
FORBIDDEN_NAMES = {
    ".env",
    ".dev.vars",
    "models.tsv",
    "build_model_layer.py",
    "hf_token",
}
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}


class StagingError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(path: Path, relative: str) -> None:
    if path.is_symlink():
        raise StagingError(f"allowlisted source must not be a symlink: {relative}")
    if not path.is_file():
        raise StagingError(f"allowlisted source is missing: {relative}")
    if path.name.lower() in FORBIDDEN_NAMES:
        raise StagingError(f"forbidden source name: {relative}")
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        raise StagingError(f"model-weight extension is forbidden: {relative}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise StagingError(f"allowlisted source unexpectedly exceeds 10 MiB: {relative}")


def approved_mappings(source_root: Path) -> dict[str, str]:
    mappings = dict(STATIC_MAPPINGS)
    for package in PACKAGE_DIRECTORIES:
        package_root = source_root / package
        if not package_root.is_dir() or package_root.is_symlink():
            raise StagingError(f"required package directory is missing or linked: {package}")
        for path in sorted(package_root.rglob("*.py")):
            relative_parts = path.relative_to(source_root).parts
            parents_inside_source = [source_root.joinpath(*relative_parts[:index])
                                     for index in range(1, len(relative_parts))]
            if any(parent.is_symlink() for parent in parents_inside_source):
                raise StagingError(f"package source traverses a symlink: {path}")
            relative = path.relative_to(source_root).as_posix()
            mappings[relative] = relative
    return mappings


def _safe_destination(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts):
        raise StagingError(f"unsafe staging destination: {relative}")
    return root.joinpath(*posix.parts)


def stage(
    source_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    build_id: str,
    space_repository: str,
) -> dict:
    source_root = source_root.resolve(strict=True)
    if not SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise StagingError("source commit must be a full lowercase Git SHA")
    if not BUILD_ID_RE.fullmatch(build_id):
        raise StagingError("build id contains unsupported characters")
    if space_repository != EXPECTED_SPACE_REPOSITORY:
        raise StagingError("Space repository does not match the approved destination")
    if os.path.lexists(output_root):
        raise StagingError("staging output already exists; refusing to merge or overwrite it")
    output_root.mkdir(parents=True)

    mappings = approved_mappings(source_root)
    copied: list[str] = []
    try:
        for source_relative, destination_relative in sorted(mappings.items()):
            source = source_root.joinpath(*PurePosixPath(source_relative).parts)
            _validate_source(source, source_relative)
            destination = _safe_destination(output_root, destination_relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(destination_relative)

        identity = {
            "schemaVersion": 1,
            "sourceCommit": source_commit,
            "buildId": build_id,
            "spaceRepository": space_repository,
        }
        identity_path = output_root / "space-build-identity.json"
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        copied.append("space-build-identity.json")

        file_records = []
        for relative in sorted(copied):
            path = _safe_destination(output_root, relative)
            file_records.append({
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        publication = {
            "schemaVersion": 1,
            "spaceRepository": space_repository,
            "sourceCommit": source_commit,
            "buildId": build_id,
            "fileCount": len(file_records) + 1,
            "files": file_records,
        }
        publication_path = output_root / "space-publication-manifest.json"
        publication_path.write_text(
            json.dumps(publication, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        copied.append("space-publication-manifest.json")
        return publication
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--space-repository", default=EXPECTED_SPACE_REPOSITORY)
    args = parser.parse_args(argv)
    try:
        publication = stage(
            args.source_root,
            args.output,
            source_commit=args.source_commit,
            build_id=args.build_id,
            space_repository=args.space_repository,
        )
    except (OSError, StagingError, ValueError, json.JSONDecodeError) as error:
        print(f"Space staging failed: {error}")
        return 1
    total = sum(record["bytes"] for record in publication["files"])
    print(
        f"Space staging OK: approved_files={publication['fileCount']} "
        f"payload_bytes={total} model_weights=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
