#!/usr/bin/env python3
"""Fail when a source tree or image filesystem contains model-weight payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}
DEFAULT_MAX_BYTES = 100 * 1024 * 1024


def approved_names(manifest_path: Path) -> set[str]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != 8:
        raise ValueError("model manifest must contain exactly eight assets")
    return {Path(asset["path"]).name for asset in assets}


def scan(roots: list[Path], manifest_path: Path, max_bytes: int) -> list[str]:
    """Return deterministic, human-readable violations without following directory links."""

    exact_names = approved_names(manifest_path)
    violations: list[str] = []
    for root in roots:
        if not os.path.lexists(root):
            continue
        if not root.is_dir():
            raise ValueError(f"scan root is not a directory: {root}")
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                path = Path(directory) / filename
                suffix = path.suffix.lower()
                if filename in exact_names:
                    violations.append(f"approved model filename is present: {path}")
                    continue
                if suffix not in WEIGHT_SUFFIXES:
                    continue
                try:
                    size = path.stat().st_size
                except (FileNotFoundError, OSError):
                    size = path.lstat().st_size
                if size > max_bytes:
                    violations.append(f"large {suffix} payload is present: {path} ({size} bytes)")
    return sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    if args.max_bytes < 0:
        parser.error("--max-bytes cannot be negative")
    try:
        violations = scan(args.root, args.manifest, args.max_bytes)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"model-weight guard could not run: {error}")
        return 2
    if violations:
        for violation in violations:
            print(f"ERROR: {violation}")
        return 1
    print(f"model-weight guard OK across {len(args.root)} root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
