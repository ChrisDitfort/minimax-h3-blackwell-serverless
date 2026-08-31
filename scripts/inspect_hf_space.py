#!/usr/bin/env python3
"""Inspect a private Space build through metadata only; never pull image layers."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


EXPECTED_SPACE_REPOSITORY = "CDitfort/privora-h3-runpod-worker"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9._-]+")
URL_RE = re.compile(r"https?://[^\s'\"<>]+")
ONGOING_MARKERS = ("BUILDING", "STARTING", "PREPARING")
BUILD_FAILURE_MARKERS = ("BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE")


def safe_error(error: BaseException) -> str:
    rendered = TOKEN_RE.sub("[REDACTED]", str(error))
    token = os.environ.get("HF_TOKEN")
    if token:
        rendered = rendered.replace(token, "[REDACTED]")
    def redact_url(match: re.Match[str]) -> str:
        parsed = urllib.parse.urlsplit(match.group(0))
        if parsed.query or parsed.fragment:
            return urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, "[REDACTED]", "")
            )
        return match.group(0)
    rendered = URL_RE.sub(redact_url, rendered)
    return rendered[:2000]


def fetch(repo: str, token: str) -> dict:
    url = "https://huggingface.co/api/spaces/" + urllib.parse.quote(repo, safe="/")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=EXPECTED_SPACE_REPOSITORY)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if args.repo != EXPECTED_SPACE_REPOSITORY:
        parser.error("--repo is not the approved Space")
    if not REVISION_RE.fullmatch(args.revision):
        parser.error("--revision must be a full immutable Space revision")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Space inspection failed: HF_TOKEN is absent")
        return 1
    try:
        document = fetch(args.repo, token)
        revision = document.get("sha")
        if revision != args.revision:
            raise RuntimeError(f"Space main revision changed: expected {args.revision}, found {revision}")
        if document.get("private") is not True or document.get("sdk") != "docker":
            raise RuntimeError("Space is not the expected private Docker Space")
        runtime = document.get("runtime") or {}
        stage = str(runtime.get("stage") or "UNKNOWN")
        subdomain = document.get("subdomain") or runtime.get("host")
        if isinstance(subdomain, str) and subdomain.endswith(".hf.space"):
            subdomain = subdomain.removesuffix(".hf.space")
        if not isinstance(subdomain, str) or not re.fullmatch(r"[a-z0-9-]+", subdomain):
            subdomain = "unavailable"
        registry_ref = (
            f"registry.hf.space/{subdomain}:latest" if subdomain != "unavailable" else "unavailable"
        )
        hardware = runtime.get("hardware")
        if isinstance(hardware, dict):
            hardware = hardware.get("current")
        safe = {
            "repository": args.repo,
            "revision": revision,
            "private": True,
            "sdk": "docker",
            "stage": stage,
            "hardware": hardware,
            "subdomain": subdomain,
            "registryRef": registry_ref,
        }
        print(json.dumps(safe, sort_keys=True))
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write(f"space_stage={stage}\n")
                handle.write(f"space_subdomain={subdomain}\n")
                handle.write(f"registry_ref={registry_ref}\n")
        if any(marker in stage for marker in BUILD_FAILURE_MARKERS):
            return 1
        if any(marker in stage for marker in ONGOING_MARKERS):
            return 10
        return 0
    except (OSError, RuntimeError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"Space inspection failed: {safe_error(error)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
