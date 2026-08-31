#!/usr/bin/env python3
"""Credentialed visibility check and allowlisted upload for the private Docker Space."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
from pathlib import Path


EXPECTED_SPACE_REPOSITORY = "CDitfort/privora-h3-runpod-worker"
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9._-]+")
URL_RE = re.compile(r"https?://[^\s'\"<>]+")


def _redact_url(match: re.Match[str]) -> str:
    value = match.group(0)
    parsed = urllib.parse.urlsplit(value)
    if parsed.query or parsed.fragment:
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "[REDACTED]", ""))
    return value


def _safe_error(error: BaseException) -> str:
    rendered = TOKEN_RE.sub("[REDACTED]", str(error))
    token = os.environ.get("HF_TOKEN")
    if token:
        rendered = rendered.replace(token, "[REDACTED]")
    rendered = URL_RE.sub(_redact_url, rendered)
    return rendered[:2000]


def _expected_files(folder: Path) -> set[str]:
    manifest_path = folder / "space-publication-manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        publication = json.load(handle)
    records = publication.get("files")
    if publication.get("spaceRepository") != EXPECTED_SPACE_REPOSITORY or not isinstance(records, list):
        raise ValueError("invalid Space publication manifest")
    expected = {record["path"] for record in records}
    expected.add("space-publication-manifest.json")
    actual = {
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise ValueError(f"staged file set differs from its manifest: {sorted(actual ^ expected)}")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", default=EXPECTED_SPACE_REPOSITORY)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--commit-message", default="Publish model-free Privora H3 worker")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if args.repo != EXPECTED_SPACE_REPOSITORY:
        parser.error("--repo is not the approved private Docker Space")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Space publishing failed: HF_TOKEN is absent (an OIDC token is required)")
        return 1

    try:
        from huggingface_hub import HfApi

        expected = _expected_files(args.folder)
        api = HfApi(token=token)
        before = api.space_info(repo_id=args.repo, token=token)
        if before.id != args.repo or before.private is not True:
            raise RuntimeError("destination is absent, not private, or resolved to the wrong Space")
        print(
            f"Private Space visibility OK: repository={before.id} "
            f"revision={before.sha or 'empty'} approved_files={len(expected)}"
        )
        if args.check_only:
            print("Dry run complete: execute_publish=false writes=0")
            return 0

        commit = api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=str(args.folder),
            path_in_repo=".",
            delete_patterns=["*"],
            commit_message=args.commit_message,
            token=token,
        )
        revision = commit.oid
        if not revision:
            raise RuntimeError("Hugging Face did not return the Space commit revision")
        after = api.space_info(repo_id=args.repo, revision=revision, token=token)
        remote_files = {sibling.rfilename for sibling in (after.siblings or [])}
        allowed_remote = expected | {".gitattributes"}
        if not expected.issubset(remote_files) or not remote_files.issubset(allowed_remote):
            raise RuntimeError(
                "published Space file set is not exact: "
                f"missing={sorted(expected - remote_files)} unexpected={sorted(remote_files - allowed_remote)}"
            )
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write(f"space_revision={revision}\n")
        print(
            f"Space publication committed: repository={args.repo} revision={revision} "
            f"files={len(remote_files)} model_weights=0"
        )
        return 0
    except Exception as error:  # the Hub client has several version-specific error types
        print(f"Space publishing failed: {_safe_error(error)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
