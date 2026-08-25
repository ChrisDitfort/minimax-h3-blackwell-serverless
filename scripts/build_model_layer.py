#!/usr/bin/env python3
"""Stream a MiniMax H3 model straight from Hugging Face into a Docker layer tarball.

The model files total ~42 GB, which does not fit on a GitHub Actions runner if we
download a file and *then* copy it into a tar (that needs 2x the file size). Because
the exact byte size of every asset is known up front (see models.tsv), we can write
the tar header before the bytes arrive and stream the HTTP body directly into the
archive. Peak disk is therefore ~1x the model size instead of ~2x.

The stream is resumable: a dropped connection is retried with a Range request from
the last byte received. SHA-256 is computed incrementally over the bytes in order, so
resuming does not invalidate the digest.

Usage:
    build_model_layer.py --url URL --dest PATH_IN_IMAGE --size BYTES --sha256 HEX --out LAYER.tar
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

USER_AGENT = "minimax-h3-blackwell-serverless/1.0"
CHUNK = 8 * 1024 * 1024
MAX_ATTEMPTS = 8
PROGRESS_EVERY = 2 * 1024 * 1024 * 1024  # log roughly every 2 GB


def log(message: str) -> None:
    print(message, flush=True)


def human(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000_000:.2f} GB"


class ResumableHTTPReader:
    """A read-only file-like object that transparently resumes a broken download.

    ``tarfile.addfile`` pulls exactly ``expected_size`` bytes out of this object. If the
    socket dies partway through, we reconnect with ``Range: bytes=<received>-`` and keep
    going, so the tar member is still written in one uninterrupted pass.
    """

    def __init__(self, url: str, expected_size: int, digest: "hashlib._Hash") -> None:
        self.url = url
        self.expected_size = expected_size
        self.digest = digest
        self.received = 0
        self._response = None
        self._next_progress = PROGRESS_EVERY
        self._started = time.monotonic()

    def _open(self) -> None:
        headers = {"User-Agent": USER_AGENT}
        if self.received:
            headers["Range"] = f"bytes={self.received}-"

        request = urllib.request.Request(self.url, headers=headers)
        response = urllib.request.urlopen(request, timeout=120)

        status = response.getcode()
        if self.received and status != 206:
            response.close()
            raise RuntimeError(
                f"resume requested at byte {self.received} but server returned HTTP {status} "
                "(no ranged-request support); cannot continue safely"
            )
        if not self.received and status != 200:
            response.close()
            raise RuntimeError(f"unexpected HTTP {status} for initial request")

        self._response = response

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.expected_size - self.received
        size = min(size, self.expected_size - self.received)
        if size <= 0:
            return b""

        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if self._response is None:
                    self._open()
                chunk = self._response.read(size)
            except (urllib.error.URLError, OSError, RuntimeError) as error:
                last_error = error
                self._close_response()
                if attempt == MAX_ATTEMPTS:
                    break
                delay = min(30, 2 ** attempt)
                log(
                    f"  ! transfer error at byte {self.received} "
                    f"({type(error).__name__}); retry {attempt}/{MAX_ATTEMPTS} in {delay}s"
                )
                time.sleep(delay)
                continue

            if not chunk:
                # Clean EOF before the expected size means the connection was truncated.
                self._close_response()
                if self.received >= self.expected_size:
                    return b""
                last_error = RuntimeError(f"stream truncated at byte {self.received}")
                if attempt == MAX_ATTEMPTS:
                    break
                delay = min(30, 2 ** attempt)
                log(f"  ! stream truncated at byte {self.received}; resuming in {delay}s")
                time.sleep(delay)
                continue

            self.received += len(chunk)
            self.digest.update(chunk)
            self._report()
            return chunk

        raise RuntimeError(f"download failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def _report(self) -> None:
        if self.received < self._next_progress:
            return
        self._next_progress += PROGRESS_EVERY
        elapsed = max(time.monotonic() - self._started, 1e-6)
        rate = self.received / elapsed / 1_000_000
        pct = 100.0 * self.received / self.expected_size
        log(
            f"  .. {human(self.received)} / {human(self.expected_size)} "
            f"({pct:5.1f}%) at {rate:.0f} MB/s"
        )

    def _close_response(self) -> None:
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None

    def close(self) -> None:
        self._close_response()


def build_layer(url: str, dest: str, size: int, expected_sha256: str, out_path: str) -> None:
    dest = dest.lstrip("/")
    if not dest:
        raise SystemExit("--dest must be a non-empty path inside the image")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    log(f"Building layer for /{dest}")
    log(f"  expecting {human(size)} ({size} bytes), sha256 {expected_sha256}")

    digest = hashlib.sha256()
    reader = ResumableHTTPReader(url, size, digest)

    # Uncompressed tar: safetensors are already quantised and compress almost not at all,
    # so gzip here would burn runner CPU for no size win. `crane append` handles the
    # registry-side compression.
    try:
        with tarfile.open(out_path, "w", format=tarfile.GNU_FORMAT) as tar:  # noqa: SIM117
            # Parent directories already exist in the base image; emitting them keeps the
            # layer self-contained if that ever changes.
            parts = dest.split("/")[:-1]
            for index in range(1, len(parts) + 1):
                info = tarfile.TarInfo("/".join(parts[:index]))
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)

            info = tarfile.TarInfo(dest)
            info.size = size
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, reader)
    except OSError as error:
        raise SystemExit(
            f"FATAL: could not write {size} bytes for {dest} into the layer "
            f"(got {reader.received} bytes): {error}"
        ) from error
    finally:
        reader.close()

    if reader.received != size:
        raise SystemExit(
            f"FATAL: size mismatch for {dest}: expected {size} bytes, received {reader.received}"
        )

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"FATAL: sha256 mismatch for {dest}\n"
            f"  expected {expected_sha256}\n"
            f"  actual   {actual_sha256}"
        )

    layer_bytes = os.path.getsize(out_path)
    if layer_bytes <= size:
        raise SystemExit(f"FATAL: layer {out_path} is implausibly small ({layer_bytes} bytes)")

    log(f"  OK  {human(size)} verified (size + sha256), layer at {out_path} ({human(layer_bytes)})")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dest", required=True, help="destination path inside the image")
    parser.add_argument("--size", required=True, type=int, help="exact expected byte size")
    parser.add_argument("--sha256", required=True, help="expected lowercase hex sha256")
    parser.add_argument("--out", required=True, help="tar file to write")
    args = parser.parse_args(argv)

    if args.size <= 0:
        raise SystemExit("--size must be positive")

    build_layer(args.url, args.dest, args.size, args.sha256.strip().lower(), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
