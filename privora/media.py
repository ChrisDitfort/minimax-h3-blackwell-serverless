"""Validation and probing for user-supplied reference video and audio.

Images already have a hardened path in handler.py - magic-byte sniffing, a Pillow decode,
pixel and byte caps, an SSRF guard on every redirect hop, generated filenames and
containment checks. This module is the equivalent for the two media types that path never
had to handle, and it follows the same rules:

* **Nothing the caller supplied becomes a filename.** Names are generated; the caller's
  id, url and original filename never touch the filesystem or a subprocess argument.
* **Never a shell.** Every ffprobe/ffmpeg invocation is an argument array, so a filename
  cannot become a command no matter what it contains.
* **Probe before decode.** Duration, codec and stream shape come from a metadata probe
  with a hard timeout, so an over-long or malformed file is rejected before anything
  expensive touches it.
* **Bounded everywhere.** Byte caps, duration caps, stream-count caps and a probe timeout,
  because "the file decoded fine" is not the same as "the file is reasonable".

ffprobe is invoked rather than linked: it ships with the base image for ComfyUI's own
video nodes, and running it as a separate short-lived process means a parser crash costs a
non-zero exit status rather than the worker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

from . import errors, references as references_module

#: Byte ceilings. Generous enough for a 15s reference, small enough that a hostile upload
#: cannot fill the container's disk before validation runs.
MAX_VIDEO_BYTES = int(os.environ.get("H3_MAX_REF_VIDEO_BYTES", str(256 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.environ.get("H3_MAX_REF_AUDIO_BYTES", str(64 * 1024 * 1024)))

#: A probe that has not answered in this long is a malformed file, not a slow one.
PROBE_TIMEOUT_SECONDS = int(os.environ.get("H3_MEDIA_PROBE_TIMEOUT", "20"))

#: Decode ceilings, applied to what the probe reports rather than to what the container
#: claims. A 16K reference contributes nothing and costs a great deal to decode.
MAX_VIDEO_WIDTH = 4096
MAX_VIDEO_HEIGHT = 4096
MAX_VIDEO_STREAMS = 4
MAX_AUDIO_CHANNELS = 8

#: Containers and codecs worth accepting. Anything else is refused with a clear code rather
#: than handed to a demuxer to find out.
VIDEO_CODECS = frozenset({"h264", "hevc", "vp9", "av1", "mpeg4", "vp8"})
AUDIO_CODECS = frozenset({"aac", "mp3", "opus", "vorbis", "flac", "pcm_s16le", "pcm_f32le"})

#: Magic bytes, checked before ffprobe ever sees the file. Cheap, and it means a mislabelled
#: upload is rejected by us rather than by a parser.
_VIDEO_SIGNATURES = (
    ("mp4", lambda d: d[4:8] == b"ftyp"),
    ("webm", lambda d: d[:4] == b"\x1a\x45\xdf\xa3"),
    ("mov", lambda d: d[4:12] in (b"ftypqt  ", b"moov")),
)
_AUDIO_SIGNATURES = (
    ("wav", lambda d: d[:4] == b"RIFF" and d[8:12] == b"WAVE"),
    ("mp3", lambda d: d[:3] == b"ID3" or d[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")),
    ("flac", lambda d: d[:4] == b"fLaC"),
    ("ogg", lambda d: d[:4] == b"OggS"),
    ("m4a", lambda d: d[4:8] == b"ftyp"),
)


@dataclass(frozen=True)
class MediaInfo:
    """What the probe established about one reference file."""

    kind: str
    duration_seconds: float
    codec: str
    width: int = 0
    height: int = 0
    sample_rate: int = 0
    channels: int = 0
    frame_rate: float = 0.0

    def as_metadata(self) -> dict:
        """Shape and duration only. Never a filename, never a container tag."""
        data = {"durationSeconds": round(self.duration_seconds, 3), "codec": self.codec}
        if self.kind == "video":
            data.update(width=self.width, height=self.height, frameRate=round(self.frame_rate, 3))
        else:
            data.update(sampleRate=self.sample_rate, channels=self.channels)
        return data


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def sniff(data: bytes, kind: str) -> str:
    """Identify a container from its first bytes, or refuse.

    Deliberately before ffprobe: a caller who labels a ZIP as a video should be turned away
    by a byte comparison, not by a demuxer.
    """
    signatures = _VIDEO_SIGNATURES if kind == "video" else _AUDIO_SIGNATURES
    header = data[:32]
    for name, matches in signatures:
        try:
            if matches(header):
                return name
        except Exception:  # pragma: no cover - a short header must not raise
            continue
    raise errors.PrivoraError(
        errors.INVALID_REFERENCE_TYPE,
        f"The supplied {kind} reference is not in a supported container.",
        {"type": kind},
    )


def check_size(data: bytes, kind: str) -> None:
    limit = MAX_VIDEO_BYTES if kind == "video" else MAX_AUDIO_BYTES
    if not data:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE, f"The supplied {kind} reference is empty.", {"type": kind}
        )
    if len(data) > limit:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"The supplied {kind} reference is larger than the {limit // (1024 * 1024)} MB limit.",
            {"type": kind, "limitBytes": limit, "suppliedBytes": len(data)},
        )


def probe(path: str, kind: str, *, runner=None) -> MediaInfo:
    """Read a file's shape with ffprobe. `runner` is injected so tests need no ffmpeg.

    The argument array is fixed and the path is the last element; nothing is interpolated
    into a string and no shell is involved, so a filename cannot become a command.
    """
    command = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        "-i", path,
    ]
    execute = runner or _run_ffprobe
    try:
        payload = execute(command)
    except subprocess.TimeoutExpired as error:
        raise errors.PrivoraError(
            errors.REFERENCE_PREPROCESSING_FAILED,
            f"The supplied {kind} reference could not be read within {PROBE_TIMEOUT_SECONDS}s.",
            {"type": kind},
            internal=f"ffprobe timeout on {path}",
        ) from error
    except Exception as error:
        raise errors.PrivoraError(
            errors.REFERENCE_PREPROCESSING_FAILED,
            f"The supplied {kind} reference could not be read.",
            {"type": kind},
            internal=f"{type(error).__name__}: {error}",
        ) from error

    return _interpret(payload, kind)


def _run_ffprobe(command: list[str]) -> dict:
    completed = subprocess.run(
        command, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe exited {completed.returncode}")
    return json.loads(completed.stdout or b"{}")


def _interpret(payload: dict, kind: str) -> MediaInfo:
    streams = payload.get("streams") or []
    if len(streams) > MAX_VIDEO_STREAMS:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"The supplied {kind} reference has too many streams.",
            {"type": kind, "streams": len(streams), "limit": MAX_VIDEO_STREAMS},
        )

    wanted = "video" if kind == "video" else "audio"
    chosen = next((s for s in streams if s.get("codec_type") == wanted), None)
    if chosen is None:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"The supplied {kind} reference contains no {wanted} stream.",
            {"type": kind},
        )

    codec = str(chosen.get("codec_name") or "")
    allowed = VIDEO_CODECS if kind == "video" else AUDIO_CODECS
    if codec not in allowed:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"The {kind} codec {codec!r} is not supported.",
            {"type": kind, "codec": codec, "supported": sorted(allowed)},
        )

    duration = _duration_of(payload, chosen, kind)

    if kind == "video":
        width, height = int(chosen.get("width") or 0), int(chosen.get("height") or 0)
        if width <= 0 or height <= 0:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_TYPE,
                "The supplied video reference has no usable dimensions.", {"type": kind},
            )
        if width > MAX_VIDEO_WIDTH or height > MAX_VIDEO_HEIGHT:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_TYPE,
                f"The supplied video reference is larger than "
                f"{MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT}.",
                {"type": kind, "width": width, "height": height},
            )
        return MediaInfo("video", duration, codec, width=width, height=height,
                         frame_rate=_frame_rate(chosen))

    channels = int(chosen.get("channels") or 0)
    if channels > MAX_AUDIO_CHANNELS:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"The supplied audio reference has {channels} channels; the limit is "
            f"{MAX_AUDIO_CHANNELS}.",
            {"type": kind, "channels": channels, "limit": MAX_AUDIO_CHANNELS},
        )
    return MediaInfo("audio", duration, codec,
                     sample_rate=int(chosen.get("sample_rate") or 0), channels=channels)


def _duration_of(payload: dict, stream: dict, kind: str) -> float:
    for candidate in (stream.get("duration"), (payload.get("format") or {}).get("duration")):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    raise errors.PrivoraError(
        errors.INVALID_REFERENCE_DURATION,
        f"The supplied {kind} reference reports no duration.",
        {"type": kind},
    )


def _frame_rate(stream: dict) -> float:
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        numerator, _, denominator = str(raw).partition("/")
        denominator_value = float(denominator or 1)
        return float(numerator) / denominator_value if denominator_value else 0.0
    except (TypeError, ValueError):
        return 0.0


def check_aggregate(infos: list[MediaInfo], kind: str, *, limit_seconds: float) -> None:
    """Reject a set whose combined duration is unreasonable even if each file is legal.

    Three 15-second videos are individually fine and collectively 45 seconds of reference
    tokens riding through every sampling step. The per-file limit does not catch that.
    """
    total = sum(info.duration_seconds for info in infos)
    if total > limit_seconds:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_DURATION,
            f"The {kind} references total {total:.1f}s; the combined limit is "
            f"{limit_seconds:.0f}s.",
            {"type": kind, "totalSeconds": round(total, 2), "limitSeconds": limit_seconds},
        )


#: Aggregate ceilings. Each is the per-file maximum times the number of files that type
#: allows, which is the point at which the model would be doing nothing but reading
#: references.
MAX_TOTAL_VIDEO_SECONDS = references_module.MAX_VIDEO_SECONDS * references_module.MAX_VIDEOS
MAX_TOTAL_AUDIO_SECONDS = references_module.MAX_AUDIO_SECONDS * references_module.MAX_STANDALONE_AUDIO
