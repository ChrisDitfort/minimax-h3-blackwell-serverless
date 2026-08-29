"""The reference model: roles, limits, ordering, validation.

The product lets a user say "this image is the character, this video is the motion". H3
knows nothing about that. It sees an ordered list of references and a prompt that points
at them by ordinal - ``<Picture 1>``, ``<Video 2>``, ``<Audio 1>``. Everything in this
module exists to turn the first thing into the second, deterministically.

Limits are read from the node schema, not from documentation. At ComfyUI dec5d945 the
`MiniMaxH3ReferenceToVideo` autogrow templates declare:

    ref_images        min=0  max=9
    ref_videos        min=0  max=3
    ref_video_audios  min=0  max=3     soundtrack of the same-numbered ref_video
    ref_audios        min=0  max=3     standalone

Ordering is not a choice either. The node's own docstring fixes it: images, then videos
(each soundtrack's <Audio j> label emitted immediately before its <Video k>), then
standalone audio. Getting that wrong would misalign every ordinal in the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import errors

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_VIDEO_SOUNDTRACKS = 3
MAX_STANDALONE_AUDIO = 3

#: The model's own ceiling on distinct reference files, soundtracks included.
MODEL_MAX_REFERENCES = MAX_IMAGES + MAX_VIDEOS + MAX_STANDALONE_AUDIO + MAX_VIDEO_SOUNDTRACKS

#: The product's ceiling. Deliberately lower than the model's, and deliberately a separate
#: constant: 12 is a PrivoraVideo decision about cost and latency, not a model limit, and
#: conflating the two would make a future policy change look like a model change.
PRODUCT_MAX_REFERENCES = 12

#: Reference video duration, from the node's tooltip ("Reference video frames at 24 fps
#: (2-15s)") and its executable behaviour (fewer than 5 frames raises; longer than the
#: generation is truncated to it).
MIN_VIDEO_SECONDS = 2.0
MAX_VIDEO_SECONDS = 15.0
MIN_VIDEO_FRAMES = 5

#: Audio is resampled to the audio VAE's rate internally, so the constraint is duration,
#: not format. The ceiling matches video: a reference longer than the generation
#: contributes nothing the model can use.
MAX_AUDIO_SECONDS = 15.0

IMAGE_ROLES = frozenset(
    {
        "character", "identity", "face", "clothing", "product",
        "object", "environment", "style", "pose", "composition", "general",
    }
)

VIDEO_ROLES = frozenset(
    {
        "source", "motion", "body_performance", "camera_motion",
        "scene_structure", "timing", "action", "visual_style", "general",
    }
)

AUDIO_ROLES = frozenset(
    {"voice", "dialogue", "music", "rhythm", "ambience", "sound_effect", "general"}
)

ROLES_BY_TYPE = {"image": IMAGE_ROLES, "video": VIDEO_ROLES, "audio": AUDIO_ROLES}

#: How each role is described to the model in compiled prompt guidance. Several roles map
#: onto the same phrasing on purpose - the distinction is for the person choosing it in the
#: UI, and H3 has no separate handle for "identity" versus "character".
ROLE_GUIDANCE: dict[tuple[str, str], str] = {
    ("image", "character"): "the character",
    ("image", "identity"): "the character",
    ("image", "face"): "the face",
    ("image", "clothing"): "the clothing",
    ("image", "product"): "the product",
    ("image", "object"): "the object",
    ("image", "environment"): "the setting",
    ("image", "style"): "the visual style",
    ("image", "pose"): "the pose",
    ("image", "composition"): "the composition",
    ("video", "source"): "the source footage",
    ("video", "motion"): "the motion",
    ("video", "body_performance"): "the body performance",
    ("video", "camera_motion"): "the camera movement",
    ("video", "scene_structure"): "the scene structure",
    ("video", "timing"): "the timing",
    ("video", "action"): "the action",
    ("video", "visual_style"): "the visual style",
    ("audio", "voice"): "the voice",
    ("audio", "dialogue"): "the dialogue",
    ("audio", "music"): "the music",
    ("audio", "rhythm"): "the rhythm",
    ("audio", "ambience"): "the ambience",
    ("audio", "sound_effect"): "the sound effect",
}

#: Fidelity. `standard` -> the node's "match", `high` -> its "max" (2048px short edge).
#: The node's own tooltip is explicit that reference tokens ride through every sampling
#: step, so "max can be several times slower" - which is why this is a caller's choice
#: rather than a default.
FIDELITY_TO_REF_IMAGE_SIZE = {"standard": "match", "high": "max"}
DEFAULT_FIDELITY = "standard"


@dataclass
class Reference:
    """One user-supplied reference, after validation and before preprocessing."""

    type: str
    role: str = "general"
    #: Opaque handle from the control plane. Never a filename, never user text.
    id: str | None = None
    url: str | None = None
    data_base64: str | None = None
    #: Measured during preprocessing. None until then.
    duration_seconds: float | None = None
    #: For a video: the soundtrack travels with it rather than as a separate reference.
    soundtrack: "Reference | None" = None
    #: Assigned by `assign_ordinals`, 1-based per type, in presentation order.
    ordinal: int = 0

    @property
    def guidance(self) -> str:
        return ROLE_GUIDANCE.get((self.type, self.role), "")

    @property
    def tag(self) -> str:
        """The token H3 uses to refer to this reference in the prompt."""
        return {"image": "Picture", "video": "Video", "audio": "Audio"}[self.type] + f" {self.ordinal}"


@dataclass
class ReferenceSet:
    """Validated references in the exact order the node will present them."""

    images: list[Reference] = field(default_factory=list)
    videos: list[Reference] = field(default_factory=list)
    audios: list[Reference] = field(default_factory=list)

    @property
    def all(self) -> list[Reference]:
        """Presentation order: images, then videos, then standalone audio.

        Mirrors MiniMaxH3ReferenceToVideo's documented ordering. A video's soundtrack is
        not in this list - it rides on the video and takes its own <Audio j> ordinal.
        """
        return [*self.images, *self.videos, *self.audios]

    @property
    def file_count(self) -> int:
        """Distinct files the caller supplied, counting a soundtrack separately."""
        return (
            len(self.images)
            + len(self.videos)
            + len(self.audios)
            + sum(1 for video in self.videos if video.soundtrack is not None)
        )

    def is_empty(self) -> bool:
        return not (self.images or self.videos or self.audios)


def assign_ordinals(references: ReferenceSet) -> None:
    """Number references the way the node will, so prompt tags line up with conditioning.

    Audio ordinals are the subtle part. A video's soundtrack gets an ``<Audio j>`` label
    emitted *before* its ``<Video k>``, and standalone audio continues the same counter
    afterwards - so a soundtrack on video 1 makes the first standalone clip ``<Audio 2>``.
    Numbering standalone audio from 1 would silently point every audio instruction at the
    wrong clip.
    """
    for index, image in enumerate(references.images, start=1):
        image.ordinal = index

    audio_ordinal = 0
    for index, video in enumerate(references.videos, start=1):
        video.ordinal = index
        if video.soundtrack is not None:
            audio_ordinal += 1
            video.soundtrack.ordinal = audio_ordinal

    for audio in references.audios:
        audio_ordinal += 1
        audio.ordinal = audio_ordinal


def validate(references: ReferenceSet, *, max_references: int = PRODUCT_MAX_REFERENCES) -> None:
    """Reject an impossible reference set before it costs a model load.

    Raises PrivoraError with a code the control plane can branch on. Messages carry counts
    and limits only - never a filename, never a role the user typed as free text.
    """
    if len(references.images) > MAX_IMAGES:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_COUNT,
            f"At most {MAX_IMAGES} reference images are supported; {len(references.images)} were supplied.",
            {"type": "image", "limit": MAX_IMAGES, "supplied": len(references.images)},
        )
    if len(references.videos) > MAX_VIDEOS:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_COUNT,
            f"At most {MAX_VIDEOS} reference videos are supported; {len(references.videos)} were supplied.",
            {"type": "video", "limit": MAX_VIDEOS, "supplied": len(references.videos)},
        )
    if len(references.audios) > MAX_STANDALONE_AUDIO:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_COUNT,
            f"At most {MAX_STANDALONE_AUDIO} reference audio clips are supported; "
            f"{len(references.audios)} were supplied.",
            {"type": "audio", "limit": MAX_STANDALONE_AUDIO, "supplied": len(references.audios)},
        )

    total = references.file_count
    if total > max_references:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_COUNT,
            f"At most {max_references} reference files are supported in total; {total} were supplied.",
            {"limit": max_references, "supplied": total, "modelLimit": MODEL_MAX_REFERENCES},
        )

    for reference in references.all:
        allowed = ROLES_BY_TYPE.get(reference.type)
        if allowed is None:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_TYPE,
                f"Reference type {reference.type!r} is not supported. "
                "Supported types are image, video and audio.",
                {"type": reference.type},
            )
        if reference.role not in allowed:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_ROLE,
                f"Role {reference.role!r} is not valid for a {reference.type} reference.",
                {"type": reference.type, "role": reference.role, "allowed": sorted(allowed)},
            )


def validate_durations(references: ReferenceSet) -> None:
    """Check measured durations. Separate from `validate` because it needs the files.

    Called after preprocessing has probed each file, and before the model is loaded, so an
    over-long reference costs a decode rather than a generation.
    """
    for index, video in enumerate(references.videos, start=1):
        duration = video.duration_seconds
        if duration is None:
            continue
        if duration < MIN_VIDEO_SECONDS:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_DURATION,
                f"Reference video {index} is {duration:.1f}s; the minimum is {MIN_VIDEO_SECONDS:.0f}s.",
                {"type": "video", "index": index, "seconds": round(duration, 2),
                 "min": MIN_VIDEO_SECONDS, "max": MAX_VIDEO_SECONDS},
            )
        if duration > MAX_VIDEO_SECONDS:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_DURATION,
                f"Reference video {index} is {duration:.1f}s; the maximum is {MAX_VIDEO_SECONDS:.0f}s.",
                {"type": "video", "index": index, "seconds": round(duration, 2),
                 "min": MIN_VIDEO_SECONDS, "max": MAX_VIDEO_SECONDS},
            )

    for index, audio in enumerate(references.audios, start=1):
        duration = audio.duration_seconds
        if duration is not None and duration > MAX_AUDIO_SECONDS:
            raise errors.PrivoraError(
                errors.INVALID_REFERENCE_DURATION,
                f"Reference audio {index} is {duration:.1f}s; the maximum is {MAX_AUDIO_SECONDS:.0f}s.",
                {"type": "audio", "index": index, "seconds": round(duration, 2),
                 "max": MAX_AUDIO_SECONDS},
            )


def resolve_fidelity(value: str | None) -> str:
    """Product fidelity -> the node's `ref_image_size`."""
    name = (value or DEFAULT_FIDELITY).strip().lower()
    if name not in FIDELITY_TO_REF_IMAGE_SIZE:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"referenceFidelity must be one of {sorted(FIDELITY_TO_REF_IMAGE_SIZE)}; got {name!r}.",
            {"supplied": name, "allowed": sorted(FIDELITY_TO_REF_IMAGE_SIZE)},
        )
    return FIDELITY_TO_REF_IMAGE_SIZE[name]
