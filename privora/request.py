"""The PrivoraVideo worker request: parse, validate, route.

This is the stable surface. Everything below it - ComfyUI, node names, checkpoint files,
even H3 itself - is an implementation detail the control plane never sees.

    PrivoraVideo request  ->  GenerationRequest  ->  workflow builder  ->  ComfyUI

Four modes, two model families:

    create      prompt only                          FL2VA
    animate     first and/or last keyframe           FL2VA
    references  mixed multimodal references          REF2VA
    remix       a source video plus references       REF2VA

`remix` is deliberately routed to the same place as `references`. H3 has no deterministic
video-editing capability, and pretending otherwise in the API would be a promise the model
cannot keep - see MODE_NOTES.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import canvas as canvas_module
from . import errors, models as models_module
from . import references as references_module
from .prompt import CompiledPrompt, compile_prompt
from .references import Reference, ReferenceSet

CREATE = "create"
ANIMATE = "animate"
REFERENCES = "references"
REMIX = "remix"

MODES = (CREATE, ANIMATE, REFERENCES, REMIX)

#: Which model family each mode needs. The two checkpoints are separate files and cannot
#: serve each other's mode.
MODE_FAMILY = {CREATE: "fl2va", ANIMATE: "fl2va", REFERENCES: "ref2va", REMIX: "ref2va"}

#: Honest description of what each mode does, surfaced in `/capabilities`.
MODE_NOTES = {
    CREATE: "Text to video with native audio.",
    ANIMATE: "Text to video anchored to a first and/or last keyframe.",
    REFERENCES: "Reference-guided generation from images, videos and audio.",
    REMIX: (
        "Reference-guided regeneration using a source video. This is not deterministic "
        "video editing: H3 regenerates the clip guided by the references rather than "
        "modifying the source frames, so unreferenced detail will change."
    ),
}

#: 2^64 - 1, the ceiling ComfyUI's RandomNoise.noise_seed enforces.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


@dataclass
class GenerationRequest:
    """A validated request, resolved to everything the workflow builder needs."""

    mode: str
    prompt: CompiledPrompt
    canvas: canvas_module.Canvas
    seed: int
    references: ReferenceSet = field(default_factory=ReferenceSet)
    first_frame: Reference | None = None
    last_frame: Reference | None = None
    ref_image_size: str = "match"
    #: quality | turbo | turbo_fast. Resolved to a checkpoint+LoRA by privora.models.
    generation_mode: str = models_module.DEFAULT_GENERATION_MODE
    #: Carried through untouched; the handler owns privacy, not this layer.
    privacy: dict = field(default_factory=dict)
    encryption: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    #: True when the request arrived in the pre-rebuild schema.
    legacy: bool = False

    @property
    def family(self) -> str:
        return MODE_FAMILY[self.mode]

    def as_metadata(self) -> dict:
        """Job metadata: what was actually run, not what was asked for."""
        data = {
            "mode": self.mode,
            "seed": self.seed,
            "model": self.family,
            **self.canvas.as_metadata(),
        }
        if not self.references.is_empty():
            data["references"] = {
                "images": len(self.references.images),
                "videos": len(self.references.videos),
                "audio": len(self.references.audios),
                "soundtracks": sum(
                    1 for video in self.references.videos if video.soundtrack is not None
                ),
                "total": self.references.file_count,
                "fidelity": "high" if self.ref_image_size == "max" else "standard",
            }
        if self.first_frame is not None or self.last_frame is not None:
            data["keyframes"] = {
                "first": self.first_frame is not None,
                "last": self.last_frame is not None,
            }
        if self.legacy:
            data["schema"] = "legacy"
        return data


def _string(value, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise errors.PrivoraError(
            errors.MISSING_PROMPT, f"{name} must be a string.", {"field": name}
        )
    return value


def _resolve_seed(value) -> int:
    if value is None:
        return random.randrange(0, MAX_SEED)
    if isinstance(value, bool) or not isinstance(value, int):
        raise errors.PrivoraError(
            errors.INVALID_SEED, "seed must be an integer.", {"supplied": str(type(value).__name__)}
        )
    if not 0 <= value <= MAX_SEED:
        raise errors.PrivoraError(
            errors.INVALID_SEED,
            f"seed must be between 0 and {MAX_SEED}.",
            {"supplied": value, "max": MAX_SEED},
        )
    return value


def _reference_from(spec, expected_type: str | None = None) -> Reference:
    if not isinstance(spec, dict):
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE, "Each reference must be an object.", {}
        )
    kind = str(spec.get("type") or expected_type or "").strip().lower()
    if kind not in references_module.ROLES_BY_TYPE:
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE,
            f"Reference type {kind!r} is not supported. Supported types are image, video and audio.",
            {"type": kind},
        )
    soundtrack = None
    if spec.get("soundtrack") is not None:
        soundtrack = _reference_from(spec["soundtrack"], expected_type="audio")
    return Reference(
        type=kind,
        role=str(spec.get("role") or "general").strip().lower(),
        id=spec.get("id"),
        url=spec.get("url"),
        data_base64=spec.get("data") or spec.get("dataBase64"),
        soundtrack=soundtrack,
    )


def _collect_references(raw) -> ReferenceSet:
    if raw is None:
        return ReferenceSet()
    if not isinstance(raw, list):
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_TYPE, "references must be a list.", {}
        )

    collected = ReferenceSet()
    for spec in raw:
        reference = _reference_from(spec)
        if reference.type == "image":
            collected.images.append(reference)
        elif reference.type == "video":
            collected.videos.append(reference)
        else:
            collected.audios.append(reference)
    return collected


def parse(payload: dict, *, max_references: int = references_module.PRODUCT_MAX_REFERENCES) -> GenerationRequest:
    """Validate a request and resolve everything the model will need.

    Every rejection here happens before a checkpoint is touched, which is the point: a
    malformed reference set should cost a few milliseconds, not a model load.
    """
    if not isinstance(payload, dict):
        raise errors.PrivoraError(errors.UNSUPPORTED_MODE, "The request must be an object.", {})

    if "workflow" in payload or ("width" in payload and "height" in payload and "mode" not in payload):
        return parse_legacy(payload)

    mode = str(payload.get("mode") or CREATE).strip().lower()
    if mode not in MODES:
        raise errors.PrivoraError(
            errors.UNSUPPORTED_MODE,
            f"mode must be one of {', '.join(MODES)}.",
            {"supplied": mode, "supported": list(MODES)},
        )

    prompt_text = _string(payload.get("prompt"), "prompt").strip()
    if not prompt_text:
        raise errors.PrivoraError(
            errors.MISSING_PROMPT, "A prompt is required.", {"mode": mode}
        )

    quality = str(payload.get("quality") or canvas_module.DEFAULT_QUALITY).strip().lower()
    if quality not in canvas_module.QUALITY_TIERS:
        raise errors.PrivoraError(
            errors.INVALID_QUALITY,
            f"quality must be one of {', '.join(canvas_module.QUALITY_TIERS)}.",
            {"supplied": quality, "supported": sorted(canvas_module.QUALITY_TIERS)},
        )

    aspect = str(payload.get("aspectRatio") or canvas_module.DEFAULT_ASPECT_RATIO).strip()
    if aspect not in canvas_module.ASPECT_RATIOS:
        raise errors.PrivoraError(
            errors.INVALID_ASPECT_RATIO,
            f"aspectRatio must be one of {', '.join(canvas_module.ASPECT_RATIOS)}.",
            {"supplied": aspect, "supported": sorted(canvas_module.ASPECT_RATIOS)},
        )

    duration = payload.get("duration", 5)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise errors.PrivoraError(
            errors.INVALID_DURATION, "duration must be a positive number of seconds.",
            {"supplied": str(duration)},
        )

    try:
        resolved = canvas_module.resolve_canvas(
            quality=quality, aspect_ratio=aspect, duration_seconds=float(duration)
        )
    except ValueError as error:
        raise errors.PrivoraError(
            errors.INVALID_DURATION, str(error), {"duration": duration}
        ) from error

    first_frame = last_frame = None
    if payload.get("firstFrame") is not None:
        first_frame = _reference_from(payload["firstFrame"], expected_type="image")
    if payload.get("lastFrame") is not None:
        last_frame = _reference_from(payload["lastFrame"], expected_type="image")

    collected = _collect_references(payload.get("references"))

    if mode == ANIMATE and first_frame is None and last_frame is None:
        raise errors.PrivoraError(
            errors.MISSING_FRAME,
            "animate needs at least a firstFrame or a lastFrame.",
            {"mode": mode},
        )
    if mode in (REFERENCES, REMIX) and collected.is_empty():
        raise errors.PrivoraError(
            errors.INVALID_REFERENCE_COUNT,
            f"{mode} needs at least one reference.",
            {"mode": mode},
        )
    if mode in (CREATE, ANIMATE) and not collected.is_empty():
        raise errors.PrivoraError(
            errors.UNSUPPORTED_MODE,
            f"{mode} does not accept references; use the references mode instead.",
            {"mode": mode, "supplied": collected.file_count},
        )

    references_module.validate(collected, max_references=max_references)
    references_module.assign_ordinals(collected)

    return GenerationRequest(
        mode=mode,
        prompt=compile_prompt(
            prompt_text, collected,
            camera=payload.get("camera"), style=payload.get("style"),
        ),
        canvas=resolved,
        seed=_resolve_seed(payload.get("seed")),
        references=collected,
        first_frame=first_frame,
        last_frame=last_frame,
        ref_image_size=references_module.resolve_fidelity(payload.get("referenceFidelity")),
        generation_mode=models_module.parse_generation_mode(payload.get("generationMode")),
        privacy=payload.get("privacy") or {},
        encryption=payload.get("encryption") or {},
        output=payload.get("output") or {},
        progress=payload.get("progress") or {},
    )


def parse_legacy(payload: dict) -> GenerationRequest:
    """Accept the pre-rebuild request shape and resolve it into the new structure.

    Kept because rollback and benchmark comparison both depend on being able to run the
    exact old workload through the new worker. The old schema carried explicit
    width/height/frames/steps, so nothing is re-derived from the tier table - the numbers
    the caller asked for are the numbers that run.
    """
    prompt_text = _string(payload.get("prompt"), "prompt").strip()
    width = payload.get("width")
    height = payload.get("height")

    if not prompt_text or not isinstance(width, int) or not isinstance(height, int):
        raise errors.PrivoraError(
            errors.MISSING_PROMPT,
            "A legacy request needs prompt, width and height.",
            {"schema": "legacy"},
        )

    resolved = canvas_module.canvas_from_dimensions(
        width=width,
        height=height,
        frames=int(payload.get("frames", 124)),
        steps=int(payload.get("steps", 20)),
    )

    first_frame = None
    if payload.get("image_url") or payload.get("image_base64"):
        first_frame = Reference(
            type="image", role="general",
            url=payload.get("image_url"), data_base64=payload.get("image_base64"),
        )

    family = MODE_FAMILY[ANIMATE if first_frame is not None else CREATE]
    legacy_mode = models_module.steps_to_generation_mode(family, resolved.steps)
    if legacy_mode is None:
        # An unusual legacy step count keeps running on the base checkpoint at that
        # count. Turbo is never selected implicitly: swapping the model underneath a
        # caller who only asked for fewer steps would change what they are paying for.
        legacy_mode = models_module.QUALITY

    return GenerationRequest(
        mode=ANIMATE if first_frame is not None else CREATE,
        prompt=compile_prompt(prompt_text),
        canvas=resolved,
        seed=_resolve_seed(payload.get("seed")),
        first_frame=first_frame,
        privacy=payload.get("privacy") or {},
        encryption=payload.get("encryption") or {},
        output=payload.get("output") or {},
        progress=payload.get("progress") or {},
        generation_mode=legacy_mode,
        legacy=True,
    )


def capabilities(*, ref2va_available: bool) -> dict:
    """What this worker can currently do. Honest about what is not installed."""
    return {
        "modes": {
            name: {
                "family": MODE_FAMILY[name],
                "description": MODE_NOTES[name],
                "available": ref2va_available or MODE_FAMILY[name] == "fl2va",
            }
            for name in MODES
        },
        "quality": {
            name: {
                "steps": tier.steps,
                "dimensions": {
                    ratio: "{}x{}".format(
                        *(
                            canvas_module.resolve_canvas(quality=name, aspect_ratio=ratio).width,
                            canvas_module.resolve_canvas(quality=name, aspect_ratio=ratio).height,
                        )
                    )
                    for ratio in canvas_module.ASPECT_RATIOS
                },
            }
            for name, tier in canvas_module.QUALITY_TIERS.items()
        },
        "aspectRatios": sorted(canvas_module.ASPECT_RATIOS),
        "duration": {
            "fps": canvas_module.FPS,
            "frameGrid": "frames % 17 == 5",
            "minSeconds": round(canvas_module.MIN_FRAMES / canvas_module.FPS, 4),
            "maxSeconds": round(canvas_module.MAX_FRAMES / canvas_module.FPS, 4),
            "trainedRangeSeconds": [
                round(canvas_module.TRAINED_MIN_FRAMES / canvas_module.FPS, 4),
                round(canvas_module.TRAINED_MAX_FRAMES / canvas_module.FPS, 4),
            ],
        },
        "references": {
            "maxImages": references_module.MAX_IMAGES,
            "maxVideos": references_module.MAX_VIDEOS,
            "maxAudio": references_module.MAX_STANDALONE_AUDIO,
            "maxVideoSoundtracks": references_module.MAX_VIDEO_SOUNDTRACKS,
            "maxTotal": references_module.PRODUCT_MAX_REFERENCES,
            "modelMaxTotal": references_module.MODEL_MAX_REFERENCES,
            "videoSeconds": [references_module.MIN_VIDEO_SECONDS, references_module.MAX_VIDEO_SECONDS],
            "audioMaxSeconds": references_module.MAX_AUDIO_SECONDS,
            "roles": {
                "image": sorted(references_module.IMAGE_ROLES),
                "video": sorted(references_module.VIDEO_ROLES),
                "audio": sorted(references_module.AUDIO_ROLES),
            },
            "fidelity": sorted(references_module.FIDELITY_TO_REF_IMAGE_SIZE),
        },
        "maxSeed": MAX_SEED,
    }
