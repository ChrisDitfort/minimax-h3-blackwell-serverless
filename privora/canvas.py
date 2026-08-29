"""Canvas and timing arithmetic, mirrored from the model's own implementation.

Every number here is a restatement of ComfyUI's `comfy_extras/nodes_minimax_h3.py` at the
revision this image pins (dec5d945 / v0.30.2). Nothing is invented, and nothing is copied
from documentation: the constants below were read out of the node source, because the node
is what will actually reject a bad request.

    CANVAS_MULTIPLE = 32          per-axis rounding
    BASE_SHORT_EDGE = 768         the canvas short edge H3 was trained on
    MAX_PIXELS = 768 * 1344       area cap
    FPS = 24
    frames must satisfy n % 17 == 5
    length: min 5, max 3600, trained range ~124-362

The point of restating them rather than calling the node is that the worker has to reject
an impossible request *before* it costs a model load, and it has to tell the caller the
real numbers it will use. `verify_against_comfyui()` exists so the copy cannot rot: it
re-derives these from the installed node source when it is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24
AUDIO_LATENT_FPS = 40

#: The frame grid. H3's temporal packing is 17k+5 latent frames.
FRAME_GRID_MODULUS = 17
FRAME_GRID_REMAINDER = 5

#: Hard bounds from the node schema (`length` min/max/step).
MIN_FRAMES = 5
MAX_FRAMES = 3600

#: The range the checkpoint was actually trained on, per the node's own tooltip:
#: "124 = ~5s, trained range is ~124-362". Requests outside it are honoured but flagged,
#: because "the node accepts it" and "the model was trained for it" are different claims.
TRAINED_MIN_FRAMES = 124
TRAINED_MAX_FRAMES = 362

#: Reference images at `ref_image_size="max"` are scaled to this short edge.
REF_IMAGE_SHORT_EDGE = 2048

#: Semantic ratios the product exposes. The model has no notion of these - it only sees
#: a width and a height - so this table is a PrivoraVideo abstraction over adapt_canvas().
ASPECT_RATIOS: dict[str, float] = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "21:9": 21 / 9,
}


def align_frame_count(frames: int) -> int:
    """Snap up to the 17k+5 grid. Identical to the node's `align_frame_count`."""
    frames = max(MIN_FRAMES, int(frames))
    while frames % FRAME_GRID_MODULUS != FRAME_GRID_REMAINDER:
        frames += 1
    return frames


def maximum_aligned_frame_count() -> int:
    """Largest frame count that is both on H3's grid and within the node ceiling.

    ``MAX_FRAMES`` itself is not necessarily executable: 3600 is off the 17k+5 grid,
    and aligning it upward produces 3609. Walking backward through ``align_frame_count``
    keeps capability reporting tied to the validation arithmetic instead of publishing a
    separately hard-coded duration that can drift.
    """
    for requested in range(MAX_FRAMES, MIN_FRAMES - 1, -1):
        aligned = align_frame_count(requested)
        if aligned <= MAX_FRAMES:
            return aligned
    raise RuntimeError(  # pragma: no cover - impossible with the pinned node schema
        "The configured H3 frame bounds contain no legal frame count."
    )


def video_latent_t(frame_count: int) -> int:
    """Latent temporal length. Identical to the node's `video_latent_t`."""
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // FRAME_GRID_MODULUS) * 5 + 2


def audio_latent_t(frame_count: int) -> int:
    return round((frame_count / FPS) * AUDIO_LATENT_FPS)


def adapt_canvas(width: float, height: float) -> tuple[int, int]:
    """768-short-edge canvas with a 768*1344 area cap, per-axis rounded to 32.

    A verbatim restatement of the node's `adapt_canvas`. Given any aspect this returns the
    canvas H3 actually wants for it, which is why the product can offer semantic ratios
    instead of asking a caller to guess legal dimensions.
    """
    ratio = width / height
    if ratio >= 1.0:
        nominal_w, nominal_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nominal_w, nominal_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio

    if nominal_w * nominal_h > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_w * nominal_h))
        nominal_w, nominal_h = nominal_w * scale, nominal_h * scale

    return (
        max(CANVAS_MULTIPLE, round(nominal_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


@dataclass(frozen=True)
class QualityTier:
    """A product tier. `short_edge` is what the tier means; the canvas follows from it."""

    name: str
    short_edge: int
    steps: int
    #: None means "the model's own area cap"; a tier below native scales the canvas down.
    max_pixels: int | None = None


#: The three measured tiers, plus the one the model natively wants.
#:
#: Draft/Standard/HD are kept at the exact pixel dimensions the existing benchmarks were
#: measured at, so the pre-rebuild baseline stays comparable. `ultra` is not a 2K upscale
#: or a second model - it is simply H3's own default canvas (1344x768), which the current
#: product has never exposed.
QUALITY_TIERS: dict[str, QualityTier] = {
    "draft": QualityTier("draft", short_edge=288, steps=20, max_pixels=512 * 288),
    "standard": QualityTier("standard", short_edge=576, steps=20, max_pixels=1024 * 576),
    "hd": QualityTier("hd", short_edge=704, steps=20, max_pixels=1280 * 704),
    "ultra": QualityTier("ultra", short_edge=768, steps=20, max_pixels=MAX_PIXELS),
}

DEFAULT_QUALITY = "standard"
DEFAULT_ASPECT_RATIO = "16:9"


@dataclass(frozen=True)
class Canvas:
    """The resolved geometry and timing for one generation."""

    width: int
    height: int
    frames: int
    fps: int
    duration_seconds: float
    quality: str
    aspect_ratio: str
    steps: int
    #: True when the requested duration was snapped to the frame grid.
    duration_adjusted: bool
    #: True when the frame count sits outside the checkpoint's trained range.
    outside_trained_range: bool

    def as_metadata(self) -> dict:
        """What the job result reports back, so a caller never has to guess."""
        data = {
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            "durationSeconds": round(self.duration_seconds, 4),
            "quality": self.quality,
            "aspectRatio": self.aspect_ratio,
            "steps": self.steps,
        }
        if self.duration_adjusted:
            data["durationAdjusted"] = True
        if self.outside_trained_range:
            data["outsideTrainedRange"] = True
        return data


def resolve_canvas(
    *,
    quality: str = DEFAULT_QUALITY,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    duration_seconds: float = 5.0,
    steps: int | None = None,
) -> Canvas:
    """Turn a product request into the exact geometry the model will run.

    The caller asks for "standard 16:9, about 5 seconds". What comes back is
    1024x576 at 124 frames - and the *actual* duration those frames represent, which is
    5.1667s, not 5. Reporting the real number rather than the requested one is the whole
    point: a caller that needs to label a clip should never have to re-derive it.
    """
    tier = QUALITY_TIERS.get(quality)
    if tier is None:
        raise ValueError(f"unknown quality tier {quality!r}")
    ratio = ASPECT_RATIOS.get(aspect_ratio)
    if ratio is None:
        raise ValueError(f"unknown aspect ratio {aspect_ratio!r}")

    width, height = adapt_canvas(ratio, 1.0)

    # A tier below the model's native canvas scales the whole canvas down, preserving the
    # aspect, then re-rounds. Scaling by area keeps the ratio honest at every tier.
    if tier.max_pixels is not None and width * height > tier.max_pixels:
        scale = math.sqrt(tier.max_pixels / (width * height))
        width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

    requested_frames = max(MIN_FRAMES, round(duration_seconds * FPS))
    frames = align_frame_count(requested_frames)
    if frames > MAX_FRAMES:
        raise ValueError(
            f"{duration_seconds}s needs {frames} frames; the model accepts at most {MAX_FRAMES}"
        )

    return Canvas(
        width=width,
        height=height,
        frames=frames,
        fps=FPS,
        duration_seconds=frames / FPS,
        quality=quality,
        aspect_ratio=aspect_ratio,
        steps=tier.steps if steps is None else steps,
        duration_adjusted=frames != requested_frames,
        outside_trained_range=not (TRAINED_MIN_FRAMES <= frames <= TRAINED_MAX_FRAMES),
    )


def canvas_from_dimensions(width: int, height: int, frames: int, steps: int) -> Canvas:
    """The legacy path: explicit width/height/frames, reported in the new metadata shape.

    Used by the backwards-compatibility adapter so an old-style request produces the same
    response envelope as a new one, without being re-derived through the tier table.
    """
    aligned = align_frame_count(frames)
    return Canvas(
        width=width,
        height=height,
        frames=aligned,
        fps=FPS,
        duration_seconds=aligned / FPS,
        quality=_nearest_tier(width, height),
        aspect_ratio=_nearest_ratio(width, height),
        steps=steps,
        duration_adjusted=aligned != frames,
        outside_trained_range=not (TRAINED_MIN_FRAMES <= aligned <= TRAINED_MAX_FRAMES),
    )


def _nearest_tier(width: int, height: int) -> str:
    pixels = width * height
    return min(
        QUALITY_TIERS,
        key=lambda name: abs((QUALITY_TIERS[name].max_pixels or MAX_PIXELS) - pixels),
    )


def _nearest_ratio(width: int, height: int) -> str:
    ratio = width / height
    return min(ASPECT_RATIOS, key=lambda name: abs(ASPECT_RATIOS[name] - ratio))


def verify_against_comfyui(module) -> list[str]:
    """Re-derive the constants from the installed node module and report any drift.

    Called at boot when ComfyUI is importable. The constants above are a copy, and a copy
    of somebody else's arithmetic is a liability unless something checks it: a base-image
    bump that changed the canvas rule would otherwise be discovered as bad output rather
    than as a startup warning.
    """
    problems: list[str] = []
    expected = {
        "CANVAS_MULTIPLE": CANVAS_MULTIPLE,
        "BASE_SHORT_EDGE": BASE_SHORT_EDGE,
        "MAX_PIXELS": MAX_PIXELS,
        "FPS": FPS,
        "REF_IMAGE_SHORT_EDGE": REF_IMAGE_SHORT_EDGE,
    }
    for name, value in expected.items():
        actual = getattr(module, name, None)
        if actual is None:
            problems.append(f"{name} is missing from the node module")
        elif actual != value:
            problems.append(f"{name} is {actual} in ComfyUI but {value} here")

    adapt = getattr(module, "adapt_canvas", None)
    if adapt is not None:
        for ratio_name, ratio in ASPECT_RATIOS.items():
            theirs = tuple(adapt(ratio, 1.0))
            ours = adapt_canvas(ratio, 1.0)
            if theirs != ours:
                problems.append(f"adapt_canvas({ratio_name}) is {theirs} in ComfyUI but {ours} here")

    align = getattr(module, "align_frame_count", None)
    if align is not None:
        for probe in (5, 6, 100, 124, 125, 300):
            if align(probe) != align_frame_count(probe):
                problems.append(
                    f"align_frame_count({probe}) is {align(probe)} in ComfyUI "
                    f"but {align_frame_count(probe)} here"
                )
    return problems
