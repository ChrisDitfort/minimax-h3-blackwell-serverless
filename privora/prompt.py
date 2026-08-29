"""Compile a PrivoraVideo request into the prompt text H3 actually wants.

The contract this exists to protect: **the frontend never writes H3 syntax.** A caller
sends prose plus structured metadata - which reference is the character, which video
carries the motion, what the camera should do - and this module produces the
``<Picture 1>`` / ``<Video 1>`` / ``<Audio 2>`` tagging the model was trained on.

Three properties, all tested:

* **Deterministic.** The same request compiles to the same string, every time. A prompt
  that varies run to run makes a seed meaningless.
* **Ordinals match the conditioning.** Tags are assigned by references.assign_ordinals(),
  which mirrors the node's own presentation order. A mismatch here would point every
  instruction at the wrong reference.
* **User content is never structural.** A prompt containing the literal text
  ``<Picture 1>`` must not be able to impersonate a reference tag the caller did not
  supply. See `_neutralise_tags`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .references import Reference, ReferenceSet

#: Anything shaped like an H3 reference tag. Matched case-insensitively and tolerant of
#: internal spacing, because the point is to catch impersonation, not to be tidy.
_TAG_PATTERN = re.compile(r"<\s*(picture|video|audio)\s*(\d+)\s*>", re.IGNORECASE)

#: Camera vocabulary. These compile to prose because H3 has no camera conditioning input -
#: the model reads them as language, the same way a human director would. Inventing a
#: latent "camera strength" control would be dishonest about what the model does.
SHOTS = {
    "extreme_wide": "an extreme wide shot",
    "wide": "a wide shot",
    "medium_wide": "a medium wide shot",
    "medium": "a medium shot",
    "medium_close": "a medium close-up",
    "close": "a close-up",
    "extreme_close": "an extreme close-up",
}

MOVEMENTS = {
    "static": "a locked-off camera",
    "pan": "the camera panning",
    "tilt": "the camera tilting",
    "dolly": "the camera dollying",
    "truck": "the camera trucking sideways",
    "orbit": "the camera orbiting the subject",
    "crane": "a crane move",
    "handheld": "handheld camera movement",
    "zoom": "a zoom",
    "push_in": "the camera pushing in",
    "pull_out": "the camera pulling out",
}

STRENGTHS = {"subtle": "subtle", "moderate": "moderate", "strong": "pronounced"}
SPEEDS = {"slow": "slowly", "medium": "at a steady pace", "fast": "quickly"}

VISUAL_STYLES = {
    "cinematic": "cinematic",
    "documentary": "documentary",
    "animation": "animated",
    "anime": "anime",
    "photoreal": "photorealistic",
    "vintage": "vintage film",
    "noir": "film noir",
}

LIGHTING = {
    "natural": "natural light",
    "golden_hour": "golden-hour light",
    "high_key": "high-key lighting",
    "low_key": "low-key lighting",
    "neon": "neon lighting",
    "candlelit": "candlelight",
    "overcast": "overcast light",
}

MOTION_STYLES = {
    "natural": "natural motion",
    "slow_motion": "slow motion",
    "timelapse": "time-lapse",
    "hyperlapse": "hyperlapse",
}


@dataclass(frozen=True)
class CompiledPrompt:
    """The compiled result, kept structured so tests can assert on each part."""

    text: str
    #: reference tag -> the role it was given. Diagnostic only; never logged with the text.
    tag_roles: dict
    #: True when the caller's own prose contained something shaped like a reference tag.
    neutralised_user_tags: bool

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


def _neutralise_tags(prompt: str) -> tuple[str, bool]:
    """Stop a user's prose from impersonating a reference tag.

    A caller who writes "make it look like <Picture 3>" when they supplied two images
    would otherwise point the model at a reference that does not exist. Structural tags
    are ours to emit; anything tag-shaped arriving in user text is rewritten to a form
    that reads the same to a person and means nothing to the tokenizer.
    """
    if not _TAG_PATTERN.search(prompt):
        return prompt, False
    return _TAG_PATTERN.sub(lambda m: f"{m.group(1).lower()} {m.group(2)}", prompt), True


def _reference_clause(reference: Reference) -> str | None:
    """One instruction binding a tag to its role, or None for an unguided reference."""
    guidance = reference.guidance
    if not guidance:
        return None
    return f"Use {guidance} from <{reference.tag}>."


def compile_camera(camera: dict | None) -> str:
    """Camera metadata -> one sentence of prose, or "" when nothing usable was given."""
    if not camera:
        return ""

    shot = SHOTS.get(str(camera.get("shot", "")).lower())
    movement = MOVEMENTS.get(str(camera.get("movement", "")).lower())
    strength = STRENGTHS.get(str(camera.get("strength", "")).lower())
    speed = SPEEDS.get(str(camera.get("speed", "")).lower())

    if not shot and not movement:
        return ""

    parts: list[str] = []
    if shot:
        parts.append(f"Filmed as {shot}")
    if movement:
        described = movement
        if strength:
            # "the camera orbiting" -> "pronounced camera orbiting" reads badly, so the
            # qualifier attaches to the movement noun instead.
            described = described.replace("the camera ", f"{strength} camera ", 1)
            if described == movement:
                described = f"{strength} {movement}"
        if speed:
            described = f"{described} {speed}"
        parts.append(f"with {described}" if shot else f"Filmed with {described}")

    return ", ".join(parts) + "."


def compile_style(style: dict | None) -> str:
    if not style:
        return ""
    parts = [
        VISUAL_STYLES.get(str(style.get("visual", "")).lower(), ""),
        LIGHTING.get(str(style.get("lighting", "")).lower(), ""),
        MOTION_STYLES.get(str(style.get("motion", "")).lower(), ""),
    ]
    chosen = [part for part in parts if part]
    if not chosen:
        return ""
    return f"Style: {', '.join(chosen)}."


def compile_prompt(
    user_prompt: str,
    references: ReferenceSet | None = None,
    *,
    camera: dict | None = None,
    style: dict | None = None,
) -> CompiledPrompt:
    """Assemble the final prompt: user prose, then reference bindings, then direction.

    Order is deliberate. The user's own words come first so they carry the most weight;
    the mechanical parts follow as separate sentences rather than being woven in, which
    keeps the compiled output readable and makes a wrong tag obvious in a diff.
    """
    text, neutralised = _neutralise_tags((user_prompt or "").strip())
    if text and text[-1] not in ".!?":
        # The reference clauses that follow are separate sentences; without this the
        # user's last word runs straight into "Use the character from...".
        text += "."
    sentences: list[str] = [text] if text else []
    tag_roles: dict[str, str] = {}

    if references is not None and not references.is_empty():
        for reference in references.all:
            tag_roles[f"<{reference.tag}>"] = reference.role
            clause = _reference_clause(reference)
            if clause:
                sentences.append(clause)
            if reference.type == "video" and reference.soundtrack is not None:
                soundtrack = reference.soundtrack
                tag_roles[f"<{soundtrack.tag}>"] = soundtrack.role
                clause = _reference_clause(soundtrack)
                if clause:
                    sentences.append(clause)

    camera_text = compile_camera(camera)
    if camera_text:
        sentences.append(camera_text)

    style_text = compile_style(style)
    if style_text:
        sentences.append(style_text)

    return CompiledPrompt(
        text=" ".join(sentences).strip(),
        tag_roles=tag_roles,
        neutralised_user_tags=neutralised,
    )
