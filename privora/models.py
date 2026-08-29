"""Which checkpoint, which LoRA, how many steps - and what is actually installed.

The product exposes three generation modes. The model exposes two checkpoints and a set of
distilled LoRAs. This module is the only place that knows how one maps onto the other, so
the control plane never learns a checkpoint filename.

    quality      base checkpoint, 20 steps
    turbo        8-step distilled LoRA
    turbo_fast   4-step distilled LoRA

Verified against the Comfy-Org/MiniMax-H3 repository rather than assumed:

* FL2VA has both an 8-step and a 4-step Turbo LoRA (v1.0).
* **Ref2VA has only a 4-step LoRA, at v0.1.** There is no 8-step Ref2VA Turbo. The brief
  asked for this to be checked rather than assumed, and the answer is that `turbo` is not
  available for reference generation at all.
* The LoRAs did not exist at the model revision the DiT weights are pinned to. They are
  therefore pinned to their own, later revision - deliberately, so the diffusion weights
  used by every existing benchmark stay byte-identical.

Availability is decided by looking at the filesystem, not by this table. `/capabilities`
must describe the image that was built, not the image we intended to build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import errors

FL2VA = "fl2va"
REF2VA = "ref2va"

QUALITY = "quality"
TURBO = "turbo"
TURBO_FAST = "turbo_fast"

GENERATION_MODES = (QUALITY, TURBO, TURBO_FAST)
DEFAULT_GENERATION_MODE = QUALITY

#: JSON-facing spelling. The wire format is camelCase; the internals are snake_case.
GENERATION_MODE_ALIASES = {
    "quality": QUALITY,
    "turbo": TURBO,
    "turbofast": TURBO_FAST,
    "turbo_fast": TURBO_FAST,
}

CHECKPOINTS = {
    FL2VA: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    REF2VA: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}


@dataclass(frozen=True)
class Acceleration:
    """One (family, generation mode) configuration."""

    family: str
    mode: str
    steps: int
    lora: str | None
    #: What goes in job metadata. "none" for the base path, "turbo_lora" otherwise.
    kind: str
    #: Set when the LoRA carries a caveat worth reporting rather than hiding.
    note: str | None = None


ACCELERATIONS: dict[tuple[str, str], Acceleration] = {
    (FL2VA, QUALITY): Acceleration(FL2VA, QUALITY, 20, None, "none"),
    (FL2VA, TURBO): Acceleration(
        FL2VA, TURBO, 8, "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "turbo_lora"
    ),
    (FL2VA, TURBO_FAST): Acceleration(
        FL2VA, TURBO_FAST, 4,
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", "turbo_lora",
        note="Distilled for the 768p canvas; behaviour at smaller tiers is unvalidated.",
    ),
    (REF2VA, QUALITY): Acceleration(REF2VA, QUALITY, 20, None, "none"),
    (REF2VA, TURBO_FAST): Acceleration(
        REF2VA, TURBO_FAST, 4,
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "turbo_lora",
        note="v0.1 - an early release, unlike the v1.0 FL2VA LoRAs.",
    ),
    # (REF2VA, TURBO) is absent on purpose: no 8-step Ref2VA LoRA has been published.
}


class ModelInventory:
    """What is actually on disk in this image.

    Constructed once at boot from the real model directories. Everything the worker says
    about its own capability flows from here, so an image built without the Ref2VA
    checkpoint reports reference modes as unavailable instead of failing at job time.
    """

    def __init__(self, diffusion_dir: str | None = None, lora_dir: str | None = None):
        self.diffusion_dir = diffusion_dir
        self.lora_dir = lora_dir
        self._present: set[str] = set()
        if diffusion_dir and os.path.isdir(diffusion_dir):
            self._present |= set(os.listdir(diffusion_dir))
        if lora_dir and os.path.isdir(lora_dir):
            self._present |= set(os.listdir(lora_dir))

    @classmethod
    def from_names(cls, names) -> "ModelInventory":
        """A synthetic inventory, for tests and for describing a planned image."""
        inventory = cls()
        inventory._present = set(names)
        return inventory

    def has_family(self, family: str) -> bool:
        return CHECKPOINTS.get(family, "") in self._present

    def has_acceleration(self, family: str, mode: str) -> bool:
        acceleration = ACCELERATIONS.get((family, mode))
        if acceleration is None:
            return False
        if not self.has_family(family):
            return False
        return acceleration.lora is None or acceleration.lora in self._present

    def available_modes(self, family: str) -> list[str]:
        return [mode for mode in GENERATION_MODES if self.has_acceleration(family, mode)]

    def resolve(self, family: str, mode: str) -> Acceleration:
        """Pick the configuration, or explain precisely why it is unavailable."""
        acceleration = ACCELERATIONS.get((family, mode))
        if acceleration is None:
            raise errors.PrivoraError(
                errors.UNSUPPORTED_MODE,
                f"generationMode {_wire(mode)!r} is not available for this kind of "
                f"generation. Available: {', '.join(_wire(m) for m in self.available_modes(family))}.",
                {"family": family, "generationMode": _wire(mode),
                 "available": [_wire(m) for m in self.available_modes(family)]},
            )
        if not self.has_family(family):
            raise errors.PrivoraError(
                errors.MODEL_LOAD_FAILED,
                "This worker was built without the model needed for that mode.",
                {"family": family, "checkpoint": CHECKPOINTS.get(family)},
            )
        if acceleration.lora is not None and acceleration.lora not in self._present:
            raise errors.PrivoraError(
                errors.MODEL_LOAD_FAILED,
                f"generationMode {_wire(mode)!r} needs an acceleration weight this worker "
                "was not built with.",
                {"family": family, "generationMode": _wire(mode)},
            )
        return acceleration

    def describe(self) -> dict:
        """The capabilities fragment. Describes the built image, never the intended one."""
        return {
            "models": {family: self.has_family(family) for family in CHECKPOINTS},
            "generationModes": {
                _wire(mode): any(self.has_acceleration(f, mode) for f in CHECKPOINTS)
                for mode in GENERATION_MODES
            },
            "turbo": {
                family: {
                    f"{ACCELERATIONS[(family, mode)].steps}step": self.has_acceleration(family, mode)
                    for mode in (TURBO, TURBO_FAST)
                    if (family, mode) in ACCELERATIONS
                }
                for family in CHECKPOINTS
            },
            "byFamily": {
                family: [_wire(mode) for mode in self.available_modes(family)]
                for family in CHECKPOINTS
            },
        }


def _wire(mode: str) -> str:
    """Internal name -> the spelling used on the wire."""
    return "turboFast" if mode == TURBO_FAST else mode


def parse_generation_mode(value) -> str:
    if value is None:
        return DEFAULT_GENERATION_MODE
    key = str(value).strip().lower().replace("-", "_")
    resolved = GENERATION_MODE_ALIASES.get(key.replace("_", ""))
    if resolved is None:
        resolved = GENERATION_MODE_ALIASES.get(key)
    if resolved is None:
        raise errors.PrivoraError(
            errors.UNSUPPORTED_MODE,
            "generationMode must be one of quality, turbo, turboFast.",
            {"supplied": str(value), "supported": ["quality", "turbo", "turboFast"]},
        )
    return resolved


def steps_to_generation_mode(family: str, steps: int) -> str | None:
    """Map a legacy `steps` value onto a generation mode, or None if it maps to nothing.

    Deliberately conservative. A legacy caller sending `steps: 20` gets the base workflow,
    which is what it has always got. Any other value keeps running on the base checkpoint
    at that step count rather than silently selecting a distilled LoRA: a 4-step request
    against base weights produces a poor video, but a 4-step request that quietly swaps the
    model would change what the caller is paying for without being asked.
    """
    base = ACCELERATIONS.get((family, QUALITY))
    if base is not None and steps == base.steps:
        return QUALITY
    return None
