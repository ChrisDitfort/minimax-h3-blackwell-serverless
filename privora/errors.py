"""Structured, sanitised errors for the PrivoraVideo worker API.

Two rules, both learned from the privacy audit rather than invented:

1.  A caller gets a stable machine-readable ``code`` and a message written for a person.
    Neither ever contains the prompt, a reference filename, a local path, or key material.
2.  Anything genuinely diagnostic that *might* carry user content goes in ``internal``,
    which is logged and never returned.

The split matters because the same exception object ends up in three places - the
container log, the RunPod job result and the progress callback - and only the first of
those is ours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# -- validation -------------------------------------------------------------------------
UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
INVALID_ASPECT_RATIO = "INVALID_ASPECT_RATIO"
INVALID_QUALITY = "INVALID_QUALITY"
INVALID_DURATION = "INVALID_DURATION"
INVALID_SEED = "INVALID_SEED"
INVALID_STEPS = "INVALID_STEPS"
UNKNOWN_FIELD = "UNKNOWN_FIELD"
MISSING_PROMPT = "MISSING_PROMPT"
MISSING_FRAME = "MISSING_FRAME"

# -- references -------------------------------------------------------------------------
INVALID_REFERENCE_COUNT = "INVALID_REFERENCE_COUNT"
INVALID_REFERENCE_TYPE = "INVALID_REFERENCE_TYPE"
INVALID_REFERENCE_ROLE = "INVALID_REFERENCE_ROLE"
INVALID_REFERENCE_DURATION = "INVALID_REFERENCE_DURATION"
REFERENCE_PREPROCESSING_FAILED = "REFERENCE_PREPROCESSING_FAILED"

# -- execution --------------------------------------------------------------------------
MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
GENERATION_FAILED = "GENERATION_FAILED"
OUT_OF_MEMORY = "OUT_OF_MEMORY"
ENCODE_FAILED = "ENCODE_FAILED"
CONFIDENTIAL_ENCRYPTION_FAILED = "CONFIDENTIAL_ENCRYPTION_FAILED"
UPLOAD_FAILED = "UPLOAD_FAILED"

#: Codes a caller can fix by changing the request. Everything else is ours to fix.
CLIENT_ERRORS = frozenset(
    {
        UNSUPPORTED_MODE,
        INVALID_ASPECT_RATIO,
        INVALID_QUALITY,
        INVALID_DURATION,
        INVALID_SEED,
        INVALID_STEPS,
        UNKNOWN_FIELD,
        MISSING_PROMPT,
        MISSING_FRAME,
        INVALID_REFERENCE_COUNT,
        INVALID_REFERENCE_TYPE,
        INVALID_REFERENCE_ROLE,
        INVALID_REFERENCE_DURATION,
    }
)


@dataclass
class PrivoraError(Exception):
    """A failure with a stable code and a message safe to hand back to a caller."""

    code: str
    message: str
    #: Machine-readable specifics - counts, limits, indices. Never user content.
    details: dict = field(default_factory=dict)
    #: Diagnostic text that may contain anything. Logged, never returned.
    internal: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def is_client_error(self) -> bool:
        return self.code in CLIENT_ERRORS

    def as_response(self) -> dict:
        """The shape that goes into the RunPod job result."""
        payload = {"error": self.message, "errorCode": self.code}
        if self.details:
            payload["errorDetails"] = self.details
        return payload

    def as_log_line(self) -> str:
        """What the container logs. Includes `internal`; the response never does."""
        parts = [f"code={self.code}", f"message={self.message!r}"]
        if self.details:
            parts.append(f"details={self.details}")
        if self.internal:
            parts.append(f"internal={self.internal!r}")
        return " ".join(parts)
