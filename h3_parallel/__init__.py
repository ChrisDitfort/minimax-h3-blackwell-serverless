"""Experimental multi-GPU execution for the MiniMax H3 RunPod worker.

This package is two things at once, deliberately:

  * a plain importable library (config / ulysses / collectives / patches), so the
    sharding algebra can be unit-tested on a laptop with no GPU and no ComfyUI; and
  * a ComfyUI custom node, because dropping it in `custom_nodes/` is the one supported
    hook that runs inside the ComfyUI process after its attention backend is chosen and
    before any model is loaded.

Importing it is free. It only does something when the process was launched as a member of
a sequence-parallel group - `H3_GPU_MODE=dual` with `H3_SP_RANK` set, which only the
handler does. That guard is what keeps `import h3_parallel` inert in the test suite and
keeps single-GPU mode identical to the known-good image.

ComfyUI expects NODE_CLASS_MAPPINGS from a custom node. In single-GPU mode this one
registers none at all - the workflow the Cloudflare Worker submits must keep working
byte-for-byte. On a sequence-parallel rank it registers exactly one internal node, the
shadow-rank sink (see nodes.py), which no caller-supplied workflow ever references.
"""

from __future__ import annotations

import os

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _running_inside_comfyui() -> bool:
    """True only in a ComfyUI process the handler launched as a group member."""
    return bool(os.environ.get("H3_SP_RANK", "").strip())


if _running_inside_comfyui():
    from . import nodes, runtime

    NODE_CLASS_MAPPINGS.update(nodes.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(nodes.NODE_DISPLAY_NAME_MAPPINGS)

    # The status route goes up first, and on purpose. ComfyUI swallows exceptions raised
    # by a custom node (nodes.load_custom_node logs IMPORT FAILED and carries on), so a
    # broken dual-GPU setup would otherwise leave a perfectly healthy-looking ComfyUI
    # quietly running the unpatched single-GPU path and reporting it as a 2-GPU benchmark.
    # Registering first means /h3/gpu can always be asked what actually happened, and the
    # handler refuses to serve when the answer is "not ready".
    runtime.register_status_route()
    try:
        runtime.boot()
    except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
        runtime.record_boot_failure(error)
        raise
