"""Build the workflow a shadow rank runs, from the workflow the caller submitted.

A shadow rank (any rank above 0) exists to execute its half of the DiT and the video VAE
so that rank 0's collectives have a partner. It must therefore run every node up to and
including both VAE decodes, in the same order, on the same inputs. It must NOT run the
tail: muxing an MP4 and writing it to disk is pure CPU work that produces a second
plaintext copy of the customer's video, which under Confidential Generation is exactly the
thing the whole design exists to avoid.

So the tail is pruned and replaced with H3ParallelSink, an output node that consumes the
decoded tensors and discards them. Pruning is done by repeatedly removing *unreferenced*
terminal nodes, never by cutting a link, so the result cannot contain a dangling reference
that ComfyUI would reject.

Pure dict manipulation: no torch, no ComfyUI, no network. Tested in tests/test_shadow.py.
"""

from __future__ import annotations

import copy

from .nodes import SINK_CLASS_TYPE

#: Nodes that turn finished tensors into files or streams. None of them ever runs a
#: collective, so a shadow that skips them stays perfectly in step with rank 0.
TERMINAL_CLASS_TYPES = frozenset(
    {
        "SaveVideo",
        "SaveWEBM",
        "SaveImage",
        "SaveImageWebsocket",
        "SaveAnimatedWEBP",
        "SaveAnimatedPNG",
        "SaveAudio",
        "SaveAudioMP3",
        "SaveAudioOpus",
        "SaveGLB",
        "PreviewImage",
        "PreviewAudio",
        "CreateVideo",
        "VHS_VideoCombine",
    }
)

#: The nodes whose execution the shadow exists to reach. Each one gets its own sink so
#: that a workflow with several decodes still runs all of them.
DECODE_CLASS_TYPES = {
    "VAEDecode": "images",
    "VAEDecodeTiled": "images",
    "VAEDecodeAudio": "audio",
}

SINK_ID_PREFIX = "h3-shadow-sink-"


def _is_link(value) -> bool:
    """An API-format link is ["<source node id>", <output slot int>]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _referenced_node_ids(workflow: dict) -> set[str]:
    referenced: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if _is_link(value):
                referenced.add(str(value[0]))
    return referenced


def prune_terminal_nodes(workflow: dict) -> tuple[dict, list[str]]:
    """Drop file-producing tail nodes, leaves first, until none are left.

    Only ever removes a node nothing else points at, so no surviving node can end up
    referencing something that is gone. Repeats because removing SaveVideo is what makes
    CreateVideo removable.
    """
    pruned = dict(workflow)
    removed: list[str] = []

    while True:
        referenced = _referenced_node_ids(pruned)
        candidates = [
            node_id
            for node_id, node in pruned.items()
            if isinstance(node, dict)
            and node.get("class_type") in TERMINAL_CLASS_TYPES
            and node_id not in referenced
        ]
        if not candidates:
            return pruned, removed
        for node_id in candidates:
            pruned.pop(node_id, None)
            removed.append(node_id)


def build_shadow_workflow(workflow: dict) -> tuple[dict, str]:
    """Return (workflow for a shadow rank, one-line description of what was done).

    Falls back to an exact copy of the caller's workflow when the decode nodes cannot be
    identified. That is slower and leaves a file for the handler to delete, but it can
    never deadlock rank 0 - which is the failure that actually matters.
    """
    shadow = copy.deepcopy(workflow)

    decoders = [
        (node_id, DECODE_CLASS_TYPES[node["class_type"]])
        for node_id, node in shadow.items()
        if isinstance(node, dict) and node.get("class_type") in DECODE_CLASS_TYPES
    ]
    if not decoders:
        return shadow, "verbatim (no VAE decode node found to anchor a sink)"

    shadow, removed = prune_terminal_nodes(shadow)

    for node_id, socket in decoders:
        shadow[f"{SINK_ID_PREFIX}{node_id}"] = {
            "class_type": SINK_CLASS_TYPE,
            "inputs": {socket: [node_id, 0]},
        }

    return shadow, (
        f"sink-terminated (decodes={len(decoders)} pruned={len(removed)} "
        f"nodes={len(shadow)})"
    )
