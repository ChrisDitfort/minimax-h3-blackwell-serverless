"""The one ComfyUI node this package registers, and only on a sequence-parallel rank.

Ranks above 0 are shadows: they exist to hold up their half of the collectives, not to
produce a video. They must still execute everything up to and including both VAE decodes,
because those are where the collectives are - but they must not save anything.

That matters for more than tidiness. Under Confidential Generation the finished MP4 is
plaintext until the handler encrypts it, and the handler only ever sees rank 0's. A shadow
that ran SaveVideo would leave a second, unencrypted, unaccounted-for copy of the customer's
video on the container's disk. So the handler rewrites the shadow's copy of the workflow to
end at this node instead: it consumes the decoded frames and audio, forces their production,
and writes nothing anywhere.

The caller's workflow is never touched. Only the internally generated shadow copy refers to
this node, and it only exists in processes the handler launched as group members.
"""

from __future__ import annotations

SINK_CLASS_TYPE = "H3ParallelSink"


class H3ParallelSink:
    """An output node that produces the tensors and then drops them."""

    @classmethod
    def INPUT_TYPES(cls):
        # Everything optional: a shadow graph wires up whichever decode outputs its
        # workflow actually has, and a text-to-video graph with no audio is still valid.
        return {"required": {}, "optional": {"images": ("IMAGE",), "audio": ("AUDIO",)}}

    RETURN_TYPES = ()
    FUNCTION = "sink"
    OUTPUT_NODE = True
    CATEGORY = "h3_parallel"
    DESCRIPTION = "Internal: forces decode on a sequence-parallel shadow rank and discards it."

    def sink(self, images=None, audio=None):
        # Deliberately no reference kept: the tensors are freed as soon as this returns.
        del images, audio
        return {}


NODE_CLASS_MAPPINGS = {SINK_CLASS_TYPE: H3ParallelSink}
NODE_DISPLAY_NAME_MAPPINGS = {SINK_CLASS_TYPE: "H3 Parallel Sink (internal)"}
