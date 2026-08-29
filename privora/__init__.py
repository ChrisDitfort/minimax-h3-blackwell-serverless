"""PrivoraVideo's product abstraction over MiniMax H3.

The layering this package exists to enforce:

    PrivoraVideo request  ->  privora.request.parse()
                          ->  privora.workflows build the graph
                          ->  ComfyUI executes it

Nothing above this package knows about ComfyUI node names, checkpoint filenames, the
17k+5 frame grid or `<Picture 1>` syntax. That is the whole point: ComfyUI can be
upgraded, nodes replaced, or H3 swapped for another model without the control plane or
the frontend changing.
"""

from __future__ import annotations

from . import canvas, errors, prompt, references, request

__all__ = ["canvas", "errors", "prompt", "references", "request"]
