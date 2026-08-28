"""ComfyUI custom-node shim that loads /opt/serverless/h3_parallel.

The package lives with the handler rather than in custom_nodes so there is exactly one
copy of it in the image and `import h3_parallel` means the same thing in both processes.
ComfyUI only scans custom_nodes, so this single file is the bridge.

Inert unless the process was launched as a sequence-parallel rank: h3_parallel does
nothing at import time without H3_SP_RANK, and in single-GPU mode it registers no nodes,
patches nothing, and opens no sockets. That is what keeps H3_GPU_MODE=single behaving
exactly like the known-good image.
"""

import os
import sys

SERVERLESS_DIR = os.environ.get("H3_SERVERLESS_DIR", "/opt/serverless")

if SERVERLESS_DIR not in sys.path:
    sys.path.insert(0, SERVERLESS_DIR)

try:
    import h3_parallel

    NODE_CLASS_MAPPINGS = h3_parallel.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = h3_parallel.NODE_DISPLAY_NAME_MAPPINGS
except Exception as error:
    # Re-raised so ComfyUI logs the traceback. It will still start - load_custom_node
    # catches this - which is precisely why the handler independently verifies /h3/gpu
    # before it accepts a job in dual mode.
    print(
        f"[h3_parallel_boot] FAILED to initialise multi-GPU support from {SERVERLESS_DIR}: "
        f"{type(error).__name__}: {error}",
        flush=True,
    )
    raise
