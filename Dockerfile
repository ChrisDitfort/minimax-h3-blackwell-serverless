# MiniMax H3 FL2VA - RunPod Serverless worker for RTX PRO 6000 Blackwell (sm_120).
#
# The base image is a *Pod* image: its ENTRYPOINT (/entrypoint.sh) runs a driver
# preflight, downloads ~60 GB of H3 weights at runtime, bootstraps SageAttention and
# then execs /start.sh, which brings up SSH, JupyterLab, FileBrowser and a ComfyUI
# bound to 0.0.0.0. None of that belongs in a Serverless worker, so this image
# replaces ENTRYPOINT outright (overriding CMD alone is NOT enough - the base sets
# ENTRYPOINT, so CMD would just become its arguments).
#
# Everything that makes the base Blackwell-ready is inherited untouched:
#   * PyTorch 2.10.0+cu130 / CUDA 13 in system python3.12 dist-packages
#   * sageattn3 1.0.0, enabled by the COMFY_SAGE_ATTENTION3=1 env var
#   * ComfyUI pinned at dec5d945 with native MiniMax H3 support, at /opt/comfyui-baked
#   * ComfyUI-Pixaroma H3 nodes, already baked into custom_nodes
FROM ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-blackwell

ARG H3_BUILD_SOURCE_COMMIT=unknown
ARG H3_BUILD_IMAGE_TAG=unknown
ARG H3_BUILD_ID=local

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ComfyUI lives at /opt/comfyui-baked in the image; the base only copies it into
# /workspace/runpod-slim at Pod runtime. We run it in place, which avoids duplicating a
# ~550 MB tree and keeps the models safe from anything RunPod might mount over /workspace.
# The ephemeral input/output working directories do use the conventional
# /workspace/runpod-slim/ComfyUI paths: ComfyUI is pointed at them explicitly via
# --input-directory/--output-directory and the handler re-creates them at boot, so they
# still work if a volume is mounted there.
ENV COMFY_DIR=/opt/comfyui-baked \
    COMFY_PORT=8188 \
    COMFY_INPUT_DIR=/workspace/runpod-slim/ComfyUI/input \
    COMFY_OUTPUT_DIR=/workspace/runpod-slim/ComfyUI/output \
    COMFY_TEMP_DIR=/tmp/comfy-temp \
    H3_OUTPUT_MODE=base64 \
    PYTHONUNBUFFERED=1

# The Serverless handler and the model layers appended after this build both use system
# python3.12, which is where the base image installed torch/cu130 and sageattn3. The
# runtime venv the Pod entrypoint creates is only `--system-site-packages` over this same
# interpreter, so it would add nothing here.
#   boto3 + cryptography are needed only by H3_OUTPUT_MODE=r2 (encrypted R2 upload);
#   they are cheap and baking them keeps that path a pure config switch.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        runpod \
        requests \
        websocket-client \
        boto3 \
        cryptography \
 && python3 -c "import runpod, requests, websocket, boto3, cryptography; print('serverless deps OK')"

# Model directories, plus the compatibility symlink for the FL2VA diffusion model.
#
# The weights are appended as separate layers after this image is built (see
# .github/workflows/build.yml), because 42 GB will not fit through a BuildKit build.
# The real file lands at models/diffusion_models/<name>.safetensors so the existing
# Cloudflare workflow's bare filename resolves; the bundled Pixaroma workflows instead
# reference "h3/<name>.safetensors", so we point that at the same bytes. ComfyUI walks
# model dirs with os.walk(followlinks=True) and get_full_path() accepts symlinks, so both
# spellings work and nothing is stored twice.
RUN mkdir -p "$COMFY_DIR/models/diffusion_models/h3" \
             "$COMFY_DIR/models/text_encoders" \
             "$COMFY_DIR/models/vae" \
             "$COMFY_DIR/models/loras" \
             /workspace/runpod-slim/ComfyUI/input \
             /workspace/runpod-slim/ComfyUI/output \
             /tmp/comfy-temp \
 && ln -sfn ../minimax_h3_fl2va_pruned_int8_convrot.safetensors \
      "$COMFY_DIR/models/diffusion_models/h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

# Pre-apply ComfyUI's SQLite migrations at build time.
#
# ComfyUI keeps an asset database at <root>/user/comfyui.db and runs alembic against it on
# every startup. A scale-to-zero endpoint gets a fresh container filesystem every time, so
# that database never exists and the whole 0001..0006 chain is replayed on every single
# cold start - visible in the logs as "Database upgraded from None to 0006_add_loader_path".
# Baking it means startup finds the schema already at head.
#
# Safe because ComfyUI's _init_file_db() compares the current revision against head and
# skips the upgrade entirely when they match. The file stays writable and un-mounted, so if
# a future base image ships newer migrations the runtime simply applies the remainder.
#
# Driving alembic directly, rather than booting ComfyUI, keeps this GPU-free: alembic_db/env.py
# imports only app.database.models, which pulls SQLAlchemy and nothing else. The second
# command fails the build if the schema did not actually get stamped, so this can never
# quietly degrade into a no-op.
RUN cd "$COMFY_DIR" \
 && mkdir -p user \
 && python3 -c "from alembic import command; from alembic.config import Config; c = Config('alembic.ini'); c.set_main_option('script_location', 'alembic_db'); command.upgrade(c, 'head')" \
 && python3 -c "import sqlite3; r = sqlite3.connect('user/comfyui.db').execute('select version_num from alembic_version').fetchone(); assert r and r[0], 'alembic_version is empty - migrations did not run'; print('ComfyUI DB baked at revision', r[0])"

# artifacts.py is the model-independent privacy layer (privacy modes, the encrypted
# container format, plaintext hygiene, redaction). It is a sibling module rather than more
# of handler.py because nothing in it is H3-specific: a second model reuses it as-is.
# handler.py runs from /opt/serverless, so a plain `import artifacts` resolves.
COPY handler.py /opt/serverless/handler.py
COPY artifacts.py /opt/serverless/artifacts.py

# Experimental multi-GPU execution. The package lives next to the handler so both the
# handler process and every ComfyUI rank import the same copy; the one-file shim in
# custom_nodes is how ComfyUI is told to load it, since that is the only directory it
# scans. Nothing in it runs unless H3_GPU_MODE=dual - see the ENV block below.
# The PrivoraVideo product abstraction: modes, roles, the prompt compiler, the canvas
# solver and the workflow builders. This is what keeps ComfyUI node names and H3 prompt
# syntax out of the control plane.
COPY privora /opt/serverless/privora

COPY h3_parallel /opt/serverless/h3_parallel
COPY comfy_custom_nodes/h3_parallel_boot.py ${COMFY_DIR}/custom_nodes/h3_parallel_boot.py

# GPU execution mode.
#
#   single  the known-good path this image has always run: one ComfyUI, one GPU, no
#           patches applied to ComfyUI at all. This is the default, so releasing this
#           image without setting anything behaves exactly like its predecessor.
#   dual    one ComfyUI per GPU, one generation split across both by Ulysses sequence
#           parallelism. Set H3_GPU_MODE=dual on the RunPod endpoint AND give the
#           endpoint 2 GPUs per worker; the worker refuses to start if it is asked for
#           dual and cannot deliver it.
#
# NCCL_DEBUG stays at WARN on purpose: INFO prints a screenful per collective, which
# would bury the benchmark output the image exists to produce. Raise it on the endpoint
# if a rollout needs diagnosing.
ENV H3_GPU_MODE=single \
    H3_SP_MASTER_PORT=29513 \
    H3_SP_VAE=1 \
    H3_SP_SELFTEST=1 \
    H3_BUILD_SOURCE_COMMIT=${H3_BUILD_SOURCE_COMMIT} \
    H3_BUILD_IMAGE_TAG=${H3_BUILD_IMAGE_TAG} \
    H3_BUILD_ID=${H3_BUILD_ID} \
    COMFYUI_H3_COMMIT=dec5d945 \
    NCCL_DEBUG=WARN

WORKDIR /opt/serverless

# Override the base image's Pod ENTRYPOINT. This is the whole point of the image: the
# handler is PID 1, it starts one private ComfyUI on 127.0.0.1, and no model download,
# SSH, Jupyter or FileBrowser ever runs.
ENTRYPOINT ["python3", "/opt/serverless/handler.py"]
CMD []
