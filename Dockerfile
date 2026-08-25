FROM ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-blackwell

USER root

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir \
    runpod \
    requests \
    websocket-client

COPY handler.py /workspace/handler.py

CMD ["python3", "/workspace/handler.py"]