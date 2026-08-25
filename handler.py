import json
import subprocess
import time
import uuid

import requests
import runpod
from websocket import create_connection

COMFY_DIR = "/workspace/runpod-slim/ComfyUI"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_WS = "ws://127.0.0.1:8188/ws"

comfy_process = None


def start_comfyui():
    global comfy_process

    if comfy_process is not None and comfy_process.poll() is None:
        return

    comfy_process = subprocess.Popen(
        [
            "python",
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            "--preview-method",
            "none",
        ],
        cwd=COMFY_DIR,
    )

    for _ in range(180):
        try:
            response = requests.get(
                f"{COMFY_URL}/system_stats",
                timeout=2,
            )

            if response.ok:
                print("ComfyUI is ready")
                return
        except Exception:
            pass

        time.sleep(1)

    raise RuntimeError("ComfyUI failed to start")


def run_workflow(workflow):
    client_id = str(uuid.uuid4())

    ws = create_connection(
        f"{COMFY_WS}?clientId={client_id}",
        timeout=30,
    )

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={
            "prompt": workflow,
            "client_id": client_id,
        },
        timeout=30,
    )

    response.raise_for_status()

    prompt_id = response.json()["prompt_id"]

    while True:
        try:
            message = ws.recv()
        except Exception:
            continue

        if not message:
            continue

        data = json.loads(message)

        if data.get("type") == "executing":
            payload = data.get("data", {})

            if (
                payload.get("prompt_id") == prompt_id
                and payload.get("node") is None
            ):
                break

    ws.close()

    history = requests.get(
        f"{COMFY_URL}/history/{prompt_id}",
        timeout=30,
    )

    history.raise_for_status()

    return history.json()


def handler(job):
    try:
        start_comfyui()

        job_input = job.get("input", {})
        workflow = job_input.get("workflow")

        if not workflow:
            return {
                "error": "input.workflow is required"
            }

        result = run_workflow(workflow)

        return {
            "result": result
        }

    except Exception as error:
        return {
            "error": str(error)
        }


runpod.serverless.start({
    "handler": handler
})