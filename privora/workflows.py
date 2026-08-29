"""Build the ComfyUI graph for a validated request.

This is the last place ComfyUI exists as a concept. Above it everything speaks
PrivoraVideo; below it the handler simply POSTs a dict. Keeping the boundary here is what
makes the eventual "upgrade ComfyUI" or "replace H3" possible without touching the
control plane.

Graph shape, both families:

    UNETLoader ──(+ LoraLoaderModelOnly when accelerated)──> BasicGuider ─┐
    CLIPLoader ──┐                                                        ├─> SamplerCustomAdvanced
    VAELoader  ──┼─> MiniMaxH3ImageToVideo | MiniMaxH3ReferenceToVideo ───┘        │
    VAELoader  ──┘   (conditioning + empty AV latent)                              │
                                                                    ┌──────────────┴───────┐
                                                              VAEDecode           VAEDecodeAudio
                                                                    └───> CreateVideo ─> SaveVideo

Node ids are stable strings rather than numbers. The perf instrumentation labels phases by
node, and a graph whose ids shift between modes would make two jobs incomparable.
"""

from __future__ import annotations

from . import models, references as references_module
from .request import ANIMATE, CREATE, GenerationRequest

#: Model filenames as the loaders see them. The FL2VA symlink the Dockerfile creates means
#: the bare name resolves; Ref2VA is added to the same directory.
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

SAMPLER = "res_multistep"
SCHEDULER = "simple"

#: Stable node ids, shared by both families wherever the role is the same.
UNET = "unet"
LORA = "lora"
CLIP = "clip"
VIDEO_VAE_NODE = "vae_video"
AUDIO_VAE_NODE = "vae_audio"
CONDITIONING = "conditioning"
NOISE = "noise"
GUIDER = "guider"
SAMPLER_SELECT = "sampler"
SIGMAS = "sigmas"
SAMPLE = "sample"
DECODE_VIDEO = "decode_video"
DECODE_AUDIO = "decode_audio"
CREATE_VIDEO = "create_video"
SAVE = "save"

#: Where a staged reference file is wired in. The handler writes the file into the job's
#: isolated input directory and puts its generated name here; nothing user-supplied ever
#: reaches a filename.
IMAGE_LOADER = "PixaromaLoadImage"


class WorkflowPlan:
    """A built graph plus the staging the handler has to do before submitting it."""

    def __init__(self, graph: dict, staging: list[dict], acceleration: models.Acceleration):
        self.graph = graph
        #: [{"node": <id>, "field": "image", "reference": Reference}] - what to write where.
        self.staging = staging
        self.acceleration = acceleration

    def as_metadata(self) -> dict:
        data = {
            "generationMode": models._wire(self.acceleration.mode),
            "steps": self.acceleration.steps,
            "acceleration": self.acceleration.kind,
        }
        if self.acceleration.note:
            data["accelerationNote"] = self.acceleration.note
        return data


def _model_source(graph: dict, acceleration: models.Acceleration) -> list:
    """The MODEL link, with the Turbo LoRA spliced in when one is selected.

    LoraLoaderModelOnly rather than LoraLoader: these are distilled *model* LoRAs and have
    no text-encoder half, so routing CLIP through a loader that expects one would be wrong.
    """
    graph[UNET] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": models.CHECKPOINTS[acceleration.family],
            "weight_dtype": "default",
        },
    }
    if acceleration.lora is None:
        return [UNET, 0]

    graph[LORA] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": [UNET, 0], "lora_name": acceleration.lora, "strength_model": 1.0},
    }
    return [LORA, 0]


def _tail(graph: dict, model_link: list, request: GenerationRequest,
          acceleration: models.Acceleration) -> None:
    """Sampler through save. Identical for both families, which is the point."""
    graph[NOISE] = {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed}}
    graph[GUIDER] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_link, "conditioning": [CONDITIONING, 0]},
    }
    graph[SAMPLER_SELECT] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}}
    graph[SIGMAS] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_link,
            "scheduler": SCHEDULER,
            "steps": acceleration.steps,
            "denoise": 1.0,
        },
    }
    graph[SAMPLE] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": [NOISE, 0],
            "guider": [GUIDER, 0],
            "sampler": [SAMPLER_SELECT, 0],
            "sigmas": [SIGMAS, 0],
            "latent_image": [CONDITIONING, 1],
        },
    }
    graph[DECODE_VIDEO] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [SAMPLE, 0], "vae": [VIDEO_VAE_NODE, 0]},
    }
    graph[DECODE_AUDIO] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": [SAMPLE, 0], "vae": [AUDIO_VAE_NODE, 0]},
    }
    graph[CREATE_VIDEO] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": [DECODE_VIDEO, 0],
            "audio": [DECODE_AUDIO, 0],
            "fps": float(request.canvas.fps),
        },
    }
    graph[SAVE] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [CREATE_VIDEO, 0],
            # Opaque and fixed. Never derived from the prompt or a reference filename.
            "filename_prefix": "video/H3",
            "format": "mp4",
            "codec": "h264",
        },
    }


def _loaders(graph: dict, *, audio_vae: bool) -> None:
    graph[CLIP] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax"},
    }
    graph[VIDEO_VAE_NODE] = {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}}
    graph[AUDIO_VAE_NODE] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}
    if not audio_vae:  # the audio VAE is always needed for decode; kept for symmetry
        pass


def _stage_image(graph: dict, staging: list, node_id: str, reference) -> list:
    """Add a loader for one staged image and return its link."""
    graph[node_id] = {"class_type": IMAGE_LOADER, "inputs": {"image": ""}}
    staging.append({"node": node_id, "field": "image", "reference": reference})
    return [node_id, 0]


def build_fl2va(request: GenerationRequest, acceleration: models.Acceleration) -> WorkflowPlan:
    """create and animate. One node covers all four keyframe combinations."""
    graph: dict = {}
    staging: list[dict] = []
    _loaders(graph, audio_vae=True)

    inputs = {
        "clip": [CLIP, 0],
        "vae": [VIDEO_VAE_NODE, 0],
        "prompt": request.prompt.text,
        "width": request.canvas.width,
        "height": request.canvas.height,
        "length": request.canvas.frames,
    }
    if request.first_frame is not None:
        inputs["first_frame"] = _stage_image(graph, staging, "first_frame", request.first_frame)
    if request.last_frame is not None:
        inputs["last_frame"] = _stage_image(graph, staging, "last_frame", request.last_frame)

    graph[CONDITIONING] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}
    _tail(graph, _model_source(graph, acceleration), request, acceleration)
    return WorkflowPlan(graph, staging, acceleration)


def build_ref2va(request: GenerationRequest, acceleration: models.Acceleration) -> WorkflowPlan:
    """references and remix.

    Reference inputs use the node's autogrow naming - ref_image_1..9, ref_video_1..3,
    ref_video_audio_N paired by index to ref_video_N, ref_audio_1..3. Ordering here must
    match privora.references.assign_ordinals(), because that is what the compiled prompt's
    <Picture n> / <Video n> / <Audio n> tags were numbered against.
    """
    graph: dict = {}
    staging: list[dict] = []
    _loaders(graph, audio_vae=True)

    inputs = {
        "clip": [CLIP, 0],
        "vae": [VIDEO_VAE_NODE, 0],
        "audio_vae": [AUDIO_VAE_NODE, 0],
        "prompt": request.prompt.text,
        "width": request.canvas.width,
        "height": request.canvas.height,
        "length": request.canvas.frames,
        "ref_image_size": request.ref_image_size,
    }

    for index, image in enumerate(request.references.images, start=1):
        node_id = f"ref_image_{index}"
        inputs[node_id] = _stage_image(graph, staging, node_id, image)

    for index, video in enumerate(request.references.videos, start=1):
        node_id = f"ref_video_{index}"
        graph[node_id] = {"class_type": "LoadVideo", "inputs": {"file": ""}}
        staging.append({"node": node_id, "field": "file", "reference": video})
        inputs[node_id] = [node_id, 0]

        if video.soundtrack is not None:
            audio_node = f"ref_video_audio_{index}"
            graph[audio_node] = {"class_type": "LoadAudio", "inputs": {"audio": ""}}
            staging.append({"node": audio_node, "field": "audio", "reference": video.soundtrack})
            inputs[audio_node] = [audio_node, 0]

    for index, audio in enumerate(request.references.audios, start=1):
        node_id = f"ref_audio_{index}"
        graph[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": ""}}
        staging.append({"node": node_id, "field": "audio", "reference": audio})
        inputs[node_id] = [node_id, 0]

    graph[CONDITIONING] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
    _tail(graph, _model_source(graph, acceleration), request, acceleration)
    return WorkflowPlan(graph, staging, acceleration)


BUILDERS = {CREATE: build_fl2va, ANIMATE: build_fl2va}


def build(request: GenerationRequest, inventory: models.ModelInventory,
          generation_mode: str = models.DEFAULT_GENERATION_MODE) -> WorkflowPlan:
    """Route a validated request to its family's builder, with the mode resolved.

    `inventory.resolve` raises before anything is built when the requested combination is
    not installed - so an image without the Ref2VA LoRA answers "turboFast is unavailable
    for references" rather than loading 21 GB and then failing.
    """
    acceleration = inventory.resolve(request.family, generation_mode)
    builder = BUILDERS.get(request.mode, build_ref2va)
    return builder(request, acceleration)
