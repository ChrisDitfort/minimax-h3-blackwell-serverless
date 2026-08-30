#!/usr/bin/env python3
"""Verify Privora's Ref2VA graph against the ComfyUI source baked into the image.

This is intentionally a source-level image build check, not another permissive ComfyUI
mock.  MiniMaxH3ReferenceToVideo uses ComfyUI V3 Autogrow inputs, whose API-format names
are dotted and zero-based.  A graph can pass ``POST /prompt`` validation with a bare
Autogrow key and then fail only when the node's ``execute()`` method is called.  That is
the exact boundary this check protects.

The Dockerfile runs this script against /opt/comfyui-baked.  For an operator-side check of
the pinned source without an image filesystem, pass the two official raw-source URLs
instead.  No model is loaded and no sampling occurs.
"""

from __future__ import annotations

import argparse
import ast
import sys
import urllib.request
from pathlib import Path


EXPECTED_AUTOGROW = {
    "ref_images": ("ref_image_", 0, 9),
    "ref_videos": ("ref_video_", 0, 3),
    "ref_video_audios": ("ref_video_audio_", 0, 3),
    "ref_audios": ("ref_audio_", 0, 3),
}

REQUIRED_EXECUTE_ARGS = {
    "clip", "vae", "audio_vae", "prompt", "width", "height", "length"
}


class ContractError(RuntimeError):
    """The built graph and the installed ComfyUI node no longer agree."""


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _literal_keyword(call: ast.Call, name: str, default=None):
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.literal_eval(keyword.value)
    return default


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ContractError(f"Installed ComfyUI source has no {name} class")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ContractError(f"{class_node.name} has no {name} method")


def _execute_args(class_node: ast.ClassDef) -> set[str]:
    execute = _method(class_node, "execute")
    return {argument.arg for argument in execute.args.args if argument.arg != "cls"}


def _autogrow_specs(class_node: ast.ClassDef) -> dict[str, tuple[str, int, int]]:
    specs: dict[str, tuple[str, int, int]] = {}
    for call in (node for node in ast.walk(class_node) if isinstance(node, ast.Call)):
        if _dotted_name(call.func) != "io.Autogrow.Input" or not call.args:
            continue
        group = ast.literal_eval(call.args[0])
        template = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "template"), None
        )
        if not isinstance(template, ast.Call) or (
            _dotted_name(template.func) != "io.Autogrow.TemplatePrefix"
        ):
            raise ContractError(f"{group} no longer uses Autogrow.TemplatePrefix")
        prefix = _literal_keyword(template, "prefix")
        minimum = _literal_keyword(template, "min")
        maximum = _literal_keyword(template, "max")
        specs[str(group)] = (str(prefix), int(minimum), int(maximum))
    return specs


def _schema_output_calls(class_node: ast.ClassDef) -> list[str]:
    define_schema = _method(class_node, "define_schema")
    for node in ast.walk(define_schema):
        if not isinstance(node, ast.Call) or not _dotted_name(node.func).endswith(".Schema"):
            continue
        outputs = next((keyword.value for keyword in node.keywords if keyword.arg == "outputs"), None)
        if isinstance(outputs, (ast.List, ast.Tuple)):
            return [
                _dotted_name(element.func)
                for element in outputs.elts
                if isinstance(element, ast.Call)
            ]
    raise ContractError(f"Could not read {class_node.name}.define_schema() outputs")


def inspect_installed_contract(h3_source: str, video_source: str):
    h3_tree = ast.parse(h3_source)
    ref_node = _class(h3_tree, "MiniMaxH3ReferenceToVideo")
    specs = _autogrow_specs(ref_node)
    if specs != EXPECTED_AUTOGROW:
        raise ContractError(
            f"MiniMaxH3ReferenceToVideo Autogrow schema changed: {specs!r}"
        )

    execute_args = _execute_args(ref_node)
    missing_groups = set(EXPECTED_AUTOGROW) - execute_args
    if missing_groups:
        raise ContractError(
            "MiniMaxH3ReferenceToVideo.execute() no longer accepts "
            + ", ".join(sorted(missing_groups))
        )

    video_tree = ast.parse(video_source)
    load_outputs = _schema_output_calls(_class(video_tree, "LoadVideo"))
    component_outputs = _schema_output_calls(_class(video_tree, "GetVideoComponents"))
    if not load_outputs or load_outputs[0] != "io.Video.Output":
        raise ContractError(f"LoadVideo output changed: {load_outputs!r}")
    if not component_outputs or component_outputs[0] != "io.Image.Output":
        raise ContractError(f"GetVideoComponents output changed: {component_outputs!r}")

    return specs, execute_args


def _normalise_autogrow(inputs: dict, specs: dict) -> dict:
    """Mirror the relevant effect of ComfyUI build_nested_inputs()."""
    normalised = {}
    for key, value in inputs.items():
        if "." not in key:
            normalised[key] = value
            continue
        group, inner = key.split(".", 1)
        if group not in specs:
            normalised[key] = value
            continue
        normalised.setdefault(group, {})[inner] = value
    return normalised


def verify_generated_graph(specs: dict, execute_args: set[str]) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

    from privora import models, request, workflows

    inventory = models.ModelInventory.from_names([
        *models.CHECKPOINTS.values(),
        *(acceleration.lora for acceleration in models.ACCELERATIONS.values()
          if acceleration.lora is not None),
    ])

    one_image = request.parse({
        "mode": "references",
        "prompt": "image-level contract probe",
        "seed": 7,
        "generationMode": "quality",
        "referenceFidelity": "standard",
        "references": [{"type": "image", "role": "character"}],
    })
    one_plan = workflows.build(one_image, inventory, one_image.generation_mode)
    one_inputs = one_plan.graph[workflows.CONDITIONING]["inputs"]
    if one_inputs.get("ref_images.ref_image_0") != ["ref_image_1", 0]:
        raise ContractError("One-image Ref2VA graph does not use ref_images.ref_image_0")
    if "ref_image_1" in one_inputs:
        raise ContractError("One-image Ref2VA graph still emits the bare ref_image_1 key")
    if one_inputs.get("ref_image_size") != "match":
        raise ContractError("standard reference fidelity no longer maps to match")
    if "<Picture 1>" not in one_inputs["prompt"] or "<Picture 2>" in one_inputs["prompt"]:
        raise ContractError("One-image prompt numbering no longer matches Picture 1")
    if "lora" in one_plan.graph or one_plan.graph[workflows.SIGMAS]["inputs"]["steps"] != 20:
        raise ContractError("Ref2VA quality must remain base-model sampling at 20 steps")
    if one_plan.graph[workflows.UNET]["inputs"]["unet_name"] != models.CHECKPOINTS[models.REF2VA]:
        raise ContractError("Ref2VA quality selected the wrong checkpoint")

    mixed = request.parse({
        "mode": "references",
        "prompt": "mixed-media contract probe",
        "seed": 11,
        "references": [
            {"type": "image", "role": "identity"},
            {
                "type": "video", "role": "motion",
                "soundtrack": {"type": "audio", "role": "ambience"},
            },
            {"type": "audio", "role": "voice"},
        ],
    })
    mixed_plan = workflows.build(mixed, inventory, mixed.generation_mode)
    mixed_inputs = mixed_plan.graph[workflows.CONDITIONING]["inputs"]
    expected_links = {
        "ref_images.ref_image_0": ["ref_image_1", 0],
        "ref_videos.ref_video_0": ["ref_video_frames_1", 0],
        "ref_video_audios.ref_video_audio_0": ["ref_video_audio_1", 0],
        "ref_audios.ref_audio_0": ["ref_audio_1", 0],
    }
    for socket, link in expected_links.items():
        if mixed_inputs.get(socket) != link:
            raise ContractError(f"{socket} has {mixed_inputs.get(socket)!r}, expected {link!r}")

    frames = mixed_plan.graph.get("ref_video_frames_1") or {}
    if frames != {
        "class_type": workflows.VIDEO_COMPONENTS,
        "inputs": {"video": ["ref_video_1", 0]},
    }:
        raise ContractError("Reference video is not converted from VIDEO to IMAGE frames")

    for plan in (one_plan, mixed_plan):
        inputs = plan.graph[workflows.CONDITIONING]["inputs"]
        for key in inputs:
            for group, (prefix, _minimum, maximum) in specs.items():
                bare_suffix = key[len(prefix):] if key.startswith(prefix) else ""
                if bare_suffix.isdigit():
                    raise ContractError(f"Bare Autogrow socket survived: {key}")
                dotted_prefix = f"{group}.{prefix}"
                if key.startswith(dotted_prefix):
                    suffix = key[len(dotted_prefix):]
                    if not suffix.isdigit() or int(suffix) >= maximum:
                        raise ContractError(f"Autogrow socket is outside the schema: {key}")

        kwargs = _normalise_autogrow(inputs, specs)
        unexpected = set(kwargs) - execute_args
        if unexpected:
            raise ContractError(
                "Generated graph would pass unexpected execute() keywords: "
                + ", ".join(sorted(unexpected))
            )
        missing = REQUIRED_EXECUTE_ARGS - set(kwargs)
        if missing:
            raise ContractError(
                "Generated graph omits required execute() arguments: "
                + ", ".join(sorted(missing))
            )


def _read_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _sources(args) -> tuple[str, str]:
    if args.comfy_root:
        root = Path(args.comfy_root)
        return (
            (root / "comfy_extras" / "nodes_minimax_h3.py").read_text(encoding="utf-8"),
            (root / "comfy_extras" / "nodes_video.py").read_text(encoding="utf-8"),
        )
    if args.h3_source_url and args.video_source_url:
        return _read_url(args.h3_source_url), _read_url(args.video_source_url)
    raise ContractError(
        "Provide --comfy-root, or both --h3-source-url and --video-source-url"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root")
    parser.add_argument("--h3-source-url")
    parser.add_argument("--video-source-url")
    args = parser.parse_args()

    try:
        h3_source, video_source = _sources(args)
        specs, execute_args = inspect_installed_contract(h3_source, video_source)
        verify_generated_graph(specs, execute_args)
    except (ContractError, OSError, SyntaxError, ValueError) as error:
        print(f"Ref2VA contract verification FAILED: {error}", file=sys.stderr)
        return 1

    print(
        "Ref2VA contract OK: dotted V3 Autogrow inputs normalise to execute() groups; "
        "video references are IMAGE frames; quality is base Ref2VA at 20 steps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
