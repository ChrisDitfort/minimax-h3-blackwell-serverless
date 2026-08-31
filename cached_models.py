"""Resolve the immutable RunPod Hugging Face cache into ComfyUI model paths.

This module deliberately has no Hugging Face client and no networking code. RunPod must
have materialized the exact repository revision before the worker starts. Missing cache
state is a startup error, never a reason to download weights from inside a job worker.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping


DEFAULT_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
EXPECTED_REPOSITORY = "CDitfort/privora-minimax-h3-models"
EXPECTED_REVISION = "ecb69a4211d74b5798398021003bccde02d63757"
EXPECTED_MANIFEST_VERSION = "multimodal-4-hf-cache-v1"
EXPECTED_TOTAL_BYTES = 69_309_544_079
EXPECTED_SOURCE_REPOSITORY = "Comfy-Org/MiniMax-H3"
BASE_SOURCE_REVISION = "eb8a16107c595128b3a578f82d2ce2f75920c355"
TURBO_SOURCE_REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"
EXPECTED_ASSET_RECORDS = (
    ("fl2va", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
     20_970_379_616, "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
     "diffusion_model", "fl2va", BASE_SOURCE_REVISION),
    ("ref2va", "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
     20_970_379_616, "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
     "diffusion_model", "ref2va", BASE_SOURCE_REVISION),
    ("qwen3vl", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     15_687_142_551, "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
     "text_encoder", "shared", BASE_SOURCE_REVISION),
    ("video_vae", "vae/minimax_h3_video_vae_fp16.safetensors",
     5_207_808_496, "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
     "video_vae", "shared", BASE_SOURCE_REVISION),
    ("audio_vae", "vae/minimax_h3_audio_vae_fp32.safetensors",
     605_254_808, "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
     "audio_vae", "shared", BASE_SOURCE_REVISION),
    ("turbo_fl2v_4step", "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
     1_956_192_992, "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",
     "turbo_lora", "fl2va", TURBO_SOURCE_REVISION),
    ("turbo_fl2v_8step", "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
     1_956_193_000, "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
     "turbo_lora", "fl2va", TURBO_SOURCE_REVISION),
    ("turbo_ref2v_4step", "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
     1_956_193_000, "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
     "turbo_lora", "ref2va", TURBO_SOURCE_REVISION),
)
EXPECTED_ASSET_PATHS = tuple(record[1] for record in EXPECTED_ASSET_RECORDS)

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CachedModelError(RuntimeError):
    """A fail-closed cached-model configuration or filesystem error."""


@dataclass(frozen=True)
class ModelAsset:
    asset_id: str
    source_repository: str
    source_revision: str
    path: str
    expected_bytes: int
    expected_sha256: str
    role: str
    family: str


@dataclass(frozen=True)
class ModelManifest:
    manifest_version: str
    repository: str
    revision: str
    total_bytes: int
    assets: tuple[ModelAsset, ...]


@dataclass(frozen=True)
class ActivationResult:
    repository: str
    revision: str
    manifest_version: str
    asset_count: int
    total_bytes: int
    created_links: tuple[str, ...]
    existing_links: tuple[str, ...]


def _require_string(record: dict, field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise CachedModelError(f"{context}.{field} must be a non-empty string")
    return value


def _safe_relative_path(value: str, context: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise CachedModelError(f"{context} is not a safe repository-relative path")
    normalized = path.as_posix()
    if normalized != value or "\\" in value:
        raise CachedModelError(f"{context} must use normalized POSIX separators")
    return normalized


def load_manifest(path: str | os.PathLike[str]) -> ModelManifest:
    """Load and strictly validate the embedded, trusted model manifest."""

    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CachedModelError(f"Could not load the cached-model manifest: {error}") from error

    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise CachedModelError("Unsupported cached-model manifest schema")

    manifest_version = _require_string(raw, "manifestVersion", "manifest")
    repository = _require_string(raw, "repository", "manifest")
    revision = _require_string(raw, "revision", "manifest")
    total_bytes = raw.get("totalBytes")
    raw_assets = raw.get("assets")

    if manifest_version != EXPECTED_MANIFEST_VERSION:
        raise CachedModelError("Cached-model manifest identity does not match this worker release")
    if repository != EXPECTED_REPOSITORY:
        raise CachedModelError("Cached-model repository does not match this worker release")
    if revision != EXPECTED_REVISION or not _REVISION_RE.fullmatch(revision):
        raise CachedModelError("Cached-model revision does not match this worker release")
    if total_bytes != EXPECTED_TOTAL_BYTES:
        raise CachedModelError("Cached-model manifest total byte count is incorrect")
    if not isinstance(raw_assets, list) or len(raw_assets) != len(EXPECTED_ASSET_RECORDS):
        raise CachedModelError("Cached-model manifest must contain exactly eight assets")

    assets: list[ModelAsset] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_asset in enumerate(raw_assets):
        context = f"assets[{index}]"
        if not isinstance(raw_asset, dict):
            raise CachedModelError(f"{context} must be an object")
        asset_id = _require_string(raw_asset, "id", context)
        source_repository = _require_string(raw_asset, "sourceRepository", context)
        source_revision = _require_string(raw_asset, "sourceRevision", context)
        asset_path = _safe_relative_path(_require_string(raw_asset, "path", context), f"{context}.path")
        expected_bytes = raw_asset.get("expectedBytes")
        expected_sha256 = _require_string(raw_asset, "expectedSha256", context)
        role = _require_string(raw_asset, "role", context)
        family = _require_string(raw_asset, "family", context)
        if raw_asset.get("required") is not True:
            raise CachedModelError(f"{context} must be required")
        if asset_id in seen_ids or asset_path in seen_paths:
            raise CachedModelError("Cached-model manifest contains a duplicate id or path")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise CachedModelError(f"{context}.expectedBytes must be a positive integer")
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise CachedModelError(f"{context}.expectedSha256 is not a SHA-256 digest")
        if source_repository != EXPECTED_SOURCE_REPOSITORY or not _REVISION_RE.fullmatch(source_revision):
            raise CachedModelError(f"{context} has an unapproved source repository or revision")
        if role not in {"diffusion_model", "text_encoder", "video_vae", "audio_vae", "turbo_lora"}:
            raise CachedModelError(f"{context}.role is not approved")
        if family not in {"fl2va", "ref2va", "shared"}:
            raise CachedModelError(f"{context}.family is not approved")
        seen_ids.add(asset_id)
        seen_paths.add(asset_path)
        assets.append(ModelAsset(
            asset_id, source_repository, source_revision, asset_path,
            expected_bytes, expected_sha256, role, family,
        ))

    actual_records = tuple(
        (asset.asset_id, asset.path, asset.expected_bytes, asset.expected_sha256,
         asset.role, asset.family, asset.source_revision)
        for asset in assets
    )
    if actual_records != EXPECTED_ASSET_RECORDS:
        raise CachedModelError("Cached-model manifest asset metadata or ordering is incorrect")
    if sum(asset.expected_bytes for asset in assets) != total_bytes:
        raise CachedModelError("Cached-model asset sizes do not add up to the declared total")
    return ModelManifest(manifest_version, repository, revision, total_bytes, tuple(assets))


def _repo_cache_name(repository: str) -> str:
    owner, separator, name = repository.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise CachedModelError("H3_MODEL_REPO must be an owner/name repository id")
    return f"models--{owner}--{name}"


def _configured_value(environment: Mapping[str, str], name: str, expected: str) -> str:
    value = environment.get(name, expected)
    if value != expected:
        raise CachedModelError(f"{name} does not match the pinned cached-model release")
    return value


def _link_state(destination: Path, expected_target: Path) -> str:
    if not os.path.lexists(destination):
        return "missing"
    if not destination.is_symlink():
        raise CachedModelError(f"Refusing to replace existing model path: {destination}")
    actual = os.readlink(destination)
    if actual != str(expected_target):
        raise CachedModelError(f"Refusing unexpected model symlink target: {destination}")
    return "existing"


def activate_cached_models(
    manifest_path: str | os.PathLike[str],
    *,
    environment: Mapping[str, str] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    comfy_models_root: str | os.PathLike[str] | None = None,
    size_reader: Callable[[Path], int] | None = None,
) -> ActivationResult:
    """Validate the exact snapshot and expose its eight assets through ComfyUI symlinks.

    All sources and destinations are derived from the embedded manifest. Existing regular
    files and links to any other target are rejected rather than overwritten.
    """

    environment = os.environ if environment is None else environment
    manifest = load_manifest(manifest_path)
    repository = _configured_value(environment, "H3_MODEL_REPO", manifest.repository)
    revision = _configured_value(environment, "H3_MODEL_REVISION", manifest.revision)
    manifest_version = _configured_value(
        environment, "H3_MODEL_MANIFEST_VERSION", manifest.manifest_version
    )

    root = Path(cache_root or environment.get("H3_MODEL_CACHE_ROOT", str(DEFAULT_CACHE_ROOT))).absolute()
    if not root.is_dir():
        raise CachedModelError("RunPod Hugging Face cache root is absent")
    repository_root = root / _repo_cache_name(repository)
    if not repository_root.is_dir():
        raise CachedModelError("Pinned Hugging Face model repository is absent from the RunPod cache")
    snapshot = repository_root / "snapshots" / revision
    if not snapshot.is_dir():
        raise CachedModelError("Pinned Hugging Face model revision is absent from the RunPod cache")

    if comfy_models_root is None:
        comfy_root = Path(environment.get("COMFY_DIR", "/opt/comfyui-baked")) / "models"
    else:
        comfy_root = Path(comfy_models_root)
    comfy_root = comfy_root.absolute()
    size_reader = (lambda path: path.stat().st_size) if size_reader is None else size_reader

    planned: list[tuple[Path, Path, str]] = []
    existing: list[str] = []
    for asset in manifest.assets:
        source = snapshot.joinpath(*PurePosixPath(asset.path).parts)
        if not source.is_file():
            raise CachedModelError(f"Required cached model asset is absent: {asset.path}")
        actual_size = size_reader(source)
        if actual_size != asset.expected_bytes:
            raise CachedModelError(
                f"Cached model size mismatch for {asset.path}: expected {asset.expected_bytes}, "
                f"found {actual_size}"
            )
        destination = comfy_root.joinpath(*PurePosixPath(asset.path).parts)
        state = _link_state(destination, source)
        if state == "existing":
            existing.append(asset.path)
        else:
            planned.append((destination, source, asset.path))

    # The Pixaroma example workflows use the historical h3/ alias for FL2VA. Keep that
    # alias without duplicating bytes or changing the bare filename used by Privora.
    fl2va_relative = EXPECTED_ASSET_PATHS[0]
    fl2va_destination = comfy_root.joinpath(*PurePosixPath(fl2va_relative).parts)
    compatibility = comfy_root / "diffusion_models" / "h3" / Path(fl2va_relative).name
    compatibility_state = _link_state(compatibility, fl2va_destination)
    if compatibility_state == "missing":
        planned.append((compatibility, fl2va_destination, "diffusion_models/h3 compatibility alias"))

    created_paths: list[Path] = []
    created_labels: list[str] = []
    try:
        for destination, source, label in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(source), destination)
            created_paths.append(destination)
            created_labels.append(label)
    except OSError as error:
        for created in reversed(created_paths):
            try:
                if created.is_symlink():
                    created.unlink()
            except OSError:
                pass
        raise CachedModelError(f"Could not create cached-model symlinks: {error}") from error

    return ActivationResult(
        repository=repository,
        revision=revision,
        manifest_version=manifest_version,
        asset_count=len(manifest.assets),
        total_bytes=manifest.total_bytes,
        created_links=tuple(created_labels),
        existing_links=tuple(existing),
    )
