from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bootstrap
from scripts import (
    guard_no_model_weights,
    report_image_manifest,
    stage_slim_image,
    write_slim_release,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "h3-cache-manifest.json"
IMAGE_REPOSITORY = "ghcr.io/chrisditfort/privora-h3-runpod-worker"


class SlimDockerfileTests(unittest.TestCase):
    def test_slim_image_uses_cache_bootstrap_and_never_adds_weight_layers(self):
        dockerfile = (ROOT / "Dockerfile.slim").read_text(encoding="utf-8")
        self.assertIn(
            "FROM ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-blackwell@"
            "sha256:4fdcd50e8e5f54f8329933c66e2eac17680cbac82d43c1a74d00465e9413a3e1",
            dockerfile,
        )
        self.assertIn('ENTRYPOINT ["python3", "/opt/serverless/bootstrap.py"]', dockerfile)
        self.assertIn("H3_MODEL_REVISION=ecb69a4211d74b5798398021003bccde02d63757", dockerfile)
        self.assertIn("H3_MODEL_MANIFEST_VERSION=multimodal-4-hf-cache-v1", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)
        self.assertIn("guard_no_model_weights.py", dockerfile)
        self.assertIn("command -v ffmpeg", dockerfile)
        self.assertIn("import torch; assert torch.version.cuda", dockerfile)
        self.assertIn("slim-build-identity.json", dockerfile)
        self.assertNotIn("build_model_layer", dockerfile)
        self.assertNotIn("models.tsv", dockerfile)
        self.assertNotIn("resolve/eb8a", dockerfile)
        self.assertNotIn("ARG H3_", dockerfile)
        for asset in json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]:
            self.assertNotIn(f"COPY {asset['path']}", dockerfile)

    def test_hugging_face_space_publisher_has_been_removed(self):
        self.assertFalse((ROOT / ".github/workflows/publish-hf-space.yml").exists())
        self.assertFalse((ROOT / "scripts/publish_hf_space.py").exists())
        self.assertFalse((ROOT / "scripts/inspect_hf_space.py").exists())
        self.assertFalse((ROOT / "Dockerfile.hf-space").exists())

    def test_remote_test_requirements_include_the_image_fixture_dependency(self):
        requirements = (ROOT / "scripts/requirements-slim-ci.txt").read_text(encoding="utf-8")
        self.assertEqual(requirements.strip(), "Pillow==12.3.0")


class BuildIdentityTests(unittest.TestCase):
    def test_embedded_identity_overrides_runtime_attempts_to_relabel_the_image(self):
        identity = {
            "schemaVersion": 1,
            "sourceCommit": "1" * 40,
            "buildId": "12345-1",
            "imageRepository": IMAGE_REPOSITORY,
            "imageTag": "cached-models-1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(json.dumps(identity), encoding="utf-8")
            hostile = {
                "H3_BUILD_SOURCE_COMMIT": "wrong",
                "H3_BUILD_ID": "wrong",
                "H3_IMAGE_REPOSITORY": "wrong",
                "H3_BUILD_IMAGE_TAG": "wrong",
            }
            with mock.patch.object(bootstrap, "BUILD_IDENTITY_PATH", str(path)), mock.patch.dict(
                os.environ, hostile, clear=True
            ):
                bootstrap._load_build_identity()
                self.assertEqual(os.environ["H3_BUILD_SOURCE_COMMIT"], identity["sourceCommit"])
                self.assertEqual(os.environ["H3_BUILD_ID"], identity["buildId"])
                self.assertEqual(os.environ["H3_IMAGE_REPOSITORY"], identity["imageRepository"])
                self.assertEqual(os.environ["H3_BUILD_IMAGE_TAG"], identity["imageTag"])

    def test_reserved_rollback_tag_is_rejected(self):
        identity = {
            "schemaVersion": 1,
            "sourceCommit": "1" * 40,
            "buildId": "12345-1",
            "imageRepository": IMAGE_REPOSITORY,
            "imageTag": "multimodal-4",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(json.dumps(identity), encoding="utf-8")
            with mock.patch.object(bootstrap, "BUILD_IDENTITY_PATH", str(path)):
                with self.assertRaisesRegex(bootstrap.CachedModelError, "Reserved"):
                    bootstrap._load_build_identity()


class StagingTests(unittest.TestCase):
    def test_staging_contains_only_the_allowlist_and_zero_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "context"
            publication = stage_slim_image.stage(
                ROOT,
                output,
                source_commit="1" * 40,
                build_id="local-test-1",
                image_repository=IMAGE_REPOSITORY,
                image_tag="cached-models-1",
            )
            self.assertEqual(publication["imageRepository"], IMAGE_REPOSITORY)
            self.assertEqual(publication["imageTag"], "cached-models-1")
            self.assertTrue((output / "Dockerfile").is_file())
            self.assertTrue((output / "slim-build-identity.json").is_file())
            self.assertTrue((output / "slim-context-manifest.json").is_file())
            self.assertFalse((output / "models.tsv").exists())
            self.assertFalse((output / ".github").exists())
            self.assertFalse((output / "tests").exists())
            self.assertFalse((output / "worker.js").exists())
            weights = [
                path for path in output.rglob("*")
                if path.is_file() and path.suffix.lower() in guard_no_model_weights.WEIGHT_SUFFIXES
            ]
            self.assertEqual(weights, [])
            self.assertEqual(
                guard_no_model_weights.scan(
                    [output], output / "models/h3-cache-manifest.json", max_bytes=0
                ),
                [],
            )

    def test_staging_refuses_existing_output_wrong_registry_and_reserved_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            cases = (
                ({"image_repository": IMAGE_REPOSITORY, "image_tag": "cached-models-1"}, "already exists"),
                ({"image_repository": "ghcr.io/cditfort/wrong", "image_tag": "cached-models-1"}, "approved GHCR"),
                ({"image_repository": IMAGE_REPOSITORY, "image_tag": "multimodal-4"}, "reserved"),
            )
            for values, error in cases:
                with self.subTest(values=values):
                    target = output if "already" in error else Path(directory) / ("new-" + error.replace(" ", "-"))
                    with self.assertRaisesRegex(stage_slim_image.StagingError, error):
                        stage_slim_image.stage(
                            ROOT,
                            target,
                            source_commit="1" * 40,
                            build_id="local-test-1",
                            **values,
                        )

    def test_weight_guard_detects_exact_approved_name_and_large_alternate_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "minimax_h3_video_vae_fp16.safetensors"
            exact.write_bytes(b"x")
            alternate = root / "unapproved.ckpt"
            alternate.write_bytes(b"xx")
            violations = guard_no_model_weights.scan([root], MANIFEST, max_bytes=1)
            self.assertEqual(len(violations), 2)
            self.assertIn("approved model filename", violations[0] + violations[1])
            self.assertIn("large .ckpt", violations[0] + violations[1])


class ImageMetadataTests(unittest.TestCase):
    def test_registry_metadata_requires_linux_amd64_and_cache_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            config = root / "config.json"
            manifest.write_text(json.dumps({"layers": [{"size": 11}, {"size": 22}]}), encoding="utf-8")
            config.write_text(json.dumps({
                "os": "linux",
                "architecture": "amd64",
                "config": {
                    "Entrypoint": ["python3", "/opt/serverless/bootstrap.py"],
                    "Cmd": [],
                    "Env": [f"{key}={value}" for key, value in
                            report_image_manifest.EXPECTED_ENVIRONMENT.items()],
                    "Labels": {
                        "org.opencontainers.image.source":
                            "https://github.com/ChrisDitfort/minimax-h3-blackwell-serverless",
                        "org.opencontainers.image.revision": "1" * 40,
                        "org.opencontainers.image.version": "cached-models-1",
                        "ai.privora.build.id": "12345-1",
                        "ai.privora.model.repository":
                            "CDitfort/privora-minimax-h3-models",
                        "ai.privora.model.revision":
                            "ecb69a4211d74b5798398021003bccde02d63757",
                        "ai.privora.model.manifest": "multimodal-4-hf-cache-v1",
                    },
                },
            }), encoding="utf-8")
            result = report_image_manifest.main([
                "--manifest", str(manifest),
                "--config", str(config),
                "--digest", "sha256:" + "a" * 64,
                "--platform-digest", "sha256:" + "b" * 64,
                "--registry-ref", IMAGE_REPOSITORY + "@sha256:" + "a" * 64,
                "--source-commit", "1" * 40,
                "--image-tag", "cached-models-1",
                "--build-id", "12345-1",
            ])
            self.assertEqual(result, 0)

    def test_release_record_contains_separate_application_and_model_identity(self):
        document = write_slim_release.release_document(
            source_commit="1" * 40,
            image_tag="cached-models-1",
            image_digest="sha256:" + "a" * 64,
            platform_digest="sha256:" + "b" * 64,
            build_id="12345-1",
            compressed_bytes=123456,
        )
        self.assertEqual(document["sourceBranch"], "hf-cached-models-slim")
        self.assertEqual(document["image"], IMAGE_REPOSITORY + ":cached-models-1")
        self.assertEqual(document["platformImageDigest"], "sha256:" + "b" * 64)
        self.assertEqual(
            document["baseImageDigest"],
            "sha256:4fdcd50e8e5f54f8329933c66e2eac17680cbac82d43c1a74d00465e9413a3e1",
        )
        self.assertEqual(document["modelRelease"], {
            "repository": "CDitfort/privora-minimax-h3-models",
            "revision": "ecb69a4211d74b5798398021003bccde02d63757",
            "manifest": "multimodal-4-hf-cache-v1",
        })
        self.assertEqual(document["modelWeightsInImage"], 0)


class PreservedContractTests(unittest.TestCase):
    def test_ref2va_socket_prompt_ordinals_and_turbo_names_remain_exact(self):
        workflows = (ROOT / "privora/workflows.py").read_text(encoding="utf-8")
        models = (ROOT / "privora/models.py").read_text(encoding="utf-8")
        verifier = (ROOT / "scripts/verify_ref2va_contract.py").read_text(encoding="utf-8")
        self.assertIn("ref_images.ref_image_0", workflows)
        self.assertIn("ref_images.ref_image_0", verifier)
        self.assertIn("<Picture 1>", verifier)
        for name in (
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        ):
            self.assertIn(name, models)


if __name__ == "__main__":
    unittest.main()
