from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import guard_no_model_weights, publish_hf_space, stage_hf_space


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "h3-cache-manifest.json"


class SpaceDockerfileTests(unittest.TestCase):
    def test_space_image_uses_cache_bootstrap_and_never_adds_weight_layers(self):
        dockerfile = (ROOT / "Dockerfile.hf-space").read_text(encoding="utf-8")
        self.assertIn('ENTRYPOINT ["python3", "/opt/serverless/bootstrap.py"]', dockerfile)
        self.assertIn("H3_MODEL_REVISION=ecb69a4211d74b5798398021003bccde02d63757", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)
        self.assertIn("guard_no_model_weights.py", dockerfile)
        self.assertNotIn("build_model_layer", dockerfile)
        self.assertNotIn("models.tsv", dockerfile)
        self.assertNotIn("resolve/eb8a", dockerfile)
        for asset in json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]:
            self.assertNotIn(f"COPY {asset['path']}", dockerfile)

    def test_workflow_is_manual_oidc_only_and_has_no_artifact_or_token_fallback(self):
        workflow = (ROOT / ".github/workflows/publish-hf-space.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("id-token: write", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("HF_OIDC_RESOURCE: spaces/CDitfort/privora-h3-runpod-worker", workflow)
        self.assertIn("default: false", workflow)
        self.assertNotIn("secrets.HF_TOKEN", workflow)
        self.assertNotIn("upload-artifact", workflow)
        self.assertNotIn("docker build", workflow)
        self.assertNotIn("docker pull", workflow)


class StagingTests(unittest.TestCase):
    def test_staging_contains_only_the_allowlist_and_zero_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "space"
            publication = stage_hf_space.stage(
                ROOT,
                output,
                source_commit="1" * 40,
                build_id="local-test-1",
                space_repository=stage_hf_space.EXPECTED_SPACE_REPOSITORY,
            )
            self.assertEqual(publication["spaceRepository"], "CDitfort/privora-h3-runpod-worker")
            self.assertTrue((output / "Dockerfile").is_file())
            self.assertTrue((output / "README.md").is_file())
            self.assertFalse((output / "models.tsv").exists())
            self.assertFalse((output / ".github").exists())
            self.assertFalse((output / "tests").exists())
            self.assertFalse((output / "worker.js").exists())
            weights = [
                path for path in output.rglob("*")
                if path.is_file() and path.suffix.lower() in guard_no_model_weights.WEIGHT_SUFFIXES
            ]
            self.assertEqual(weights, [])
            self.assertEqual(guard_no_model_weights.scan([output], output / "models/h3-cache-manifest.json", 0), [])

    def test_staging_refuses_to_merge_into_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(stage_hf_space.StagingError, "already exists"):
                stage_hf_space.stage(
                    ROOT,
                    output,
                    source_commit="1" * 40,
                    build_id="local-test-1",
                    space_repository=stage_hf_space.EXPECTED_SPACE_REPOSITORY,
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

    def test_token_redaction_removes_short_lived_or_long_lived_values(self):
        first = "hf_oidc_ThisMustNeverAppear"
        second = "opaque-token-not-matching-prefix"
        with mock.patch.dict("os.environ", {"HF_TOKEN": second}):
            rendered = publish_hf_space._safe_error(RuntimeError(
                f"failure {first} {second} https://storage.invalid/object?X-Amz-Signature=secret"
            ))
        self.assertNotIn(first, rendered)
        self.assertNotIn(second, rendered)
        self.assertNotIn("X-Amz-Signature", rendered)
        self.assertNotIn("secret", rendered)
        self.assertIn("[REDACTED]", rendered)


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
