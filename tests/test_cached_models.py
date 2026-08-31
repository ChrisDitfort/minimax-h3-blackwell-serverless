from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bootstrap
import cached_models


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "h3-cache-manifest.json"


class CacheFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.cache_root = base / "hub"
        self.models_root = base / "comfy-models"
        self.repo_root = self.cache_root / "models--CDitfort--privora-minimax-h3-models"
        self.snapshot = self.repo_root / "snapshots" / cached_models.EXPECTED_REVISION
        self.manifest = cached_models.load_manifest(MANIFEST_PATH)

    def close(self):
        self.temp.cleanup()

    def materialize(self, *, missing: str | None = None):
        self.snapshot.mkdir(parents=True)
        for asset in self.manifest.assets:
            if asset.path == missing:
                continue
            path = self.snapshot.joinpath(*asset.path.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cache-metadata-fixture")

    def size_reader(self, path: Path) -> int:
        relative = path.relative_to(self.snapshot).as_posix()
        return next(asset.expected_bytes for asset in self.manifest.assets if asset.path == relative)

    def activate(self, **kwargs):
        return cached_models.activate_cached_models(
            MANIFEST_PATH,
            environment={},
            cache_root=self.cache_root,
            comfy_models_root=self.models_root,
            size_reader=self.size_reader,
            **kwargs,
        )


class ManifestTests(unittest.TestCase):
    def test_manifest_is_the_exact_verified_multimodal_4_inventory(self):
        manifest = cached_models.load_manifest(MANIFEST_PATH)
        self.assertEqual(manifest.repository, cached_models.EXPECTED_REPOSITORY)
        self.assertEqual(manifest.revision, cached_models.EXPECTED_REVISION)
        self.assertEqual(manifest.manifest_version, "multimodal-4-hf-cache-v1")
        self.assertEqual(manifest.total_bytes, 69_309_544_079)
        self.assertEqual(tuple(asset.path for asset in manifest.assets), cached_models.EXPECTED_ASSET_PATHS)
        self.assertEqual(
            tuple(
                (asset.asset_id, asset.path, asset.expected_bytes, asset.expected_sha256,
                 asset.role, asset.family, asset.source_revision)
                for asset in manifest.assets
            ),
            cached_models.EXPECTED_ASSET_RECORDS,
        )
        self.assertTrue(all(asset.source_repository == "Comfy-Org/MiniMax-H3" for asset in manifest.assets))
        self.assertIn(
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            cached_models.EXPECTED_ASSET_PATHS,
        )

    def test_changed_manifest_identity_is_rejected(self):
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        raw["manifestVersion"] = "mutable-or-wrong"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(cached_models.CachedModelError, "identity"):
                cached_models.load_manifest(path)

    def test_changed_known_hash_is_rejected_even_if_identity_and_sizes_stay_the_same(self):
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        raw["assets"][0]["expectedSha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(cached_models.CachedModelError, "metadata"):
                cached_models.load_manifest(path)


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CacheFixture()

    def tearDown(self):
        self.fixture.close()

    def test_missing_cache_root_fails_closed(self):
        with self.assertRaisesRegex(cached_models.CachedModelError, "cache root"):
            self.fixture.activate()

    def test_missing_repository_fails_closed(self):
        self.fixture.cache_root.mkdir()
        with self.assertRaisesRegex(cached_models.CachedModelError, "repository"):
            self.fixture.activate()

    def test_refs_main_is_never_used_as_a_snapshot(self):
        mutable = self.fixture.repo_root / "refs" / "main"
        mutable.parent.mkdir(parents=True)
        mutable.write_text(cached_models.EXPECTED_REVISION, encoding="utf-8")
        with self.assertRaisesRegex(cached_models.CachedModelError, "revision"):
            self.fixture.activate()

    def test_missing_one_asset_fails_before_creating_links(self):
        missing = cached_models.EXPECTED_ASSET_PATHS[-1]
        self.fixture.materialize(missing=missing)
        with self.assertRaisesRegex(cached_models.CachedModelError, missing):
            self.fixture.activate()
        self.assertFalse(self.fixture.models_root.exists())

    def test_size_mismatch_fails_before_creating_links(self):
        self.fixture.materialize()
        first = cached_models.EXPECTED_ASSET_PATHS[0]

        def wrong_size(path: Path) -> int:
            return 1 if path.relative_to(self.fixture.snapshot).as_posix() == first else self.fixture.size_reader(path)

        with self.assertRaisesRegex(cached_models.CachedModelError, "size mismatch"):
            cached_models.activate_cached_models(
                MANIFEST_PATH,
                environment={},
                cache_root=self.fixture.cache_root,
                comfy_models_root=self.fixture.models_root,
                size_reader=wrong_size,
            )
        self.assertFalse(self.fixture.models_root.exists())

    @unittest.skipIf(os.name == "nt", "Windows host does not grant symbolic-link privilege")
    def test_exact_snapshot_creates_all_eight_links_and_compatibility_alias(self):
        self.fixture.materialize()
        result = self.fixture.activate()
        self.assertEqual(result.asset_count, 8)
        self.assertEqual(result.total_bytes, 69_309_544_079)
        for relative in cached_models.EXPECTED_ASSET_PATHS:
            destination = self.fixture.models_root.joinpath(*relative.split("/"))
            source = self.fixture.snapshot.joinpath(*relative.split("/"))
            self.assertTrue(destination.is_symlink(), relative)
            self.assertEqual(os.readlink(destination), str(source))
        alias = (
            self.fixture.models_root
            / "diffusion_models/h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        )
        canonical = self.fixture.models_root.joinpath(*cached_models.EXPECTED_ASSET_PATHS[0].split("/"))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(os.readlink(alias), str(canonical))

    @unittest.skipIf(os.name == "nt", "Windows host does not grant symbolic-link privilege")
    def test_second_activation_is_idempotent_and_writes_nothing(self):
        self.fixture.materialize()
        first = self.fixture.activate()
        second = self.fixture.activate()
        self.assertEqual(len(first.created_links), 9)
        self.assertEqual(second.created_links, ())
        self.assertEqual(set(second.existing_links), set(cached_models.EXPECTED_ASSET_PATHS))

    def test_all_eight_server_controlled_mappings_are_requested(self):
        self.fixture.materialize()
        created = []
        with mock.patch.object(cached_models, "_link_state", return_value="missing"), mock.patch.object(
            cached_models.os, "symlink", side_effect=lambda source, destination: created.append(
                (str(source), str(destination))
            )
        ):
            result = self.fixture.activate()
        self.assertEqual(len(result.created_links), 9)
        self.assertEqual(len(created), 9)
        for relative in cached_models.EXPECTED_ASSET_PATHS:
            expected_source = str(self.fixture.snapshot.joinpath(*relative.split("/")))
            expected_destination = str(self.fixture.models_root.joinpath(*relative.split("/")))
            self.assertIn((expected_source, expected_destination), created)

    def test_idempotent_preexisting_links_issue_zero_symlink_calls(self):
        self.fixture.materialize()

        def all_existing(destination: Path, expected_target: Path) -> str:
            return "existing"

        with mock.patch.object(cached_models, "_link_state", side_effect=all_existing), mock.patch.object(
            cached_models.os, "symlink"
        ) as symlink:
            result = self.fixture.activate()
        symlink.assert_not_called()
        self.assertEqual(result.created_links, ())
        self.assertEqual(set(result.existing_links), set(cached_models.EXPECTED_ASSET_PATHS))

    def test_wrong_existing_destination_is_not_overwritten(self):
        self.fixture.materialize()
        destination = self.fixture.models_root.joinpath(*cached_models.EXPECTED_ASSET_PATHS[0].split("/"))
        destination.parent.mkdir(parents=True)
        destination.write_text("do not replace", encoding="utf-8")
        with self.assertRaisesRegex(cached_models.CachedModelError, "Refusing to replace"):
            self.fixture.activate()
        self.assertEqual(destination.read_text(encoding="utf-8"), "do not replace")

    def test_operator_cannot_select_another_repo_revision_or_manifest(self):
        self.fixture.materialize()
        for name, value in (
            ("H3_MODEL_REPO", "someone/other"),
            ("H3_MODEL_REVISION", "0" * 40),
            ("H3_MODEL_MANIFEST_VERSION", "other"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(cached_models.CachedModelError, name):
                    cached_models.activate_cached_models(
                        MANIFEST_PATH,
                        environment={name: value},
                        cache_root=self.fixture.cache_root,
                        comfy_models_root=self.fixture.models_root,
                        size_reader=self.fixture.size_reader,
                    )


class OfflineBootstrapTests(unittest.TestCase):
    def test_bootstrap_forces_offline_mode_before_resolution(self):
        with mock.patch.object(bootstrap, "MANIFEST_PATH", "fixture-manifest"), mock.patch.object(
            bootstrap, "_load_build_identity"
        ), mock.patch.object(
            bootstrap, "activate_cached_models", side_effect=cached_models.CachedModelError("stop")
        ), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(bootstrap.main(), 1)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")

    def test_resolver_has_no_hub_or_network_client(self):
        source = (ROOT / "cached_models.py").read_text(encoding="utf-8")
        self.assertNotIn("huggingface_hub", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("urlopen", source)
        self.assertNotIn("snapshot_download", source)


if __name__ == "__main__":
    unittest.main()
