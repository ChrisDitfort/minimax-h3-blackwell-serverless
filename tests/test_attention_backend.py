"""Tests for the SageAttention3 compute-capability guard in build_comfy_env().

Run with:  python -m unittest discover -s tests -v

The sageattn3 wheel in the base image is compiled for one compute capability (sm_120).
Importing it succeeds on any GPU, so the mismatch only shows up as a
`cudaErrorNoKernelImageForDevice` on every attention call at generation time - which no
local test can reproduce. These tests stub torch instead, so the decision table is
pinned down without a GPU, ComfyUI or a RunPod account.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

# handler.py reads its directories at import time, so point it at a scratch dir first.
os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-attn-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-attn-output-"))
os.environ.setdefault("H3_OUTPUT_MODE", "base64")

# Stub the two dependencies that only exist inside the container image.
if "runpod" not in sys.modules:
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = runpod_stub

if "websocket" not in sys.modules:
    websocket_stub = types.ModuleType("websocket")
    websocket_stub.WebSocketTimeoutException = type("WebSocketTimeoutException", (Exception,), {})
    websocket_stub.create_connection = lambda *a, **k: None
    sys.modules["websocket"] = websocket_stub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402

SAGE_KEYS = ("COMFY_SAGE_ATTENTION3", "H3_SAGE_AUTODETECT", "SAGE_SUPPORTED_CC")


class AttentionBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {key: os.environ.get(key) for key in SAGE_KEYS}
        self._saved_modules = {name: sys.modules.get(name) for name in ("torch", "sageattn3")}
        for key in SAGE_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    # -- helpers ----------------------------------------------------------------------

    def set_gpu(self, capability: tuple[int, int] | None) -> None:
        """Stub torch reporting `capability`, or None for 'no CUDA visible'."""
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(
            is_available=lambda: capability is not None,
            get_device_capability=lambda index: capability,
        )
        sys.modules["torch"] = torch_stub

    def set_sage(self, present: bool) -> None:
        if present:
            sys.modules["sageattn3"] = types.ModuleType("sageattn3")
        else:
            # A None entry makes `import sageattn3` raise ImportError deterministically,
            # whether or not the real package is installed where the tests run.
            sys.modules["sageattn3"] = None

    def resolve(self, **env: str) -> str:
        os.environ.update(env)
        return handler.build_comfy_env().get("COMFY_SAGE_ATTENTION3", "<unset>")

    # -- the decision table -----------------------------------------------------------

    def test_disabled_by_env_stays_disabled(self):
        self.set_gpu((12, 0))
        self.set_sage(True)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="0"), "0")

    def test_blackwell_sm120_keeps_sage_enabled(self):
        self.set_gpu((12, 0))
        self.set_sage(True)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1"), "1")

    def test_missing_version_metadata_does_not_disable_sage(self):
        """A version string is not a capability check.

        sageattn3 imports fine here but has no installed distribution metadata, so
        importlib.metadata.version() raises. That must not turn the backend off on a
        card the wheel was actually built for.
        """
        import importlib.metadata

        self.set_gpu((12, 0))
        self.set_sage(True)
        with self.assertRaises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.version("sageattn3")
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1"), "1")

    def test_non_blackwell_gpus_disable_sage(self):
        # B200 sm_100, H100 sm_90, A100 sm_80 - all import sageattn3 fine and all would
        # fail on every attention call.
        for capability in ((10, 0), (9, 0), (8, 0)):
            with self.subTest(capability=capability):
                for key in SAGE_KEYS:
                    os.environ.pop(key, None)
                self.set_gpu(capability)
                self.set_sage(True)
                self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1"), "0")

    def test_unreadable_gpu_leaves_setting_alone(self):
        self.set_gpu(None)
        self.set_sage(True)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1"), "1")

    def test_autodetect_off_forces_the_configured_value(self):
        self.set_gpu((9, 0))
        self.set_sage(True)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1", H3_SAGE_AUTODETECT="0"), "1")

    def test_supported_cc_override_is_honoured(self):
        self.set_gpu((10, 0))
        self.set_sage(True)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1", SAGE_SUPPORTED_CC="10.0"), "1")

    def test_unparseable_supported_cc_falls_back_to_12(self):
        self.set_gpu((12, 0))
        self.set_sage(True)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1", SAGE_SUPPORTED_CC="sm_120"), "1")

    def test_unimportable_sage_is_disabled_so_comfyui_can_boot(self):
        self.set_gpu((12, 0))
        self.set_sage(False)
        self.assertEqual(self.resolve(COMFY_SAGE_ATTENTION3="1"), "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
