"""Tests for GPU mode selection, rank launching and the shadow-rank workflow rewrite.

Run with:  python -m unittest discover -s tests -v

Three things are pinned down here, and all three are about not lying:

  * a worker only runs the dual path when it was asked to AND can; anything else is an
    explicit failure, never a quiet downgrade to single-GPU with dual-GPU labelling;
  * rank 0's command line, directories and environment in dual mode are the same ones the
    known-good single-GPU image uses, so the A/B compares two execution strategies rather
    than two configurations; and
  * the shadow rank's copy of a workflow stops at the VAE decodes, so a confidential
    generation never produces a second plaintext video anywhere on the worker.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("COMFY_INPUT_DIR", tempfile.mkdtemp(prefix="h3-gpu-input-"))
os.environ.setdefault("COMFY_OUTPUT_DIR", tempfile.mkdtemp(prefix="h3-gpu-output-"))
os.environ.setdefault("H3_OUTPUT_MODE", "base64")

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
from h3_parallel import config as gpu_config  # noqa: E402
from h3_parallel import shadow as shadow_graph  # noqa: E402

try:  # h3_parallel.runtime pulls in torch; the container always has it, a laptop may not.
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EnvOverride:
    """Set environment variables for the duration of a block, then put them back."""

    def __init__(self, **values):
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


class ModeSelectionTests(unittest.TestCase):
    """The decision table, which is the whole reason this configuration is explicit."""

    def test_default_is_single(self):
        with EnvOverride(H3_GPU_MODE=None):
            settings = gpu_config.resolve(device_count=2)
        self.assertEqual(settings.mode, gpu_config.SINGLE)
        self.assertEqual(settings.world_size, 1)
        self.assertEqual(settings.strategy, "none")

    def test_two_gpus_alone_do_not_enable_dual(self):
        """Being scheduled two GPUs must never silently change how a job is executed."""
        with EnvOverride(H3_GPU_MODE="single"):
            self.assertFalse(gpu_config.resolve(device_count=8).dual)

    def test_dual_with_two_gpus(self):
        with EnvOverride(H3_GPU_MODE="dual"):
            settings = gpu_config.resolve(device_count=2)
        self.assertTrue(settings.dual)
        self.assertEqual(settings.world_size, 2)
        self.assertEqual(settings.strategy, gpu_config.STRATEGY_ULYSSES_SP)

    def test_dual_with_one_gpu_refuses_to_start(self):
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_ALLOW_FALLBACK=None):
            with self.assertRaises(gpu_config.ConfigurationError) as caught:
                gpu_config.resolve(device_count=1)
        message = str(caught.exception)
        self.assertIn("1 CUDA device", message)
        self.assertIn("H3_SP_ALLOW_FALLBACK", message)

    def test_dual_with_no_gpus_refuses_to_start(self):
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_ALLOW_FALLBACK=None):
            with self.assertRaises(gpu_config.ConfigurationError):
                gpu_config.resolve(device_count=0)

    def test_explicit_fallback_degrades_to_single(self):
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_ALLOW_FALLBACK="1"):
            settings = gpu_config.resolve(device_count=1)
        self.assertEqual(settings.mode, gpu_config.SINGLE)
        self.assertEqual(settings.world_size, 1)

    def test_head_count_must_divide_across_ranks(self):
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_WORLD_SIZE="3"):
            with self.assertRaises(gpu_config.ConfigurationError) as caught:
                gpu_config.resolve(device_count=3, num_heads=56)
        self.assertIn("56 attention heads", str(caught.exception))

    def test_unknown_mode_is_rejected(self):
        with EnvOverride(H3_GPU_MODE="quad"):
            with self.assertRaises(gpu_config.ConfigurationError):
                gpu_config.resolve(device_count=4)

    def test_banner_states_the_strategy(self):
        with EnvOverride(H3_GPU_MODE="dual"):
            settings = gpu_config.resolve(device_count=2, rank=1)
        banner = settings.describe()
        self.assertIn("mode=dual", banner)
        self.assertIn("gpu_count=2", banner)
        self.assertIn(gpu_config.STRATEGY_ULYSSES_SP, banner)

    def test_single_banner_says_single(self):
        with EnvOverride(H3_GPU_MODE="single"):
            self.assertIn("mode=single", gpu_config.resolve(device_count=1).describe())


class RankConfigurationTests(unittest.TestCase):
    def test_a_rank_gets_its_own_port(self):
        self.assertEqual(gpu_config.comfy_port_for_rank(0, 8188), 8188)
        self.assertEqual(gpu_config.comfy_port_for_rank(1, 8188), 8189)

    def test_child_requires_a_rank_in_dual_mode(self):
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_RANK=None):
            with self.assertRaises(gpu_config.ConfigurationError):
                gpu_config.resolve_for_rank()

    def test_child_rank_out_of_range_is_rejected(self):
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_RANK="2", H3_SP_WORLD_SIZE="2"):
            with self.assertRaises(gpu_config.ConfigurationError):
                gpu_config.resolve_for_rank()

    def test_child_does_not_recheck_the_device_count(self):
        # A rank only ever sees its own GPU; the handler is what verified there were two.
        with EnvOverride(H3_GPU_MODE="dual", H3_SP_RANK="1", H3_SP_WORLD_SIZE="2"):
            settings = gpu_config.resolve_for_rank()
        self.assertTrue(settings.dual)
        self.assertEqual(settings.rank, 1)


class RankLaunchTests(unittest.TestCase):
    """Rank 0 must be launched exactly as the single-GPU image launches ComfyUI."""

    def setUp(self):
        self._saved_config = handler.GPU_CONFIG
        with EnvOverride(H3_GPU_MODE="dual"):
            handler.GPU_CONFIG = gpu_config.resolve(device_count=2)

    def tearDown(self):
        handler.GPU_CONFIG = self._saved_config

    def test_rank_zero_keeps_the_known_good_paths(self):
        rank = handler.ComfyRank(0)
        self.assertEqual(rank.port, handler.COMFY_PORT)
        self.assertEqual(rank.output_dir, handler.COMFY_OUTPUT_DIR)
        self.assertEqual(rank.temp_dir, handler.COMFY_TEMP_DIR)
        self.assertIsNone(rank.user_dir, "rank 0 uses the ComfyUI database baked at build")
        self.assertFalse(rank.shadow)

    def test_shadow_ranks_are_isolated(self):
        rank = handler.ComfyRank(1)
        self.assertTrue(rank.shadow)
        self.assertNotEqual(rank.output_dir, handler.COMFY_OUTPUT_DIR)
        self.assertNotEqual(rank.temp_dir, handler.COMFY_TEMP_DIR)
        self.assertIsNotNone(rank.user_dir, "two ComfyUIs must not share one SQLite file")

    def test_every_rank_shares_the_input_directory(self):
        # A staged keyframe is written once and has to be readable by every rank.
        for index in (0, 1):
            command = handler.comfy_command(handler.ComfyRank(index))
            self.assertIn("--input-directory", command)
            self.assertEqual(
                command[command.index("--input-directory") + 1], handler.COMFY_INPUT_DIR
            )

    def test_each_rank_is_pinned_to_one_gpu(self):
        zero = handler.rank_env(handler.ComfyRank(0))
        one = handler.rank_env(handler.ComfyRank(1))
        self.assertEqual(zero["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(one["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(zero["H3_SP_RANK"], "0")
        self.assertEqual(one["H3_SP_RANK"], "1")
        self.assertEqual(zero["H3_SP_WORLD_SIZE"], "2")

    def test_single_mode_leaves_no_rank_marker(self):
        with EnvOverride(H3_GPU_MODE="single", H3_SP_RANK="0"):
            handler.GPU_CONFIG = gpu_config.resolve(device_count=1)
            env = handler.rank_env(handler.ComfyRank(0))
        self.assertNotIn(
            "H3_SP_RANK", env,
            "a stale rank marker would make ComfyUI wait for a peer that never comes",
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env.keys() - os.environ.keys())

    def test_ports_do_not_collide(self):
        ports = {handler.ComfyRank(index).port for index in range(2)}
        self.assertEqual(len(ports), 2)


class ShadowWorkflowTests(unittest.TestCase):
    """The shadow must reach both decodes and must not write a file."""

    def workflow(self):
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "video.safetensors"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "audio.safetensors"}},
            "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"model": ["1", 0]}},
            "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
            "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
            "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0]}},
            "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0]}},
        }

    def test_the_saving_tail_is_removed(self):
        shadow, description = shadow_graph.build_shadow_workflow(self.workflow())
        classes = {node["class_type"] for node in shadow.values()}
        self.assertNotIn("SaveVideo", classes)
        self.assertNotIn("CreateVideo", classes)
        self.assertIn("sink-terminated", description)

    def test_both_decodes_survive_and_are_anchored(self):
        shadow, _ = shadow_graph.build_shadow_workflow(self.workflow())
        sinks = [
            node for node in shadow.values()
            if node["class_type"] == shadow_graph.SINK_CLASS_TYPE
        ]
        self.assertEqual(len(sinks), 2, "each decode needs its own sink or it will not run")
        anchored = {tuple(list(node["inputs"].values())[0]) for node in sinks}
        self.assertEqual(anchored, {("11", 0), ("12", 0)})
        self.assertEqual(
            [list(node["inputs"])[0] for node in sinks], ["images", "audio"]
        )

    def test_the_sampler_and_loaders_are_untouched(self):
        original = self.workflow()
        shadow, _ = shadow_graph.build_shadow_workflow(original)
        for node_id in ("1", "3", "4", "10", "11", "12"):
            self.assertEqual(shadow[node_id], original[node_id])

    def test_no_dangling_links_are_left_behind(self):
        shadow, _ = shadow_graph.build_shadow_workflow(self.workflow())
        for node_id, node in shadow.items():
            for value in node["inputs"].values():
                if shadow_graph._is_link(value):
                    self.assertIn(
                        str(value[0]), shadow,
                        f"node {node_id} points at a node that was pruned away",
                    )

    def test_the_callers_workflow_is_never_mutated(self):
        original = self.workflow()
        snapshot = json.dumps(original, sort_keys=True)
        shadow_graph.build_shadow_workflow(original)
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot)

    def test_a_workflow_with_no_decode_falls_back_to_a_verbatim_copy(self):
        # Better a duplicated render the handler then deletes than a deadlocked rank 0.
        odd = {"1": {"class_type": "UNETLoader", "inputs": {}}}
        shadow, description = shadow_graph.build_shadow_workflow(odd)
        self.assertEqual(shadow, odd)
        self.assertIn("verbatim", description)

    def test_an_unrecognised_output_node_still_leaves_a_runnable_graph(self):
        workflow = self.workflow()
        workflow["15"] = {"class_type": "SomeVendorSaveNode", "inputs": {"images": ["11", 0]}}
        shadow, _ = shadow_graph.build_shadow_workflow(workflow)
        # It survives - which is why the handler also empties the shadow's output
        # directory after every job.
        self.assertIn("15", shadow)
        self.assertIn("11", shadow)

    def test_image_to_video_keyframe_graphs_survive(self):
        workflow = self.workflow()
        workflow["20"] = {"class_type": "LoadImage", "inputs": {"image": "frame.png"}}
        workflow["10"]["inputs"]["latent_image"] = ["20", 0]
        shadow, _ = shadow_graph.build_shadow_workflow(workflow)
        self.assertIn("20", shadow)
        self.assertEqual(shadow["20"]["inputs"]["image"], "frame.png")


class ShadowSubmissionTests(unittest.TestCase):
    """What the handler does with those shadow graphs."""

    def setUp(self):
        self._saved_config = handler.GPU_CONFIG
        self._saved_shadows = handler._shadow_ranks
        self._saved_post = handler.requests.post
        with EnvOverride(H3_GPU_MODE="dual"):
            handler.GPU_CONFIG = gpu_config.resolve(device_count=2)

    def tearDown(self):
        handler.GPU_CONFIG = self._saved_config
        handler._shadow_ranks = self._saved_shadows
        handler.requests.post = self._saved_post

    def _capture_posts(self, status_code=200, payload=None, raise_error=None):
        calls: list[tuple[str, dict]] = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            calls.append((url, json))
            if raise_error is not None:
                raise raise_error
            return types.SimpleNamespace(
                status_code=status_code,
                json=lambda: payload if payload is not None else {"prompt_id": "p"},
                text="",
            )

        handler.requests.post = fake_post
        return calls

    def test_nothing_is_submitted_when_there_are_no_shadows(self):
        handler._shadow_ranks = []
        calls = self._capture_posts()
        handler._queue_on_shadows({"1": {"class_type": "VAEDecode", "inputs": {}}}, "cid")
        self.assertEqual(calls, [])

    def test_each_shadow_receives_the_rewritten_graph(self):
        handler._shadow_ranks = [handler.ComfyRank(1)]
        calls = self._capture_posts()
        handler._queue_on_shadows(
            {
                "10": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
                "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0]}},
                "14": {"class_type": "SaveVideo", "inputs": {"video": ["11", 0]}},
            },
            "cid",
        )
        self.assertEqual(len(calls), 1)
        url, body = calls[0]
        self.assertTrue(url.endswith(":8189/prompt"))
        classes = {node["class_type"] for node in body["prompt"].values()}
        self.assertNotIn("SaveVideo", classes)
        self.assertIn(shadow_graph.SINK_CLASS_TYPE, classes)

    def test_a_shadow_that_rejects_the_graph_fails_the_job(self):
        handler._shadow_ranks = [handler.ComfyRank(1)]
        self._capture_posts(status_code=400, payload={"error": "bad node"})
        with self.assertRaises(handler.WorkflowError) as caught:
            handler._queue_on_shadows({"11": {"class_type": "VAEDecode", "inputs": {}}}, "cid")
        self.assertIn("rejected the shadow workflow", str(caught.exception))

    def test_an_unreachable_shadow_fails_the_job_rather_than_deadlocking(self):
        handler._shadow_ranks = [handler.ComfyRank(1)]
        self._capture_posts(raise_error=handler.requests.RequestException("connection refused"))
        with self.assertRaises(handler.WorkflowError) as caught:
            handler._queue_on_shadows({"11": {"class_type": "VAEDecode", "inputs": {}}}, "cid")
        self.assertIn("deadlock", str(caught.exception))


class ShadowCleanupTests(unittest.TestCase):
    """A shadow must not carry anything across a job boundary."""

    def setUp(self):
        self._saved_shadows = handler._shadow_ranks
        self._saved_post = handler.requests.post
        self.directory = tempfile.mkdtemp(prefix="h3-shadow-out-")
        rank = handler.ComfyRank(1)
        rank.output_dir = self.directory
        rank.temp_dir = self.directory
        handler._shadow_ranks = [rank]
        handler.requests.post = lambda *a, **k: types.SimpleNamespace(status_code=200)

    def tearDown(self):
        handler._shadow_ranks = self._saved_shadows
        handler.requests.post = self._saved_post

    def test_leftover_media_is_deleted(self):
        stray = os.path.join(self.directory, "MiniMaxH3_00001.mp4")
        with open(stray, "wb") as handle:
            handle.write(b"plaintext video")
        nested = os.path.join(self.directory, "video")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, "another.mp4"), "wb") as handle:
            handle.write(b"more plaintext")

        handler._drain_shadow_ranks(reason="ok")

        self.assertFalse(os.path.exists(stray))
        self.assertFalse(os.path.exists(nested))
        self.assertTrue(os.path.isdir(self.directory), "the directory itself must survive")

    def test_cleanup_survives_a_missing_directory(self):
        rank = handler._shadow_ranks[0]
        rank.output_dir = os.path.join(self.directory, "does-not-exist")
        rank.temp_dir = rank.output_dir
        handler._drain_shadow_ranks(reason="error")  # must not raise


class PerfLineTests(unittest.TestCase):
    def test_the_perf_line_names_the_execution_mode(self):
        saved = handler.GPU_CONFIG
        try:
            with EnvOverride(H3_GPU_MODE="dual"):
                handler.GPU_CONFIG = gpu_config.resolve(device_count=2)
            line = handler.JobTimer({}).summary(job_index=1, status="ok")
            self.assertIn("gpu_mode=dual", line)
            self.assertIn("gpu_count=2", line)
            self.assertIn(f"strategy={gpu_config.STRATEGY_ULYSSES_SP}", line)
        finally:
            handler.GPU_CONFIG = saved

    def test_single_mode_says_so(self):
        saved = handler.GPU_CONFIG
        try:
            with EnvOverride(H3_GPU_MODE="single"):
                handler.GPU_CONFIG = gpu_config.resolve(device_count=1)
            line = handler.JobTimer({}).summary(job_index=1, status="ok")
            self.assertIn("gpu_mode=single", line)
            self.assertIn("gpu_count=1", line)
        finally:
            handler.GPU_CONFIG = saved

    def test_gpu_fields_are_empty_in_single_mode(self):
        saved = handler.GPU_CONFIG
        try:
            with EnvOverride(H3_GPU_MODE="single"):
                handler.GPU_CONFIG = gpu_config.resolve(device_count=1)
            self.assertEqual(handler.gpu_perf_fields(), [])
        finally:
            handler.GPU_CONFIG = saved


class RankVerificationTests(unittest.TestCase):
    """The guard that stops a broken dual setup from being benchmarked as a working one."""

    def setUp(self):
        self._saved_config = handler.GPU_CONFIG
        self._saved_ranks = handler._ranks
        with EnvOverride(H3_GPU_MODE="dual"):
            handler.GPU_CONFIG = gpu_config.resolve(device_count=2)

    def tearDown(self):
        handler.GPU_CONFIG = self._saved_config
        handler._ranks = self._saved_ranks

    def _ranks_reporting(self, *statuses):
        ranks = []
        for index, status in enumerate(statuses):
            rank = handler.ComfyRank(index)
            if isinstance(status, Exception):
                rank.gpu_status = lambda *a, _error=status, **k: (_ for _ in ()).throw(_error)
            else:
                rank.gpu_status = lambda *a, _status=status, **k: _status
            ranks.append(rank)
        handler._ranks = ranks

    def _ready(self):
        return {
            "ready": True,
            "strategy": gpu_config.STRATEGY_ULYSSES_SP,
            "world_size": 2,
            "patched": {"dit": True, "attention": True, "vae_decode": True},
            "selftest": "selftest=pass",
        }

    def test_two_ready_ranks_pass(self):
        self._ranks_reporting(self._ready(), self._ready())
        handler._verify_dual_ranks()  # must not raise

    def test_a_rank_that_never_loaded_the_node_is_fatal(self):
        self._ranks_reporting(self._ready(), ConnectionError("404"))
        with self.assertRaises(RuntimeError) as caught:
            handler._verify_dual_ranks()
        self.assertIn("not loaded", str(caught.exception))

    def test_a_rank_whose_selftest_failed_is_fatal(self):
        failed = {"ready": False, "error": "selftest FAILED: max abs error 3.1"}
        self._ranks_reporting(self._ready(), failed)
        with self.assertRaises(RuntimeError) as caught:
            handler._verify_dual_ranks()
        self.assertIn("NOT ready", str(caught.exception))

    def test_a_rank_that_came_up_unpatched_is_fatal(self):
        unpatched = dict(self._ready(), patched={"dit": False, "attention": False})
        self._ranks_reporting(self._ready(), unpatched)
        with self.assertRaises(RuntimeError) as caught:
            handler._verify_dual_ranks()
        self.assertIn("not patched", str(caught.exception))


class TransportLadderTests(unittest.TestCase):
    """Picking an NCCL transport by trying, rather than by making the operator guess."""

    def test_default_ladder_ends_somewhere_that_always_works(self):
        with EnvOverride(H3_SP_TRANSPORT=None, NCCL_P2P_DISABLE=None, NCCL_SHM_DISABLE=None,
                         NCCL_NET=None):
            ladder = gpu_config.transport_ladder()
        self.assertEqual([name for name, _, _ in ladder], ["auto", "no-shm", "no-p2p", "sockets"])
        # The last rung disables both GPU transports, leaving loopback sockets - slow, but
        # available in any container. A slow measurement beats no measurement.
        self.assertEqual(ladder[-1][1], {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"})

    def test_the_first_rung_changes_nothing(self):
        with EnvOverride(H3_SP_TRANSPORT=None, NCCL_P2P_DISABLE=None, NCCL_SHM_DISABLE=None,
                         NCCL_NET=None):
            self.assertEqual(gpu_config.transport_ladder()[0][1], {})

    def test_an_operator_nccl_setting_is_never_second_guessed(self):
        with EnvOverride(H3_SP_TRANSPORT=None, NCCL_SHM_DISABLE="1", NCCL_P2P_DISABLE=None,
                         NCCL_NET=None):
            ladder = gpu_config.transport_ladder()
        self.assertEqual(len(ladder), 1)
        self.assertEqual(ladder[0][0], "operator-pinned")
        self.assertIn("NCCL_SHM_DISABLE", ladder[0][2])

    def test_a_named_transport_is_the_only_one_tried(self):
        with EnvOverride(H3_SP_TRANSPORT="sockets", NCCL_P2P_DISABLE=None,
                         NCCL_SHM_DISABLE=None, NCCL_NET=None):
            ladder = gpu_config.transport_ladder()
        self.assertEqual(len(ladder), 1)
        self.assertEqual(ladder[0][0], "sockets")

    def test_an_unknown_transport_is_rejected(self):
        with EnvOverride(H3_SP_TRANSPORT="infiniband", NCCL_P2P_DISABLE=None,
                         NCCL_SHM_DISABLE=None, NCCL_NET=None):
            with self.assertRaises(gpu_config.ConfigurationError):
                gpu_config.transport_ladder()

    def test_the_overlay_reaches_the_rank_environment(self):
        saved = handler.GPU_CONFIG
        try:
            with EnvOverride(H3_GPU_MODE="dual", NCCL_DEBUG=None):
                handler.GPU_CONFIG = gpu_config.resolve(device_count=2)
                first = handler.rank_env(handler.ComfyRank(0), ("auto", {}, ""))
                later = handler.rank_env(
                    handler.ComfyRank(1), ("sockets", {"NCCL_P2P_DISABLE": "1"}, "")
                )
        finally:
            handler.GPU_CONFIG = saved

        self.assertEqual(first["H3_SP_TRANSPORT_LABEL"], "auto")
        self.assertNotIn("NCCL_P2P_DISABLE", first)
        self.assertEqual(later["NCCL_P2P_DISABLE"], "1")
        self.assertEqual(later["H3_SP_TRANSPORT_LABEL"], "sockets")
        # Reaching a second rung means something already went wrong; stop being quiet.
        self.assertEqual(later["NCCL_DEBUG"], "INFO")

    def test_an_operators_nccl_debug_choice_is_kept(self):
        saved = handler.GPU_CONFIG
        try:
            with EnvOverride(H3_GPU_MODE="dual", NCCL_DEBUG="WARN"):
                handler.GPU_CONFIG = gpu_config.resolve(device_count=2)
                env = handler.rank_env(
                    handler.ComfyRank(1), ("sockets", {"NCCL_P2P_DISABLE": "1"}, "")
                )
        finally:
            handler.GPU_CONFIG = saved
        self.assertEqual(env["NCCL_DEBUG"], "WARN")

    @unittest.skipUnless(HAS_TORCH, "the probe lives in h3_parallel.runtime")
    def test_the_stall_exit_code_is_shared_with_the_rank(self):
        from h3_parallel import runtime

        self.assertEqual(handler.TRANSPORT_STALLED_EXIT, runtime.EXIT_TRANSPORT_STALLED)


@unittest.skipUnless(HAS_TORCH, "the probe needs torch")
class TransportProbeTests(unittest.TestCase):
    """The probe exists so a stalled collective is loud in seconds, not silent for eight minutes."""

    def test_a_working_transport_reports_its_timings(self):
        from h3_parallel import runtime
        from h3_parallel.collectives import ThreadCollective, ThreadGroup

        group = ThreadGroup(1)
        result = runtime.probe_transport(ThreadCollective(group, 0), timeout=30)
        self.assertIn("transport=ok", result)
        for collective in ("barrier", "broadcast", "all_to_all"):
            self.assertIn(f"{collective}=", result)

    def test_a_wrong_answer_is_raised_not_ignored(self):
        from h3_parallel import runtime

        class Liar:
            rank, world_size = 0, 1

            def barrier(self):
                pass

            def broadcast(self, tensor, src):
                tensor.fill_(99.0)  # not what rank 0 sent
                return tensor

            def all_to_all(self, send, input_splits, output_splits):
                return send

        with self.assertRaises(RuntimeError) as caught:
            runtime.probe_transport(Liar(), timeout=30)
        self.assertIn("broadcast", str(caught.exception))

    def test_a_stalled_collective_exits_fast_with_a_diagnostic(self):
        # The real production failure: a collective that never returns. Run in a
        # subprocess because the fix is a hard exit - the thread is stuck inside NCCL and
        # cannot be joined, so nothing else can free the process.
        import subprocess

        script = (
            "import sys, time\n"
            f"sys.path.insert(0, {self.ROOT!r})\n"
            "from h3_parallel import runtime\n"
            "class Stuck:\n"
            "    rank, world_size = 0, 1\n"
            "    def barrier(self):\n"
            "        time.sleep(600)\n"
            "runtime.probe_transport(Stuck(), timeout=2)\n"
            "print('UNREACHABLE')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(
            result.returncode, 97,
            f"expected the stall exit code\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertNotIn("UNREACHABLE", result.stdout)
        self.assertIn("stalled_on=barrier", result.stdout)
        self.assertIn("nccl_env=", result.stdout)

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_diagnostics_never_raise(self):
        from h3_parallel import runtime

        lines = runtime.transport_diagnostics()
        self.assertTrue(any("nccl_env" in line for line in lines))
        self.assertTrue(any(line.startswith("shm=") for line in lines))


class CustomNodeEntryPointTests(unittest.TestCase):
    """The ComfyUI side of the wiring, exercised the way ComfyUI actually loads it."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(self, script: str, **env_overrides) -> dict:
        import subprocess

        env = dict(os.environ)
        env.pop("H3_SP_RANK", None)
        env["PYTHONPATH"] = self.ROOT
        env["H3_SERVERLESS_DIR"] = self.ROOT
        env.update({key: value for key, value in env_overrides.items() if value is not None})
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, cwd=self.ROOT, timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_the_shim_loads_the_package_and_registers_nothing_in_single_mode(self):
        # Exactly what ComfyUI does: exec the file in custom_nodes and read the mappings.
        report = self._run(
            "import importlib.util, json, os, sys\n"
            "path = os.path.join(os.environ['H3_SERVERLESS_DIR'],"
            " 'comfy_custom_nodes', 'h3_parallel_boot.py')\n"
            "spec = importlib.util.spec_from_file_location('h3_parallel_boot', path)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "print(json.dumps({'nodes': sorted(module.NODE_CLASS_MAPPINGS)}))\n",
            H3_GPU_MODE="single",
        )
        self.assertEqual(
            report["nodes"], [],
            "single mode must add no nodes at all - the image has to behave like its predecessor",
        )

    @unittest.skipUnless(HAS_TORCH, "h3_parallel.runtime imports torch")
    def test_a_rank_registers_the_shadow_sink_and_reports_ready(self):
        report = self._run(
            "import json, h3_parallel\n"
            "from h3_parallel import runtime\n"
            "print(json.dumps({'nodes': sorted(h3_parallel.NODE_CLASS_MAPPINGS),"
            " 'ready': runtime.STATUS['ready'], 'mode': runtime.STATUS['mode']}))\n",
            # A rank marker with single mode: the group is never formed and nothing is
            # patched, which is what makes this runnable on a machine with no GPU.
            H3_GPU_MODE="single",
            H3_SP_RANK="0",
            H3_SP_WORLD_SIZE="1",
        )
        self.assertEqual(report["nodes"], ["H3ParallelSink"])
        self.assertTrue(report["ready"])
        self.assertEqual(report["mode"], "single")

    @unittest.skipUnless(HAS_TORCH, "h3_parallel.runtime imports torch")
    def test_a_rank_that_cannot_form_a_group_reports_not_ready(self):
        # No CUDA here, so joining a NCCL group must fail. Reproduces exactly what ComfyUI
        # does with that failure - swallow the import error - and asserts the rank still
        # records why, because /h3/gpu is the handler's only way to find out.
        report = self._run(
            "import json, sys\n"
            "raised = False\n"
            "try:\n"
            "    import h3_parallel\n"
            "except BaseException:\n"
            "    raised = True\n"
            "runtime = sys.modules.get('h3_parallel.runtime')\n"
            "print(json.dumps({'raised': raised, 'ready': runtime.STATUS['ready'],"
            " 'error': (runtime.STATUS['error'] or '')[:120]}))\n",
            H3_GPU_MODE="dual",
            H3_SP_RANK="0",
            H3_SP_WORLD_SIZE="2",
            H3_SP_INIT_TIMEOUT="5",
        )
        self.assertTrue(report["raised"], "a rank that cannot join must fail its import")
        self.assertFalse(report["ready"])
        self.assertIn("CUDA", report["error"], "the recorded reason must name the real cause")


if __name__ == "__main__":
    unittest.main()
