"""Tests for the ComfyUI patch layer that turns H3 into a two-GPU model.

Run with:  python -m unittest discover -s tests -v

test_ulysses.py proves the sharding algebra. This file proves the *plumbing*: that the
patches attach to ComfyUI's real extension points correctly, that the sharded run produces
the same tensors as the unsharded one end to end, and that the work is genuinely split
rather than duplicated on both ranks.

The model under test is tests/h3_stub.py, which reproduces the calling conventions of
comfy/ldm/minimax/model.py exactly (see its docstring). The patches applied to it are the
production ones from h3_parallel/patches.py, unmodified.

In the container each rank is a separate process, so the patched module holds exactly one
collective. Here every rank is a thread in one process sharing one patched module, so the
collective handed to `install` is a router that dispatches to the calling thread's rank
(FanOutCollective). That indirection is confined to this file - production code sees an
ordinary Collective.
"""

from __future__ import annotations

import copy
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    import h3_stub
    from h3_parallel import patches, ulysses
    from h3_parallel.collectives import Collective, ThreadCollective, ThreadGroup


class FanOutCollective(Collective if torch is not None else object):
    """One Collective object, per-thread rank. Stands in for one process per rank."""

    def __init__(self, group):
        self._group = group
        self.world_size = group.world_size
        self._by_thread: dict[int, ThreadCollective] = {}

    def bind(self, rank):
        self._by_thread[threading.get_ident()] = ThreadCollective(self._group, rank)

    @property
    def _current(self):
        return self._by_thread[threading.get_ident()]

    @property
    def rank(self):
        return self._current.rank

    def all_to_all(self, send, input_splits, output_splits):
        return self._current.all_to_all(send, input_splits, output_splits)

    def broadcast(self, tensor, src):
        return self._current.broadcast(tensor, src)

    def barrier(self):
        return self._current.barrier()


class PatchedStub:
    """Install the production patches onto the stub module, then restore it."""

    def __init__(self, world_size, parallel_vae=False):
        self.group = ThreadGroup(world_size)
        self.collective = FanOutCollective(self.group)
        self.world_size = world_size
        self.parallel_vae = parallel_vae

    def __enter__(self):
        self._saved = (
            h3_stub.optimized_attention,
            h3_stub.MiniMaxH3Model.forward,
            h3_stub.MiniMaxH3VideoVAE.decode,
            h3_stub.MiniMaxH3VideoVAE._adaptive_decode,
        )
        h3_stub._h3_sequence_parallel_installed = False
        self.report = patches.install(
            self.collective,
            parallel_vae=self.parallel_vae,
            log=lambda message: None,
            model_module=h3_stub,
            vae_module=h3_stub,
        )
        return self

    def __exit__(self, *exc):
        (
            h3_stub.optimized_attention,
            h3_stub.MiniMaxH3Model.forward,
            h3_stub.MiniMaxH3VideoVAE.decode,
            h3_stub.MiniMaxH3VideoVAE._adaptive_decode,
        ) = self._saved
        h3_stub._h3_sequence_parallel_installed = False
        return False

    def run(self, body):
        """Run `body(rank)` on one thread per rank and return the results in rank order."""
        results: list = [None] * self.world_size
        errors: list = [None] * self.world_size

        def target(rank):
            try:
                self.collective.bind(rank)
                results[rank] = body(rank)
            except BaseException as error:  # noqa: BLE001
                errors[rank] = error
                self.group.barrier.abort()

        threads = [
            threading.Thread(target=target, args=(rank,)) for rank in range(self.world_size)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        for thread in threads:
            if thread.is_alive():
                raise AssertionError("a rank deadlocked; collectives are out of step")
        for error in errors:
            if error is not None:
                raise error
        return results


def build_payload(seq_len, hidden):
    """The pieces MiniMaxH3Model._forward hands to every block."""
    torch.manual_seed(5)
    # Contiguous (start, stop, modulation row) segments covering the whole sequence, the
    # way PackedLayout emits [text | cond | audio | video].
    text = seq_len // 8
    audio = seq_len // 4
    return {
        "t_emb": torch.randn(3, hidden * 6),
        "mod_segments": [
            (0, text, 0),
            (text, text + audio, 1),
            (text + audio, seq_len, 2),
        ],
        "rope_freqs": torch.randn(1, seq_len, 1, 4, 2, 2),
        "video_seg": (text + audio, seq_len, 2),
        "audio_seg": (text, text + audio, 1),
    }


def build_model(hidden=32, heads=4, head_dim=8, layers=3):
    torch.manual_seed(17)
    model = h3_stub.MiniMaxH3Model(
        hidden=hidden, num_attention_heads=heads, attention_head_dim=head_dim, num_layers=layers
    )
    model.eval()
    return model


def run_model(model, x, payload):
    with torch.no_grad():
        return h3_stub.MiniMaxH3Model.forward(
            model, x.clone(), None, None, {}, minimax_payload=copy.deepcopy(payload)
        )


@unittest.skipIf(torch is None, "torch is not installed")
class EndToEndEquivalenceTests(unittest.TestCase):
    """The sharded model must compute the same video and audio latents as the whole one."""

    def _reference(self, seq_len=61, layers=3):
        model = build_model(layers=layers)
        payload = build_payload(seq_len, model.hidden_size)
        x = torch.randn(seq_len, model.hidden_size)
        return model, x, payload, run_model(model, x, payload)

    def _assert_all_ranks_match(self, world, seq_len=61):
        model, x, payload, reference = self._reference(seq_len)
        with PatchedStub(world) as stub:
            outputs = stub.run(lambda rank: run_model(model, x, payload))
        for video, audio in outputs:
            # Every rank ends up with the full result: the last block gathers the sequence
            # back before FinalLayer, so both ranks stay in lockstep for the next step.
            torch.testing.assert_close(video, reference[0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(audio, reference[1], rtol=1e-5, atol=1e-5)

    def test_two_ranks_reproduce_the_single_gpu_output(self):
        self._assert_all_ranks_match(world=2)

    def test_odd_sequence_length(self):
        self._assert_all_ranks_match(world=2, seq_len=63)

    def test_four_ranks(self):
        self._assert_all_ranks_match(world=4)

    def test_repeated_forwards_stay_equivalent(self):
        """A warm worker runs 20 steps per job and many jobs; state must not accumulate."""
        model, x, payload, reference = self._reference()
        with PatchedStub(2) as stub:
            outputs = stub.run(
                lambda rank: [run_model(model, x, payload) for _ in range(4)]
            )
        for per_rank in outputs:
            for video, audio in per_rank:
                torch.testing.assert_close(video, reference[0], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(audio, reference[1], rtol=1e-5, atol=1e-5)


@unittest.skipIf(torch is None, "torch is not installed")
class WorkIsActuallySplitTests(unittest.TestCase):
    """Guards against the failure mode where both GPUs do all of the work."""

    def test_each_rank_only_runs_its_own_shard_through_the_blocks(self):
        seq_len = 61
        model = build_model()
        payload = build_payload(seq_len, model.hidden_size)
        x = torch.randn(seq_len, model.hidden_size)
        expected = ulysses.shard_lengths(seq_len, 2)

        seen: dict[int, list[int]] = {}
        original_block_forward = h3_stub.DiTBlock.forward

        def watched(self, x_in, t_emb, mod_segments, rope_freqs, transformer_options={}):
            seen.setdefault(threading.get_ident(), []).append(x_in.shape[0])
            return original_block_forward(
                self, x_in, t_emb, mod_segments, rope_freqs,
                transformer_options=transformer_options,
            )

        h3_stub.DiTBlock.forward = watched
        try:
            with PatchedStub(2) as stub:
                stub.run(lambda rank: run_model(model, x, payload))
        finally:
            h3_stub.DiTBlock.forward = original_block_forward

        self.assertEqual(len(seen), 2, "both ranks should have executed blocks")
        widths = sorted(width for rows in seen.values() for width in set(rows))
        self.assertEqual(
            widths, sorted(expected),
            f"blocks ran on {widths} rows; the two ranks should own {sorted(expected)}",
        )
        for rows in seen.values():
            self.assertEqual(len(rows), 3, "every block should have run exactly once per rank")
            self.assertEqual(len(set(rows)), 1, "a rank's shard must not change size mid-model")
            self.assertLess(rows[0], seq_len, "a shard must be smaller than the whole")

    def test_the_refiner_path_is_not_sharded(self):
        """Attention called outside the block loop must be left completely alone."""
        with PatchedStub(1) as stub:
            def body(rank):
                attention = h3_stub.Attention(hidden=32, heads=4, head_dim=8)
                with torch.no_grad():
                    return attention(torch.randn(9, 32))

            (out,) = stub.run(body)
        self.assertEqual(tuple(out.shape), (9, 32))


@unittest.skipIf(torch is None, "torch is not installed")
class RopeAndSegmentPlumbingTests(unittest.TestCase):
    def test_each_rank_gets_its_own_rope_rows(self):
        # h3_stub.Attention asserts the rope table matches its sequence length, so a
        # mis-sliced table fails loudly here rather than producing subtly wrong video.
        seq_len = 61
        model = build_model(layers=2)
        payload = build_payload(seq_len, model.hidden_size)
        x = torch.randn(seq_len, model.hidden_size)
        with PatchedStub(2) as stub:
            outputs = stub.run(lambda rank: run_model(model, x, payload) is not None)
        self.assertEqual(outputs, [True, True])

    def test_state_is_cleared_when_a_forward_raises(self):
        seq_len = 20
        model = build_model(layers=2)
        payload = build_payload(seq_len, model.hidden_size)
        payload["mod_segments"] = "not a segment table"  # forces a failure inside a block

        with PatchedStub(1) as stub:
            def body(rank):
                try:
                    run_model(model, torch.randn(seq_len, model.hidden_size), payload)
                except Exception:
                    return patches._shard_active()
                raise AssertionError("the malformed segment table should have raised")

            (leftover,) = stub.run(body)
        self.assertIsNone(
            leftover, "a failed step must not leave the next one believing it is sharded"
        )


@unittest.skipIf(torch is None, "torch is not installed")
class VaeChunkDistributionTests(unittest.TestCase):
    def test_chunks_are_split_and_the_output_is_identical(self):
        reference_vae = h3_stub.MiniMaxH3VideoVAE(num_chunks=7)
        with torch.no_grad():
            reference = reference_vae.decode(torch.zeros(1))
        self.assertEqual(reference_vae.calls, list(range(7)))

        with PatchedStub(2, parallel_vae=True) as stub:
            def body(rank):
                vae = h3_stub.MiniMaxH3VideoVAE(num_chunks=7)
                with torch.no_grad():
                    decoded = vae.decode(torch.zeros(1))
                return vae.calls, decoded

            results = stub.run(body)

        for rank, (calls, decoded) in enumerate(results):
            self.assertEqual(
                calls, [index for index in range(7) if index % 2 == rank],
                f"rank {rank} decoded the wrong chunks",
            )
            torch.testing.assert_close(decoded, reference)

        # Together the ranks did exactly the original amount of work, once.
        self.assertEqual(sorted(results[0][0] + results[1][0]), list(range(7)))

    def test_single_rank_is_untouched(self):
        with PatchedStub(1, parallel_vae=True) as stub:
            def body(rank):
                vae = h3_stub.MiniMaxH3VideoVAE(num_chunks=3)
                with torch.no_grad():
                    vae.decode(torch.zeros(1))
                return vae.calls

            (calls,) = stub.run(body)
        self.assertEqual(calls, [0, 1, 2])


@unittest.skipIf(torch is None, "torch is not installed")
class InstallationReportTests(unittest.TestCase):
    def test_report_states_what_was_patched(self):
        with PatchedStub(1, parallel_vae=True) as stub:
            self.assertTrue(stub.report.dit)
            self.assertTrue(stub.report.attention)
            self.assertTrue(stub.report.vae)
            self.assertIn("dit=yes", stub.report.describe())

    def test_install_is_idempotent(self):
        with PatchedStub(1) as stub:
            first = h3_stub.MiniMaxH3Model.forward
            patches.install(
                stub.collective, parallel_vae=False, log=lambda m: None,
                model_module=h3_stub, vae_module=h3_stub,
            )
            self.assertIs(
                h3_stub.MiniMaxH3Model.forward, first,
                "a second install must not stack another wrapper",
            )


if __name__ == "__main__":
    unittest.main()
