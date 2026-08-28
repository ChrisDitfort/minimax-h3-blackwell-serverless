"""Numerical conformance tests for the Ulysses sequence-parallel algebra.

Run with:  python -m unittest discover -s tests -v

These are the tests that decide whether the dual-GPU path is *correct*, as opposed to
merely fast, and they need no GPU: the collectives are swapped for an in-process
thread-rendezvous implementation, so the exact production code in h3_parallel/ulysses.py
runs with world_size 2 and 4 on CPU tensors.

The property under test is equivalence, not similarity. Sharding a transformer's sequence
across ranks and switching the shard axis inside attention is supposed to compute exactly
the same function as the single-GPU path; every assertion here is against a reference
implementation that never shards anything.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:  # pragma: no cover - torch is always present in the image
    torch = None

if torch is not None:
    from h3_parallel import ulysses
    from h3_parallel.collectives import ThreadCollective, ThreadGroup


def reference_attention(q, k, v, heads):
    """The contract ComfyUI's optimized_attention(skip_reshape=True) implements.

    q/k/v: [1, heads, S, dim] -> [1, S, heads*dim]
    """
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    batch, _, seq, dim = out.shape
    return out.transpose(1, 2).reshape(batch, seq, heads * dim)


def run_ranks(world_size, body):
    """Run `body(rank, collective)` on `world_size` threads and collect the results."""
    group = ThreadGroup(world_size)
    results: list = [None] * world_size
    errors: list = [None] * world_size

    def target(rank):
        try:
            results[rank] = body(rank, ThreadCollective(group, rank))
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            errors[rank] = error
            group.barrier.abort()

    threads = [threading.Thread(target=target, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    for error in errors:
        if error is not None:
            raise error
    return results


@unittest.skipIf(torch is None, "torch is not installed")
class ShardPlanTests(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(ulysses.shard_boundaries(8, 2), [(0, 4), (4, 8)])

    def test_uneven_split_gives_the_remainder_to_the_low_ranks(self):
        # The H3 packed sequence is text + cond + audio + video and is routinely odd, so
        # this is the normal case, not an edge case. No padding row is ever introduced.
        self.assertEqual(ulysses.shard_boundaries(7, 2), [(0, 4), (4, 7)])
        self.assertEqual(ulysses.shard_lengths(7, 2), [4, 3])
        self.assertEqual(ulysses.shard_lengths(10, 4), [3, 3, 2, 2])

    def test_split_covers_the_sequence_exactly(self):
        for total in (0, 1, 2, 3, 17, 1024, 18_913):
            for world in (1, 2, 4, 8):
                bounds = ulysses.shard_boundaries(total, world)
                self.assertEqual(bounds[0][0], 0)
                self.assertEqual(bounds[-1][1], total)
                for (_, stop), (start, _) in zip(bounds, bounds[1:]):
                    self.assertEqual(stop, start, "shards must be contiguous")

    def test_single_rank_is_the_whole_sequence(self):
        self.assertEqual(ulysses.shard_boundaries(5, 1), [(0, 5)])

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            ulysses.shard_boundaries(4, 0)
        with self.assertRaises(ValueError):
            ulysses.shard_boundaries(-1, 2)


@unittest.skipIf(torch is None, "torch is not installed")
class SegmentRemapTests(unittest.TestCase):
    """adaln modulation addresses the sequence by (start, stop); shards must renumber it."""

    def test_segments_are_clipped_and_shifted(self):
        segments = [(0, 10, 0), (10, 25, 1), (25, 40, 2)]
        self.assertEqual(
            ulysses.remap_segments(segments, 0, 20),
            [(0, 10, 0), (10, 20, 1)],
        )
        self.assertEqual(
            ulysses.remap_segments(segments, 20, 40),
            [(0, 5, 1), (5, 20, 2)],
        )

    def test_segments_entirely_outside_the_window_are_dropped(self):
        self.assertEqual(ulysses.remap_segments([(0, 4, 7)], 10, 20), [])

    def test_union_of_shards_reproduces_the_original(self):
        segments = [(0, 137, 0), (137, 900, 1), (900, 1801, 2)]
        rebuilt = []
        for start, stop in ulysses.shard_boundaries(1801, 3):
            rebuilt += [(a + start, b + start, row) for a, b, row in
                        ulysses.remap_segments(segments, start, stop)]
        # Merge the pieces that a shard boundary split back together.
        merged = [rebuilt[0]]
        for a, b, row in rebuilt[1:]:
            pa, pb, prow = merged[-1]
            if row == prow and a == pb:
                merged[-1] = (pa, b, row)
            else:
                merged.append((a, b, row))
        self.assertEqual(merged, segments)


@unittest.skipIf(torch is None, "torch is not installed")
class AllToAllTests(unittest.TestCase):
    def test_scatter_then_gather_is_the_identity(self):
        torch.manual_seed(0)
        world, heads, dim, total = 2, 8, 4, 7
        lengths = ulysses.shard_lengths(total, world)
        full = torch.randn(total, heads, dim)

        def body(rank, collective):
            start = sum(lengths[:rank])
            local = full[start : start + lengths[rank]].clone()
            switched = ulysses.scatter_heads_gather_sequence(local, collective, lengths)
            # Head-parallel view: every token, this rank's heads.
            heads_local = heads // world
            expected = full[:, rank * heads_local : (rank + 1) * heads_local]
            torch.testing.assert_close(switched, expected)
            return ulysses.gather_heads_scatter_sequence(switched, collective, lengths)

        for rank, restored in enumerate(run_ranks(world, body)):
            start = sum(lengths[:rank])
            torch.testing.assert_close(restored, full[start : start + lengths[rank]])

    def test_four_ranks(self):
        torch.manual_seed(1)
        world, heads, dim, total = 4, 12, 6, 13
        lengths = ulysses.shard_lengths(total, world)
        full = torch.randn(total, heads, dim)

        def body(rank, collective):
            start = sum(lengths[:rank])
            local = full[start : start + lengths[rank]].clone()
            switched = ulysses.scatter_heads_gather_sequence(local, collective, lengths)
            return ulysses.gather_heads_scatter_sequence(switched, collective, lengths)

        for rank, restored in enumerate(run_ranks(world, body)):
            start = sum(lengths[:rank])
            torch.testing.assert_close(restored, full[start : start + lengths[rank]])


@unittest.skipIf(torch is None, "torch is not installed")
class SequenceParallelAttentionTests(unittest.TestCase):
    """The claim the whole dual-GPU path rests on: sharded attention == full attention."""

    def _check(self, world, heads, dim, total, dtype=None):
        # Resolved here, not in the signature: the class body is evaluated even when the
        # whole class is about to be skipped for want of torch.
        dtype = dtype or torch.float32
        torch.manual_seed(7)
        q = torch.randn(total, heads, dim, dtype=dtype)
        k = torch.randn(total, heads, dim, dtype=dtype)
        v = torch.randn(total, heads, dim, dtype=dtype)

        expected = reference_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            heads,
        ).view(total, heads, dim)

        lengths = ulysses.shard_lengths(total, world)

        def body(rank, collective):
            start = sum(lengths[:rank])
            stop = start + lengths[rank]
            return ulysses.sequence_parallel_attention(
                q[start:stop].clone(),
                k[start:stop].clone(),
                v[start:stop].clone(),
                collective=collective,
                seq_lengths=lengths,
                attention=reference_attention,
            )

        shards = run_ranks(world, body)
        combined = torch.cat(shards, dim=0)
        self.assertEqual(combined.shape, expected.shape)
        torch.testing.assert_close(combined, expected, rtol=1e-5, atol=1e-5)

    def test_two_ranks_even_sequence(self):
        self._check(world=2, heads=8, dim=16, total=64)

    def test_two_ranks_odd_sequence(self):
        # The real packed layout is text + cond + audio*2 + video and is not even.
        self._check(world=2, heads=8, dim=16, total=63)

    def test_h3_head_geometry(self):
        # MiniMax H3: 56 heads x 128 dim. Small sequence, real head count.
        self._check(world=2, heads=56, dim=128, total=97)

    def test_four_ranks(self):
        self._check(world=4, heads=8, dim=16, total=70)

    def test_world_size_one_is_a_passthrough(self):
        self._check(world=1, heads=8, dim=16, total=33)

    def test_bfloat16_matches_within_tolerance(self):
        # The model runs bf16; the point is that sharding adds no error of its own beyond
        # the reduction-order noise a different GEMM tiling already produces.
        torch.manual_seed(11)
        world, heads, dim, total = 2, 8, 16, 48
        q = torch.randn(total, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(total, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(total, heads, dim, dtype=torch.bfloat16)
        expected = reference_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            heads,
        ).view(total, heads, dim)

        lengths = ulysses.shard_lengths(total, world)

        def body(rank, collective):
            start = sum(lengths[:rank])
            return ulysses.sequence_parallel_attention(
                q[start : start + lengths[rank]].clone(),
                k[start : start + lengths[rank]].clone(),
                v[start : start + lengths[rank]].clone(),
                collective=collective,
                seq_lengths=lengths,
                attention=reference_attention,
            )

        combined = torch.cat(run_ranks(world, body), dim=0)
        torch.testing.assert_close(combined, expected, rtol=1e-2, atol=1e-2)

    def test_heads_must_divide_across_ranks(self):
        lengths = [2, 2]
        group = ThreadGroup(2)
        collective = ThreadCollective(group, 0)
        with self.assertRaises(ValueError):
            ulysses.scatter_heads_gather_sequence(torch.randn(2, 7, 4), collective, lengths)


@unittest.skipIf(torch is None, "torch is not installed")
class VariableGatherTests(unittest.TestCase):
    def test_all_gather_rows_reassembles_uneven_shards(self):
        torch.manual_seed(3)
        total, world = 9, 2
        lengths = ulysses.shard_lengths(total, world)
        full = torch.randn(total, 5)

        def body(rank, collective):
            start = sum(lengths[:rank])
            local = full[start : start + lengths[rank]].clone()
            return collective.all_gather_rows(local, lengths)

        for gathered in run_ranks(world, body):
            torch.testing.assert_close(gathered, full)

    def test_all_gather_varied_handles_different_shapes(self):
        # The VAE's temporal chunks do not all decode to the same frame count.
        shapes = [(2, 3), (4, 3)]

        def body(rank, collective):
            local = torch.full(shapes[rank], float(rank))
            return collective.all_gather_varied(local, shapes)

        for gathered in run_ranks(2, body):
            self.assertEqual([tuple(t.shape) for t in gathered], shapes)
            self.assertTrue(torch.all(gathered[0] == 0.0))
            self.assertTrue(torch.all(gathered[1] == 1.0))


@unittest.skipIf(torch is None, "torch is not installed")
class ChunkAssignmentTests(unittest.TestCase):
    def test_round_robin(self):
        self.assertEqual(ulysses.assign_chunks(7, 2), [0, 1, 0, 1, 0, 1, 0])

    def test_every_chunk_is_owned_exactly_once(self):
        owners = ulysses.assign_chunks(7, 2)
        self.assertEqual(len(owners), 7)
        self.assertEqual(sorted(set(owners)), [0, 1])

    def test_single_rank_owns_everything(self):
        self.assertEqual(ulysses.assign_chunks(3, 1), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
