"""Ulysses sequence parallelism for the MiniMax H3 packed-token DiT.

Why this and not something else
-------------------------------
H3 is a single-stream transformer: the video, audio, text and conditioning rows are packed
into one flat sequence [S, hidden] and every one of the 50 blocks runs full self-attention
over all of it (comfy/ldm/minimax/model.py). The workflow the Cloudflare Worker submits
uses BasicGuider, so there is no classifier-free-guidance second pass to hand to a second
GPU, and the 50 blocks are strictly sequential so pipelining them would leave one GPU idle
half the time.

What is left is the sequence itself. Every operation in a block except attention is
token-wise:

    qkv_proj, out_proj, fc1, fc2   linear over the last dim  -> token-local
    RMSNorm                        over the last dim         -> token-local
    adaln modulate / gate          per contiguous segment    -> token-local
    RoPE                           per token position        -> token-local
    attention                      over the whole sequence   -> NOT token-local

So each rank can own a contiguous half of the sequence and do half the work of everything
but attention, with zero communication. Attention is made to work by switching the shard
axis: an all-to-all turns "my half of the tokens, all 56 heads" into "all the tokens, my 28
heads". Heads are independent in attention, so each rank computes exact attention for its
heads over the complete K/V, and a second all-to-all switches back. This is DeepSpeed
Ulysses, and it is mathematically exact - not an approximation like local/windowed
attention. The only numerical difference from the single-GPU path is GEMM tile scheduling
(the M dimension is halved), which changes reduction order inside a matmul by ~1 ulp.

Communication cost per attention, per rank, for world=2:
    3 x (S/4) x inner x 2 bytes   forward all-to-all (q, k, v)
    1 x (S/4) x inner x 2 bytes   reverse all-to-all (out)
With S ~ 20k and inner = 56*128 = 7168 that is ~287 MB per block per step. It is the
dominant risk to scaling on a PCIe-only pair, which is exactly what the RunPod benchmark
is there to measure.

Everything in this module is a pure function of its inputs plus an injected `Collective`,
so the whole algebra is verified on CPU in tests/test_ulysses.py without a GPU.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .collectives import Collective


def shard_boundaries(total: int, world_size: int) -> list[tuple[int, int]]:
    """Split `total` rows into `world_size` contiguous, near-equal spans.

    Contiguous rather than strided because the DiT's adaln modulation addresses the
    sequence as (start, stop) segment ranges, and because concatenating contiguous shards
    in rank order reconstructs the original sequence with no permutation anywhere.

    The first `total % world_size` ranks take one extra row, so no padding token is ever
    introduced. That matters: a padded row would be a real key in an unmasked attention and
    would silently perturb every output.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")

    base, remainder = divmod(total, world_size)
    bounds = []
    cursor = 0
    for rank in range(world_size):
        size = base + (1 if rank < remainder else 0)
        bounds.append((cursor, cursor + size))
        cursor += size
    return bounds


def shard_lengths(total: int, world_size: int) -> list[int]:
    return [stop - start for start, stop in shard_boundaries(total, world_size)]


def remap_segments(
    segments: Sequence[tuple[int, int, int]], start: int, stop: int
) -> list[tuple[int, int, int]]:
    """Clip global (start, stop, row) adaln segments onto a local shard window.

    The DiT hands each block a segment table covering the whole packed sequence. A rank
    that owns rows [start, stop) needs the same table expressed in its own coordinates,
    with segments that fall entirely outside dropped and the straddling one truncated.
    """
    local: list[tuple[int, int, int]] = []
    for seg_start, seg_stop, row in segments:
        lo = max(seg_start, start)
        hi = min(seg_stop, stop)
        if hi > lo:
            local.append((lo - start, hi - start, row))
    return local


def scatter_heads_gather_sequence(
    local: torch.Tensor,
    collective: Collective,
    seq_lengths: list[int],
) -> torch.Tensor:
    """[s_local, heads, dim] -> [S, heads/world, dim].

    Before: this rank holds its slice of the tokens and every head.
    After:  this rank holds every token and its slice of the heads.

    Rank ordering is preserved on both axes, so the returned sequence axis is in the
    original global token order and the head slice is heads[rank*H_local : (rank+1)*H_local].
    """
    world = collective.world_size
    if world == 1:
        return local

    s_local, heads, dim = local.shape
    if heads % world:
        raise ValueError(f"{heads} heads do not divide across {world} ranks")
    heads_local = heads // world

    # Group by destination rank first so one flat buffer can be split by peer.
    send = local.reshape(s_local, world, heads_local, dim).permute(1, 0, 2, 3).contiguous()

    chunk = s_local * heads_local * dim
    received = collective.all_to_all(
        send.reshape(-1),
        input_splits=[chunk] * world,
        output_splits=[length * heads_local * dim for length in seq_lengths],
    )

    # Received is source-rank-major; each source contributed its own token count.
    pieces = []
    offset = 0
    for length in seq_lengths:
        span = length * heads_local * dim
        pieces.append(received[offset : offset + span].view(length, heads_local, dim))
        offset += span
    return torch.cat(pieces, dim=0)


def gather_heads_scatter_sequence(
    full: torch.Tensor,
    collective: Collective,
    seq_lengths: list[int],
) -> torch.Tensor:
    """[S, heads/world, dim] -> [s_local, heads, dim]. The exact inverse of the above."""
    world = collective.world_size
    if world == 1:
        return full

    total, heads_local, dim = full.shape
    if total != sum(seq_lengths):
        raise ValueError(f"sequence is {total} rows but the shard plan sums to {sum(seq_lengths)}")

    send = full.contiguous().reshape(-1)
    received = collective.all_to_all(
        send,
        input_splits=[length * heads_local * dim for length in seq_lengths],
        output_splits=[seq_lengths[collective.rank] * heads_local * dim] * world,
    )

    s_local = seq_lengths[collective.rank]
    # Source-rank-major again: [world, s_local, heads_local, dim] -> heads concatenate.
    return received.view(world, s_local, heads_local, dim).permute(1, 0, 2, 3).reshape(
        s_local, world * heads_local, dim
    )


def sequence_parallel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    collective: Collective,
    seq_lengths: list[int],
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Exact full-sequence attention with the heads split across ranks.

    q/k/v arrive as [s_local, heads, dim] - this rank's tokens, all heads - and the result
    comes back the same way. `attention` is handed [1, heads_local, S, dim] tensors and the
    local head count, and must return [1, S, heads_local * dim]; that is exactly the
    contract of ComfyUI's optimized_attention with skip_reshape=True, so the worker's real
    attention backend (SageAttention3 on Blackwell) is used unchanged.
    """
    world = collective.world_size
    heads = q.shape[1]

    if world == 1:
        heads_local = heads
        qf, kf, vf = q, k, v
    else:
        heads_local = heads // world
        qf = scatter_heads_gather_sequence(q, collective, seq_lengths)
        kf = scatter_heads_gather_sequence(k, collective, seq_lengths)
        vf = scatter_heads_gather_sequence(v, collective, seq_lengths)

    total = qf.shape[0]
    out = attention(
        qf.transpose(0, 1).unsqueeze(0),
        kf.transpose(0, 1).unsqueeze(0),
        vf.transpose(0, 1).unsqueeze(0),
        heads_local,
    )
    # reshape, not view: an attention backend is free to return a non-contiguous
    # result, and SageAttention3's output layout is not ours to assume.
    out = out.reshape(total, heads_local, -1)

    if world == 1:
        return out
    return gather_heads_scatter_sequence(out, collective, seq_lengths)


# --------------------------------------------------------------------------------------
# Chunk assignment for the video VAE
# --------------------------------------------------------------------------------------


def assign_chunks(num_chunks: int, world_size: int) -> list[int]:
    """Which rank decodes which temporal VAE chunk.

    Round-robin rather than contiguous blocks: with 7 chunks and 2 ranks that is 4/3
    rather than 4/3 either way, but round-robin keeps the split even for any count and
    makes the assignment trivial to state in a log line.
    """
    return [index % world_size for index in range(num_chunks)]
