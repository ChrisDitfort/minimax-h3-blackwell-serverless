"""Wire Ulysses sequence parallelism into ComfyUI's MiniMax H3 implementation.

Design constraint: touch as little of ComfyUI as possible. Copying model code into this
file would mean a base-image bump could silently change the maths under us, so nothing
here reimplements a forward pass. Three seams are used instead, and all three are things
ComfyUI already does on its own:

1.  `transformer_options["patches_replace"]["dit"][("double_block", i)]`
    A documented per-block replacement hook that MiniMaxH3Model._forward already honours.
    Block 0's hook shards the packed sequence; block 49's gathers it back. Every block in
    between just runs on a shorter sequence, unmodified.

2.  `comfy.ldm.minimax.model.optimized_attention`
    The module-level name H3's Attention.forward calls. Replacing it lets the all-to-all
    happen around attention without touching the fused RMSNorm+RoPE kernel that produces
    q/k/v. The real backend (SageAttention3 on Blackwell) is still what computes attention;
    it is simply handed 28 heads over the whole sequence instead of 56 over half of it.

3.  `MiniMaxH3VideoVAE._adaptive_decode`
    decode_temporal() calls it once per temporal chunk in a loop. The patched version has
    one rank do the work and broadcast the result, so the surrounding blending code - which
    is delicate and stateful - runs completely unmodified on every rank and produces the
    identical video.

Nothing in this module activates unless dual mode is on. In single mode `install` is never
called and ComfyUI is byte-for-byte the image it has always been.
"""

from __future__ import annotations

import threading

import torch

from . import ulysses

#: Per-thread parallel state. ComfyUI executes a prompt on one worker thread, and the DiT
#: and the VAE both run on it, so thread-local is the right scope: it cannot leak into the
#: HTTP threads and it is reset around every model forward.
_STATE = threading.local()

#: torch dtypes are not directly broadcastable, so the VAE metadata hop sends an index.
_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int64,
]
_MAX_DIMS = 6


class ShardState:
    """What a rank needs to run the DiT blocks on its slice of the packed sequence."""

    __slots__ = ("seq_lengths", "start", "stop", "rope_freqs", "mod_segments")

    def __init__(self, seq_lengths, start, stop, rope_freqs, mod_segments):
        self.seq_lengths = seq_lengths
        self.start = start
        self.stop = stop
        self.rope_freqs = rope_freqs
        self.mod_segments = mod_segments


def _shard_active() -> ShardState | None:
    return getattr(_STATE, "shard", None)


def _reset_state() -> None:
    _STATE.shard = None
    _STATE.in_block = False


# --------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------


def make_parallel_attention(inner_attention, collective):
    """Wrap ComfyUI's optimized_attention with the Ulysses head/sequence axis switch.

    Signature and return shape match the original exactly, so H3's Attention.forward -
    which does `out.squeeze(0)` then out_proj - needs no changes at all.

    Falls through to the original whenever sequence parallelism is not in force: the
    TokenRefiner runs on the text embedding before the sequence is ever sharded, and a
    masked attention would not be safe to split this way.
    """

    def parallel_attention(q, k, v, heads, mask=None, **kwargs):
        state = _shard_active()
        if state is None or not getattr(_STATE, "in_block", False) or mask is not None:
            return inner_attention(q, k, v, heads, mask=mask, **kwargs)

        if not kwargs.get("skip_reshape"):
            # H3 always passes pre-reshaped [B, H, S, D]. Anything else is not ours.
            return inner_attention(q, k, v, heads, mask=mask, **kwargs)

        inner_kwargs = dict(kwargs)
        inner_kwargs["skip_reshape"] = True
        inner_kwargs.pop("skip_output_reshape", None)

        def attention(qf, kf, vf, heads_local):
            return inner_attention(qf, kf, vf, heads_local, mask=None, **inner_kwargs)

        # [1, heads, s_local, dim] -> [s_local, heads, dim]
        local = ulysses.sequence_parallel_attention(
            q.squeeze(0).transpose(0, 1),
            k.squeeze(0).transpose(0, 1),
            v.squeeze(0).transpose(0, 1),
            collective=collective,
            seq_lengths=state.seq_lengths,
            attention=attention,
        )
        rows = local.shape[0]
        return local.reshape(1, rows, -1)

    return parallel_attention


# --------------------------------------------------------------------------------------
# DiT block loop
# --------------------------------------------------------------------------------------


def _enter_shard(args, collective) -> tuple[ShardState, torch.Tensor]:
    """Split the packed sequence and rebuild the per-token inputs for this rank's slice."""
    packed = args["img"]
    total = packed.shape[0]
    lengths = ulysses.shard_lengths(total, collective.world_size)
    start = sum(lengths[: collective.rank])
    stop = start + lengths[collective.rank]

    rope = args.get("rope_freqs")
    # [1, S, 1, half, 2, 2] -> this rank's rows. Cloned, not sliced: the fused split-half
    # RoPE kernel writes through its inputs, and a view into the full table would be a
    # cross-shard alias.
    rope_local = rope[:, start:stop].clone() if rope is not None else None

    state = ShardState(
        seq_lengths=lengths,
        start=start,
        stop=stop,
        rope_freqs=rope_local,
        mod_segments=ulysses.remap_segments(args["mod_segments"], start, stop),
    )
    return state, packed[start:stop].clone()


def make_block_patch(index, last_index, previous_patch, collective):
    """A `("double_block", i)` replacement that runs block `i` on this rank's shard."""

    def block_patch(args, extra):
        local_args = dict(args)

        if index == 0:
            state, local_packed = _enter_shard(args, collective)
            _STATE.shard = state
            local_args["img"] = local_packed

        state = _shard_active()
        if state is None:
            # Sharding never happened (block 0 was skipped); run the block untouched
            # rather than guess.
            return previous_patch(args, extra) if previous_patch else extra["original_block"](args)

        local_args["rope_freqs"] = state.rope_freqs
        local_args["mod_segments"] = state.mod_segments

        _STATE.in_block = True
        try:
            if previous_patch is not None:
                result = previous_patch(local_args, extra)
            else:
                result = extra["original_block"](local_args)
        finally:
            _STATE.in_block = False

        packed = result["img"]
        if index == last_index:
            # FinalLayer addresses the video and audio segments by their global offsets,
            # so hand it the reassembled sequence and leave it completely unpatched.
            _STATE.shard = None
            return {"img": collective.all_gather_rows(packed, state.seq_lengths)}
        return {"img": packed}

    return block_patch


def make_parallel_model_forward(original_forward, collective):
    """Install the per-block hooks for one forward, and guarantee state is reset after."""

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        options = dict(transformer_options or {})
        patches_replace = dict(options.get("patches_replace") or {})
        dit = dict(patches_replace.get("dit") or {})

        last = len(self.blocks) - 1
        for index in range(len(self.blocks)):
            key = ("double_block", index)
            dit[key] = make_block_patch(index, last, dit.get(key), collective)

        patches_replace["dit"] = dit
        options["patches_replace"] = patches_replace

        _reset_state()
        try:
            return original_forward(
                self, x, timestep, context, options, minimax_payload=minimax_payload, **kwargs
            )
        finally:
            # A failed step must not leave the next one thinking it is sharded.
            _reset_state()

    return forward


# --------------------------------------------------------------------------------------
# Video VAE temporal chunks
# --------------------------------------------------------------------------------------


def make_parallel_adaptive_decode(original_adaptive_decode, collective):
    """Distribute decode_temporal's per-chunk work without touching decode_temporal.

    decode_temporal() calls _adaptive_decode once per temporal chunk (7 of them for a
    5-second clip) and then blends the results together with carried-over overlap state.
    Only the decode itself is expensive, and each chunk is a pure function of its latent
    slice - so one rank computes each chunk and broadcasts it, and every rank runs the
    identical blending pass on the identical tensors. The decoded video is the same video,
    not an approximation of it.
    """

    def adaptive_decode(self, z):
        if not getattr(_STATE, "vae_decode", False) or collective.world_size == 1:
            return original_adaptive_decode(self, z)

        index = getattr(_STATE, "vae_chunk", 0)
        _STATE.vae_chunk = index + 1
        owner = index % collective.world_size

        meta = torch.zeros(2 + _MAX_DIMS, dtype=torch.int64, device=z.device)
        decoded = None
        if collective.rank == owner:
            decoded = original_adaptive_decode(self, z)
            if decoded.dim() > _MAX_DIMS:
                raise RuntimeError(
                    f"VAE chunk has {decoded.dim()} dimensions; the parallel decode path "
                    f"supports at most {_MAX_DIMS}."
                )
            meta[0] = decoded.dim()
            meta[1] = _DTYPES.index(decoded.dtype)
            for axis, size in enumerate(decoded.shape):
                meta[2 + axis] = size

        collective.broadcast(meta, owner)

        if collective.rank == owner:
            payload = decoded.contiguous()
        else:
            dims = int(meta[0].item())
            shape = tuple(int(meta[2 + axis].item()) for axis in range(dims))
            payload = torch.empty(shape, dtype=_DTYPES[int(meta[1].item())], device=z.device)

        return collective.broadcast(payload, owner)

    return adaptive_decode


def make_parallel_vae_decode(original_decode):
    """Bracket a decode so chunk numbering restarts and can never drift between ranks."""

    def decode(self, z):
        _STATE.vae_decode = True
        _STATE.vae_chunk = 0
        try:
            return original_decode(self, z)
        finally:
            _STATE.vae_decode = False
            _STATE.vae_chunk = 0

    return decode


# --------------------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------------------


class Installation:
    """What was patched, so the boot log can state it and the tests can assert it."""

    def __init__(self) -> None:
        self.dit = False
        self.attention = False
        self.vae = False
        self.num_heads: int | None = None
        self.num_blocks: int | None = None
        self.notes: list[str] = []

    def describe(self) -> str:
        return (
            f"[H3-GPU] patched dit={'yes' if self.dit else 'no'} "
            f"attention={'yes' if self.attention else 'no'} "
            f"vae_decode={'yes' if self.vae else 'no'} "
            f"heads={self.num_heads} blocks={self.num_blocks}"
        )


def install(
    collective,
    *,
    parallel_vae: bool,
    log,
    model_module=None,
    vae_module=None,
) -> Installation:
    """Apply every patch dual mode needs. Raises if the DiT cannot be patched.

    `model_module` / `vae_module` exist so the test suite can install the real patches
    onto a stand-in that reproduces ComfyUI's calling conventions; in the container both
    are None and the genuine comfy modules are used.
    """
    if model_module is None:
        import comfy.ldm.minimax.model as model_module

    h3_model = model_module
    report = Installation()

    if getattr(h3_model, "_h3_sequence_parallel_installed", False):
        log("sequence parallelism already installed; skipping")
        return report

    # No model is instantiated at import time, so the constructor defaults are the only
    # geometry available - and they are the geometry the FL2VA checkpoint uses.
    report.num_heads = _default_arg(h3_model.MiniMaxH3Model.__init__, "num_attention_heads")
    report.num_blocks = _default_arg(h3_model.MiniMaxH3Model.__init__, "num_layers")

    if report.num_heads is not None and report.num_heads % collective.world_size:
        raise RuntimeError(
            f"MiniMax H3 has {report.num_heads} attention heads, which does not divide "
            f"across {collective.world_size} ranks; Ulysses cannot shard them."
        )

    h3_model.optimized_attention = make_parallel_attention(
        h3_model.optimized_attention, collective
    )
    report.attention = True

    h3_model.MiniMaxH3Model.forward = make_parallel_model_forward(
        h3_model.MiniMaxH3Model.forward, collective
    )
    report.dit = True

    if parallel_vae:
        try:
            h3_vae = vae_module
            if h3_vae is None:
                import comfy.ldm.minimax.vae as h3_vae

            h3_vae.MiniMaxH3VideoVAE._adaptive_decode = make_parallel_adaptive_decode(
                h3_vae.MiniMaxH3VideoVAE._adaptive_decode, collective
            )
            h3_vae.MiniMaxH3VideoVAE.decode = make_parallel_vae_decode(
                h3_vae.MiniMaxH3VideoVAE.decode
            )
            report.vae = True
        except Exception as error:
            # The DiT is 70%+ of the job; losing the VAE split is a slowdown, not a
            # correctness problem, so it must never take the worker down with it.
            report.notes.append(f"video VAE not parallelised: {error}")
            log(f"WARNING: video VAE decode was not parallelised ({error}); it stays serial.")

    h3_model._h3_sequence_parallel_installed = True
    return report


def _default_arg(function, name):
    """The declared default of a keyword argument, or None if it has none."""
    import inspect

    try:
        parameter = inspect.signature(function).parameters[name]
    except (KeyError, ValueError, TypeError):
        return None
    return None if parameter.default is inspect.Parameter.empty else parameter.default
