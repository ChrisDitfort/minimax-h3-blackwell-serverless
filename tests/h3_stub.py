"""A faithful stand-in for ComfyUI's MiniMax H3 model, for testing the patches.

The real model cannot be imported on a machine without the container: it needs
comfy.quant_ops (the closed comfy-kitchen fused kernels), comfy-aimdo's dynamic VRAM and
a CUDA build of torch. What *can* be reproduced exactly is the part the patches actually
attach to - the calling conventions - and that is what this module is.

Copied deliberately, and structurally line-for-line, from
comfy/ldm/minimax/model.py at ComfyUI dec5d945 (v0.30.2):

  * Attention.forward's q/k/v layout and its optimized_attention(skip_reshape=True) call
  * DiTBlock.forward's _mod_scale_shift / _mod_gate over (start, stop, row) segments
  * MiniMaxH3Model._forward's patches_replace["dit"][("double_block", i)] dispatch loop
  * FinalLayer reading the video and audio segments by their global offsets

The maths inside the blocks is simplified (no RoPE kernel, no adaln projection, no
quantisation) because none of it is what sequence parallelism touches. The shapes, the
call order and the segment bookkeeping are not simplified, because all of it is.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The module-level name the patches replace, exactly as the real model imports it.
optimized_attention = None  # set by _default_attention() below


def _default_attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
    """Stands in for comfy.ldm.modules.attention.optimized_attention.

    Same contract: [B, heads, S, dim] in when skip_reshape, [B, S, heads*dim] out.
    """
    if not skip_reshape:
        raise AssertionError("H3 always calls optimized_attention with skip_reshape=True")
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    batch, _, seq, dim = out.shape
    return out.transpose(1, 2).reshape(batch, seq, heads * dim)


optimized_attention = _default_attention


def _mod_scale_shift(h, shift, scale, segments):
    for a, b, row in segments:
        h[a:b] = h[a:b] * (1.0 + scale[row]) + shift[row]
    return h


def _mod_gate(x, gate, other, segments):
    for a, b, row in segments:
        x[a:b] = x[a:b] + other[a:b] * gate[row]
    return x


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = q.view(s, self.heads, self.head_dim)
        k = k.view(s, self.heads, self.head_dim)
        v = v.view(s, self.heads, self.head_dim)

        if rope_freqs is not None:
            # Stands in for the fused split-half RoPE: a per-token rotation whose only
            # relevant property here is that it is indexed by absolute sequence position,
            # so a shard must be handed its own slice of the table.
            if rope_freqs.shape[1] != s:
                raise AssertionError(
                    f"rope table has {rope_freqs.shape[1]} rows but the sequence has {s}; "
                    "the shard was handed the wrong slice"
                )
            angle = rope_freqs[0, :, 0, 0, 0, 0].view(s, 1, 1)
            cos, sin = torch.cos(angle), torch.sin(angle)
            q, k = q * cos + k * sin, k * cos - q * sin

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        # Resolved through the module global on every call, exactly as the real model
        # does, which is what makes the attention patch a simple name substitution.
        out = optimized_attention(
            q, k, v, self.heads, mask=None, skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.out_proj(out.squeeze(0))


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, modalities=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn = Attention(hidden, heads, head_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.modalities = modalities

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_emb.chunk(6, dim=-1)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(
            x, gate_msa,
            self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
            mod_segments,
        )
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class FinalLayer(nn.Module):
    def __init__(self, hidden, video_dim, audio_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.video_out = nn.Linear(hidden, video_dim)
        self.audio_out = nn.Linear(hidden, audio_dim)

    def forward(self, x, video_seg, audio_seg):
        va, vb, _ = video_seg
        aa, ab, _ = audio_seg
        return self.video_out(self.norm(x[va:vb])), self.audio_out(self.norm(x[aa:ab]))


class MiniMaxH3Model(nn.Module):
    """Only the parts of the real model the patches interact with."""

    def __init__(self, hidden=32, num_attention_heads=4, attention_head_dim=8, num_layers=3,
                 video_dim=5, audio_dim=3):
        super().__init__()
        self.hidden_size = hidden
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, num_attention_heads, attention_head_dim) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(hidden, video_dim, audio_dim)

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        return self._forward(
            x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs
        )

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        payload = minimax_payload or {}
        h = x
        t_emb = payload["t_emb"]
        mod_segments = payload["mod_segments"]
        rope_freqs = payload["rope_freqs"]

        # Verbatim in structure from MiniMaxH3Model._forward.
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args, block=block):
                    return {
                        "img": block(
                            args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                            transformer_options=args["transformer_options"],
                        )
                    }

                h = blocks_replace[("double_block", i)](
                    {
                        "img": h, "t_emb": t_emb, "mod_segments": mod_segments,
                        "rope_freqs": rope_freqs, "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)

        return self.final_layer(h, payload["video_seg"], payload["audio_seg"])


# --------------------------------------------------------------------------------------
# Video VAE stand-in
# --------------------------------------------------------------------------------------


class MiniMaxH3VideoVAE(nn.Module):
    """Reproduces decode_temporal's shape: N independent chunk decodes, then stateful blending."""

    def __init__(self, num_chunks=7):
        super().__init__()
        self.num_chunks = num_chunks
        self.calls: list[int] = []

    def _adaptive_decode(self, z):
        # Records which chunks this instance actually computed, so a test can prove the
        # work was really split rather than duplicated.
        self.calls.append(int(z[0, 0].item()))
        # Frame count varies per chunk in the real VAE; vary it here too so the metadata
        # broadcast is exercised on non-uniform shapes.
        frames = 4 + (int(z[0, 0].item()) % 2)
        return z.repeat(frames, 3) * 2.0

    def decode(self, z):
        chunks = []
        carry = None
        for index in range(self.num_chunks):
            decoded = self._adaptive_decode(torch.full((1, 2), float(index)))
            if carry is not None:
                # Stateful blend across chunk boundaries, as in the real decode_temporal.
                decoded = decoded + carry.mean()
            carry = decoded
            chunks.append(decoded)
        return torch.cat([chunk.reshape(-1) for chunk in chunks])
