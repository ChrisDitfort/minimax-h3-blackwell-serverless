# Experimental two-GPU MiniMax H3 worker

This is an experiment, not a rollout. It asks one question:

> Can a single H3 generation use two GPUs well enough to be worth paying for two GPUs?

The image ships **defaulting to the single-GPU path**. Nothing in this document happens
unless the endpoint sets `H3_GPU_MODE=dual` and gives the worker two GPUs.

---

## 1. What the workload actually looks like

Measured on the current known-good image, Standard tier (1024x576, 124 frames, 20 steps),
RTX PRO 6000 Blackwell:

| phase                                    |   time | share |
| ---------------------------------------- | -----: | ----: |
| model staging + text encode + graph setup |  ~13 s |  19 % |
| 20 denoise steps @ 2.50 s/it              |  ~50 s |  72 % |
| video VAE decode + audio VAE + mux        |   ~8 s |  11 % |
| **ComfyUI prompt total**                  | 69.7 s |       |

So denoising is the only phase where a second GPU can pay for itself, and even a perfect
2x there caps end-to-end speedup at about 1.4x unless the decode is split as well. Both
are addressed below. The text encode is not, and is the largest remaining serial phase.

## 2. Why sequence parallelism, and not something simpler

H3 is a **single-stream packed-token transformer** (`comfy/ldm/minimax/model.py`). Video,
audio, text and conditioning rows are packed into one flat `[S, 5376]` sequence and all 50
blocks run full self-attention over the whole of it. That rules out the usual easy wins:

* **CFG parallelism** - the shipped workflow uses `BasicGuider`. There is no second
  unconditional pass to hand to a second GPU.
* **Pipeline parallelism** - the 50 blocks are strictly sequential. Splitting them 25/25
  leaves each GPU idle for half of every step.
* **DataParallel** - there is one sample per job. Batch splitting has nothing to split.
* **Separating the audio and video streams** - they are jointly attended in one packed
  sequence, so they cannot be run apart without changing the model's output.

What is left is the sequence itself. Inside a block, everything except attention is
token-wise: `qkv_proj`, `out_proj`, the SwiGLU MLP, both RMSNorms, the adaln
scale/shift/gate (which addresses contiguous `(start, stop)` segments), and RoPE. Every one
of those can run on half the tokens with no communication at all.

Attention is the exception, and Ulysses is the standard answer: an all-to-all switches the
shard axis from *sequence* to *heads*, each rank computes exact attention for its 28 of
the 56 heads over the **complete** K/V, and a second all-to-all switches back. Heads are
independent in attention, so this is mathematically exact - not an approximation such as
windowed or local attention.

```
per rank, per block, per step        S ~ 20k tokens, inner = 56 x 128 = 7168, bf16
  q, k, v out   3 x (S/4) x 7168 x 2 B  = ~161 MB
  out back      1 x (S/4) x 7168 x 2 B  =  ~54 MB
                                        --------
                                          ~215 MB   x 50 blocks x 20 steps = ~215 GB
```

That traffic is the main risk to scaling. On a PCIe-only Blackwell pair with peer-to-peer
enabled it should cost a few seconds per generation; without P2P it could cost two to
three times that. Measuring it is the point of the benchmark.

## 3. Execution topology

```
RunPod worker container
 |
 +-- handler.py (PID 1)                      no GPU work of its own
 |
 +-- rank 0   CUDA_VISIBLE_DEVICES=0   127.0.0.1:8188
 |              full ComfyUI. Produces the video. The only rank the handler reads.
 |
 +-- rank 1   CUDA_VISIBLE_DEVICES=1   127.0.0.1:8189
                full ComfyUI, "shadow". Holds up its half of every collective.
                Its copy of the workflow is rewritten to stop at the VAE decodes,
                so it never writes a file.
```

Each rank sees exactly one GPU as `cuda:0`, so ComfyUI's device handling and NCCL's
rank-to-device mapping both stay completely conventional. Both ranks execute the same
graph, so they reach the same collectives in the same order.

Work division inside one denoise step:

```
packed sequence  [S, 5376]
  rank 0 -> rows [0, ceil(S/2))          rank 1 -> rows [ceil(S/2), S)
    |                                      |
    | qkv / norms / MLP / adaln            | (identical, on its own rows)
    |                                      |
    +----------- all-to-all ---------------+     sequence axis -> head axis
    |                                      |
    | attention, heads 0-27, all S rows    | attention, heads 28-55, all S rows
    |                                      |
    +----------- all-to-all ---------------+     head axis -> sequence axis
    |                                      |
   ... x 50 blocks ...
    |                                      |
    +---------- all-gather (last block) ---+     FinalLayer sees the whole sequence
```

Video VAE decode is split too. `decode_temporal()` calls `_adaptive_decode()` once per
temporal chunk - seven of them for a five-second clip - and then blends the results with
carried-over overlap state. Each chunk is a pure function of its latent slice, so one rank
computes each chunk and broadcasts it, and **every rank runs the untouched blending pass
over the identical tensors**. No seam handling is changed and no overlap is approximated:
the decoded video is the same video.

## 4. Configuration

| variable | default | meaning |
| --- | --- | --- |
| `H3_GPU_MODE` | `single` | `single` = the known-good path, no patches applied at all. `dual` = sequence parallelism across two GPUs. |
| `H3_SP_WORLD_SIZE` | `2` | Ranks. Must divide 56 (the head count). |
| `H3_SP_VAE` | `1` | Also split the video VAE's temporal chunks. |
| `H3_SP_SELFTEST` | `1` | Verify sharded attention against unsharded attention at boot, on the real GPUs, before serving. |
| `H3_SP_ALLOW_FALLBACK` | `0` | When `1`, a worker asked for dual that cannot deliver it degrades to single instead of refusing to start. |
| `H3_SP_MASTER_PORT` | `29513` | Loopback rendezvous port for the NCCL group. |
| `H3_SP_INIT_TIMEOUT` | `300` | Seconds to wait for every rank to join. |
| `NCCL_DEBUG` | `WARN` | `INFO` prints a screenful per collective and buries the benchmark output. |

**Dual mode is never inferred.** A worker that happens to be scheduled two GPUs still runs
the single-GPU path unless asked. That is deliberate: an accidental mode switch would
silently invalidate an A/B comparison.

**Asking for dual and not getting it is fatal by default.** The worker refuses to start
rather than quietly serve single-GPU results from an endpoint that believes it is
measuring two. `H3_SP_ALLOW_FALLBACK=1` opts out.

## 5. Why the shadow rank does not save a video

Under Confidential Generation the finished MP4 is plaintext until the handler encrypts it,
and the handler only ever sees rank 0's. A shadow that ran `SaveVideo` would leave a
second, unencrypted, unaccounted-for copy of the customer's video on the container disk.

So `h3_parallel/shadow.py` rewrites the shadow's copy of the workflow: file-producing tail
nodes are pruned - leaves first, so no link is ever left dangling - and each VAE decode is
anchored to `H3ParallelSink`, an output node that consumes the decoded tensors and
discards them. The caller's workflow is never modified.

Two further belts on that brace:

* every shadow gets its own output and temp directory, and both are emptied after every
  job on the same boundary as rank 0's plaintext; and
* if the decode nodes cannot be identified at all, the shadow runs the caller's workflow
  verbatim and the directory sweep is what cleans up. A duplicated render the handler then
  deletes is a much better failure than a deadlocked rank 0.

## 6. What is *not* parallel, and why

| phase | status | why |
| --- | --- | --- |
| Qwen3-VL 32B text encode | serial, duplicated on both ranks | A different model with its own quantised kernels. It is the largest remaining serial phase (part of the ~13 s pre-sampling window) and the obvious next target. |
| Model staging / VRAM warm-up | serial, duplicated | Both ranks load the full 41 GB of weights. Ulysses shards activations, not parameters. |
| Audio VAE decode | serial, duplicated | ~2 s. Splitting it would cost more in complexity than it returns. |
| Video mux / H.264 encode | rank 0 only | CPU work, and the shadow is explicitly kept away from it. |
| Confidential encryption + upload | rank 0 only, in the handler | Unchanged from the known-good image. |

Amdahl, using the measured Standard-tier split and assuming the denoise scales perfectly:
13 + 25 + 5 = 43 s against 69.7 s is about **1.6x**, before any communication cost. That
is the honest ceiling for this design as built, and the reason the benchmark measures
end-to-end latency rather than GPU utilisation.

## 7. Correctness

Ulysses is exact. Each rank computes attention for its own heads over the complete K/V,
and heads never interact inside attention, so the result is the same function the
single-GPU path computes. The only numerical difference is GEMM tile scheduling - the M
dimension of each matmul is halved, which changes reduction order by roughly one ulp.
Expect numerically equivalent output, not bit-identical output.

What is verified, and where:

| claim | how |
| --- | --- |
| sharded attention == full attention | `tests/test_ulysses.py`, CPU, world sizes 1/2/4, even and odd sequence lengths, fp32 and bf16 |
| the patched model == the unpatched model, end to end | `tests/test_h3_patches.py`, against a stand-in that reproduces ComfyUI's calling conventions |
| each rank runs only its own shard | `tests/test_h3_patches.py::WorkIsActuallySplitTests` |
| VAE chunks are split once, output identical | `tests/test_h3_patches.py::VaeChunkDistributionTests` |
| mode selection never silently downgrades | `tests/test_gpu_mode.py::ModeSelectionTests` |
| the shadow graph writes nothing and has no dangling links | `tests/test_gpu_mode.py::ShadowWorkflowTests` |
| **sharded attention == full attention on the real GPUs** | the boot self-test, in-container, before the worker serves |

The boot self-test is the one that runs on the actual hardware. It builds identical q/k/v
on both ranks from a fixed seed, computes reference attention locally, runs the production
sequence-parallel path, and refuses to serve if the two disagree by more than 1e-4.

## 8. Failure modes, and what each one does

| situation | behaviour |
| --- | --- |
| `H3_GPU_MODE=dual`, one GPU | worker refuses to start, names the reason, names `H3_SP_ALLOW_FALLBACK` |
| NCCL group never forms | the rank's ComfyUI import fails; the handler's `/h3/gpu` check turns that into a fatal startup error |
| self-test mismatch | same - the rank reports `ready: false` and the worker will not serve |
| patches did not install | the handler sees `patched.dit == false` and refuses to start rather than benchmark an unpatched worker |
| a shadow is unreachable when a job arrives | the job fails immediately with a clear error, instead of rank 0 blocking on a collective |
| a shadow rejects the workflow | the job fails before rank 0 spends GPU seconds |
| a job raises mid-generation | shadows are interrupted and their scratch directories emptied before the next job |
| collective hangs | `TORCH_NCCL_BLOCKING_WAIT` plus the group's `H3_SP_INIT_TIMEOUT` turn it into an error rather than a wedged worker |

ComfyUI swallows exceptions from a custom node - `load_custom_node` logs `IMPORT FAILED`
and carries on - which is exactly why the handler independently asks every rank
`/h3/gpu` before accepting traffic. Without that check, a broken dual setup would present
as two healthy ComfyUIs quietly computing the whole sequence twice: correct video,
single-GPU speed, and a benchmark number that means nothing.

## 9. Reading the logs

At boot, per rank:

```
[h3-parallel rank=0] [H3-GPU] mode=dual gpu_count=2 strategy=ulysses-sequence-parallel rank=0 world_size=2 backend=nccl parallel_vae=true
[h3-parallel rank=0] [H3-GPU] device0 name='NVIDIA RTX PRO 6000 Blackwell Server Edition' capability=12.0 vram=97250MB
[h3-parallel rank=0] [H3-GPU] torch=2.10.0+cu130 cuda=13.0 nccl=2.28.3
[h3-parallel rank=0] [H3-PERF] distributed_init_ms=...
[h3-parallel rank=0] [H3-GPU] patched dit=yes attention=yes vae_decode=yes heads=56 blocks=50
[h3-parallel rank=0] [H3-GPU] selftest=pass attn_max_abs_err=... gather_max_abs_err=... seq=257 shards=[129, 128] took_ms=...
[h3-parallel rank=0] [H3-GPU] ready
[handler] [H3-GPU] rank 0 verified: ... [handler] [H3-GPU] rank 1 verified: ...
```

Per job:

```
[handler] [perf] proc=... gpu_mode=dual gpu_count=2 strategy=ulysses-sequence-parallel ...
          pre_sampling=... first_step=... sampling=... steps=20/20 per_step=... decode=...
          output=... encryption_ms=... total=... status=ok
[handler] [perf] nodes SamplerCustomAdvanced=... VAEDecode=... MiniMaxH3ImageToVideo=...
[handler] [perf] gpu gpu0_peak_alloc_mb=... gpu0_peak_reserved_mb=... gpu1_peak_alloc_mb=... gpu1_peak_reserved_mb=...
```

`sampling` is the number the whole experiment turns on. `decode` covers the VAE split.
`pre_sampling` is the serial phase nothing here parallelises.

## 10. Rollback

The known-good image is untouched. `:latest`, `:code` and the `staging-N` tags are never
written by the dual-GPU workflow, which publishes only to a new immutable tag and refuses
to reuse one. Rolling back is selecting the previous image in the RunPod version, or
setting `H3_GPU_MODE=single` on the experimental image - which applies no patches to
ComfyUI at all and runs the same code path as its predecessor.
