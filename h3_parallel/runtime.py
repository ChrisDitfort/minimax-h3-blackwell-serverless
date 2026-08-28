"""Boot-time wiring for the multi-GPU H3 path, run inside each ComfyUI process.

Order matters here. By the time ComfyUI imports custom nodes it has already chosen its
attention backend and its VRAM strategy, but no model has been loaded and no prompt has
run - which is exactly the window in which the process group must be formed and the
patches installed. Doing it later would put NCCL's bootstrap inside a billed job; doing it
earlier would race ComfyUI's own CUDA initialisation.

Every process prints the same four things before serving anything:

    [H3-GPU] mode=... gpu_count=... strategy=...
    [H3-GPU] rank=... device=... vram=... cuda=... torch=... nccl=...
    [H3-GPU] patched dit=yes attention=yes vae_decode=yes heads=56 blocks=50
    [H3-GPU] selftest=pass ...

If the self-test does not pass, the process refuses to serve. A worker that quietly
produces wrong video is worse than one that never starts.
"""

from __future__ import annotations

import os
import time

from . import config, patches

#: What this rank actually managed to do, served over /h3/gpu. The handler treats
#: ready=False as a fatal startup condition, which is the only way a swallowed
#: custom-node import error can still stop a bogus benchmark from being run.
STATUS: dict = {
    "ready": False,
    "mode": None,
    "rank": None,
    "world_size": None,
    "strategy": None,
    "patched": {},
    "selftest": None,
    "error": None,
}


def record_boot_failure(error: BaseException) -> None:
    STATUS["ready"] = False
    STATUS["error"] = f"{type(error).__name__}: {error}"
    log(f"ERROR: multi-GPU boot failed: {STATUS['error']}")


def log(message: str) -> None:
    rank = os.environ.get("H3_SP_RANK")
    prefix = f"[h3-parallel rank={rank}]" if rank is not None else "[h3-parallel]"
    print(f"{prefix} {message}", flush=True)


def describe_devices() -> list[str]:
    """One line per visible CUDA device. Never raises."""
    lines: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return ["cuda unavailable"]
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            total = torch.cuda.get_device_properties(index).total_memory
            lines.append(
                f"device{index} name={name!r} capability={capability[0]}.{capability[1]} "
                f"vram={total // (1024 * 1024)}MB"
            )
    except Exception as error:  # pragma: no cover - diagnostics must not break boot
        lines.append(f"device probe failed: {error}")
    return lines


def stack_versions() -> str:
    try:
        import torch

        nccl = "n/a"
        try:
            if torch.cuda.is_available() and hasattr(torch.cuda, "nccl"):
                nccl = ".".join(str(part) for part in torch.cuda.nccl.version())
        except Exception:
            nccl = "unknown"
        return f"torch={torch.__version__} cuda={torch.version.cuda} nccl={nccl}"
    except Exception as error:  # pragma: no cover
        return f"version probe failed: {error}"


def gpu_report() -> dict:
    """Per-device memory, for the /h3/gpu route the handler polls after every job."""
    report: dict = {
        "rank": _int_env("H3_SP_RANK", -1),
        "world_size": _int_env("H3_SP_WORLD_SIZE", 1),
        "mode": os.environ.get("H3_GPU_MODE", config.SINGLE),
        "devices": [],
        **{key: value for key, value in STATUS.items() if key not in ("rank", "world_size", "mode")},
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return report
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(index) // (1024 * 1024),
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(index) // (1024 * 1024),
                    "total_mb": total // (1024 * 1024),
                    "free_mb": free // (1024 * 1024),
                }
            )
    except Exception as error:  # pragma: no cover
        report["error"] = str(error)
    return report


def reset_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
    except Exception:  # pragma: no cover
        pass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------------------
# Correctness self-test
# --------------------------------------------------------------------------------------


def selftest(collective) -> str:
    """Prove on the real GPUs that sharded attention equals unsharded attention.

    Both ranks build the same q/k/v from the same seed, each computes the reference
    attention for the whole sequence locally, then each runs the sequence-parallel path on
    its own shard and compares. This catches a wrong all-to-all layout, a mismatched head
    split and a broken NCCL topology in about a tenth of a second, before a single frame of
    anyone's video depends on it.
    """
    import torch

    from . import ulysses

    began = time.monotonic()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    # Odd length on purpose: the real packed sequence does not divide evenly by two.
    total, heads, dim = 257, 8, 64
    generator = torch.Generator(device="cpu").manual_seed(20260828)

    def make():
        return torch.randn(total, heads, dim, generator=generator, dtype=torch.float32).to(device)

    q, k, v = make(), make(), make()

    def attention(qf, kf, vf, heads_local):
        out = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf)
        batch, _, seq, head_dim = out.shape
        return out.transpose(1, 2).reshape(batch, seq, heads_local * head_dim)

    expected = attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        heads,
    ).view(total, heads, dim)

    lengths = ulysses.shard_lengths(total, collective.world_size)
    start = sum(lengths[: collective.rank])
    stop = start + lengths[collective.rank]

    actual = ulysses.sequence_parallel_attention(
        q[start:stop].contiguous(),
        k[start:stop].contiguous(),
        v[start:stop].contiguous(),
        collective=collective,
        seq_lengths=lengths,
        attention=attention,
    )

    error = (actual - expected[start:stop]).abs().max().item()
    tolerance = 1e-4
    elapsed_ms = (time.monotonic() - began) * 1000

    # Also prove the variable-length gather, which the DiT uses once per step.
    gathered = collective.all_gather_rows(q[start:stop].contiguous(), lengths)
    gather_error = (gathered - q).abs().max().item()

    if error > tolerance or gather_error > tolerance:
        raise RuntimeError(
            f"sequence-parallel self-test FAILED: attention max abs error {error:.3e}, "
            f"gather max abs error {gather_error:.3e}, tolerance {tolerance:.1e}. "
            "Refusing to serve: the dual-GPU path is not numerically equivalent to the "
            "single-GPU path on this host."
        )

    return (
        f"selftest=pass attn_max_abs_err={error:.2e} gather_max_abs_err={gather_error:.2e} "
        f"seq={total} shards={lengths} took_ms={elapsed_ms:.0f}"
    )


# --------------------------------------------------------------------------------------
# Boot
# --------------------------------------------------------------------------------------


def boot() -> None:
    """Form the process group and install the patches. Raises to abort ComfyUI startup."""
    from .collectives import init_process_group

    settings = config.resolve_for_rank()
    STATUS.update(
        mode=settings.mode,
        rank=settings.rank,
        world_size=settings.world_size,
        strategy=settings.strategy,
    )
    log(settings.describe())
    for line in describe_devices():
        log(f"[H3-GPU] {line}")
    log(f"[H3-GPU] {stack_versions()}")

    if not settings.dual:
        log("[H3-GPU] single-GPU mode: no model patches installed.")
        STATUS["ready"] = True
        return

    began = time.monotonic()
    log(
        f"[H3-GPU] joining NCCL group rank={settings.rank}/{settings.world_size} at "
        f"{settings.master_addr}:{settings.master_port} (timeout {settings.init_timeout}s)"
    )
    collective = init_process_group(settings)
    log(f"[H3-PERF] distributed_init_ms={round((time.monotonic() - began) * 1000)}")

    report = patches.install(collective, parallel_vae=settings.parallel_vae, log=log)
    log(report.describe())
    for note in report.notes:
        log(f"[H3-GPU] note: {note}")
    STATUS["patched"] = {
        "dit": report.dit,
        "attention": report.attention,
        "vae_decode": report.vae,
        "heads": report.num_heads,
        "blocks": report.num_blocks,
        "notes": report.notes,
    }

    if settings.selftest:
        result = selftest(collective)
        STATUS["selftest"] = result
        log(f"[H3-GPU] {result}")
    else:
        STATUS["selftest"] = "skipped"
        log("[H3-GPU] selftest=skipped (H3_SP_SELFTEST=0)")

    reset_peaks()
    STATUS["ready"] = True
    log("[H3-GPU] ready")


def register_status_route() -> None:
    """Expose /h3/gpu on the private loopback ComfyUI so the handler can read VRAM peaks.

    Read-only and free: it exists because per-GPU peak memory is one of the numbers the
    benchmark needs, and polling it over the HTTP server ComfyUI already runs is cheaper
    and less invasive than instrumenting the model.
    """
    try:
        from aiohttp import web
        from server import PromptServer

        instance = PromptServer.instance
        if instance is None:
            return

        @instance.routes.get("/h3/gpu")
        async def _h3_gpu(request):  # noqa: ANN001 - aiohttp signature
            payload = gpu_report()
            if request.query.get("reset") == "1":
                reset_peaks()
            return web.json_response(payload)

    except Exception as error:  # pragma: no cover - never block boot on a metrics route
        log(f"WARNING: could not register /h3/gpu status route: {error}")
