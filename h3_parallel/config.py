"""GPU execution mode resolution for the experimental multi-GPU H3 worker.

Pure configuration: no torch, no CUDA, no ComfyUI. Everything here is decided from
environment variables plus a device count the caller supplies, so the decision table is
testable on a laptop with no GPU at all.

Two modes exist:

    H3_GPU_MODE=single   the known-good path. Nothing in this package activates; one
                         ComfyUI process owns one GPU exactly as it does today.
    H3_GPU_MODE=dual     one generation is split across two GPUs by sequence
                         parallelism (see ulysses.py).

`dual` is never inferred. A worker that happens to be scheduled two GPUs still runs the
single-GPU path unless the endpoint asks for dual explicitly, so the A/B comparison can
never be contaminated by an accidental mode switch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# The strategy name that goes into the [H3-GPU] log line and the perf summary. Kept as a
# constant so the log, the tests and the report cannot drift apart.
STRATEGY_ULYSSES_SP = "ulysses-sequence-parallel"

SINGLE = "single"
DUAL = "dual"

VALID_MODES = (SINGLE, DUAL)

#: MiniMax H3 has 56 attention heads. Ulysses shards heads across ranks, so the world size
#: has to divide it. Read from the model at patch time; this is only the default used when
#: resolving configuration before any model is loaded.
H3_NUM_HEADS = 56

DEFAULT_MASTER_PORT = 29513
DEFAULT_COMFY_PORT = 8188

#: NCCL transports to try, in order, when nothing has been pinned by the operator.
#:
#: Two ranks on one host can talk over CUDA peer-to-peer, over /dev/shm, or over loopback
#: sockets. A container can break the first two without breaking communicator *creation* -
#: which is exactly the failure seen in production: `init_process_group` returned in 743 ms
#: and the first real collective then hung forever. Rather than make an operator guess one
#: environment variable per rollout, the worker walks this ladder itself and reports which
#: rung carried traffic.
#:
#: The last rung always works. It is slow - loopback sockets instead of a GPU
#: interconnect - but a slow measurement beats no measurement, and it proves the
#: parallel path end to end.
NCCL_TRANSPORTS = (
    ("auto", {}, "whatever NCCL selects on its own"),
    ("no-shm", {"NCCL_SHM_DISABLE": "1"}, "peer-to-peer or sockets; /dev/shm unused"),
    ("no-p2p", {"NCCL_P2P_DISABLE": "1"}, "shared memory or sockets; no GPU peer access"),
    ("sockets", {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"},
     "loopback sockets only - slow, but available in any container"),
)

#: Set either of these on the endpoint and the ladder is skipped: an explicit operator
#: choice is never second-guessed.
NCCL_PINNING_VARS = ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_NET")


class ConfigurationError(RuntimeError):
    """The requested GPU configuration cannot be honoured."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class GpuConfig:
    """The resolved execution plan for this worker."""

    mode: str
    world_size: int
    #: Rank of *this* process. The handler is rank -1 (it is not part of the group).
    rank: int
    strategy: str
    master_addr: str
    master_port: int
    #: Parallelise MiniMaxH3VideoVAE.decode_temporal across ranks as well as the DiT.
    parallel_vae: bool
    #: Seconds any collective may block - the initial rendezvous included - before NCCL
    #: declares the group dead. One knob rather than two: the rendezvous is the only
    #: collective that legitimately takes more than milliseconds, so it sets the bound.
    init_timeout: int
    #: Seconds a tiny transport probe may take before the rank gives up and says so.
    #: Deliberately short: a hung collective otherwise burns the eight minutes RunPod
    #: waits before killing a worker that never became ready, and logs nothing at all.
    probe_timeout: int
    #: Seconds the handler waits for one transport attempt to produce two ready ranks.
    attempt_timeout: int
    #: Run the boot-time correctness self-test before serving traffic.
    selftest: bool
    #: When dual was requested but cannot be honoured, degrade to single instead of failing.
    allow_fallback: bool
    #: Why dual was downgraded to single, when it was. None on a plain single-mode worker.
    fallback_reason: str | None = None

    @property
    def dual(self) -> bool:
        return self.mode == DUAL

    def describe(self) -> str:
        """The one-line [H3-GPU] banner every process prints at boot."""
        if not self.dual:
            banner = "[H3-GPU] mode=single gpu_count=1 strategy=none"
            if self.fallback_reason:
                # Never silent. A run whose logs say dual but whose worker ran single
                # would be the single most misleading thing this image could produce.
                banner += (
                    " downgraded_from=dual reason="
                    f"{self.fallback_reason!r} (H3_SP_ALLOW_FALLBACK=1)"
                )
            return banner
        return (
            f"[H3-GPU] mode={self.mode} gpu_count={self.world_size} "
            f"strategy={self.strategy} rank={self.rank} world_size={self.world_size} "
            f"backend=nccl parallel_vae={'true' if self.parallel_vae else 'false'}"
        )


def requested_mode() -> str:
    """The mode the endpoint asked for, before any capability check."""
    raw = _env("H3_GPU_MODE", SINGLE).lower()
    if raw not in VALID_MODES:
        raise ConfigurationError(
            f"H3_GPU_MODE={raw!r} is not valid; expected one of {', '.join(VALID_MODES)}."
        )
    return raw


def resolve(
    *,
    device_count: int,
    rank: int | None = None,
    num_heads: int = H3_NUM_HEADS,
) -> GpuConfig:
    """Decide how this process should execute, or explain why it cannot.

    `device_count` is what the *whole worker* can see - the handler passes
    torch.cuda.device_count() before it narrows CUDA_VISIBLE_DEVICES for the children.
    A child process passes the world size it was given instead, because by then it can
    only see its own GPU.

    Raises ConfigurationError when dual is requested and cannot be delivered, unless
    H3_SP_ALLOW_FALLBACK=1. Failing loudly at boot is deliberate: a benchmark image that
    quietly serves single-GPU results would invalidate the very measurement it exists to
    produce.
    """
    mode = requested_mode()
    allow_fallback = _env_flag("H3_SP_ALLOW_FALLBACK", False)
    world_size = _env_int("H3_SP_WORLD_SIZE", 2)

    common = {
        "master_addr": _env("H3_SP_MASTER_ADDR", "127.0.0.1"),
        "master_port": _env_int("H3_SP_MASTER_PORT", DEFAULT_MASTER_PORT),
        "parallel_vae": _env_flag("H3_SP_VAE", True),
        "init_timeout": _env_int("H3_SP_INIT_TIMEOUT", 300),
        "probe_timeout": _env_int("H3_SP_PROBE_TIMEOUT", 25),
        "attempt_timeout": _env_int("H3_SP_ATTEMPT_TIMEOUT", 150),
        "selftest": _env_flag("H3_SP_SELFTEST", True),
        "allow_fallback": allow_fallback,
    }

    def single(reason: str | None = None) -> GpuConfig:
        return GpuConfig(
            mode=SINGLE,
            world_size=1,
            rank=0,
            strategy="none",
            fallback_reason=reason,
            **common,
        )

    if mode == SINGLE:
        return single()

    problem = _dual_blocker(device_count=device_count, world_size=world_size, num_heads=num_heads)
    if problem is not None:
        if not allow_fallback:
            raise ConfigurationError(
                f"H3_GPU_MODE=dual was requested but cannot be honoured: {problem} "
                "Refusing to start rather than silently benchmarking the single-GPU path. "
                "Set H3_SP_ALLOW_FALLBACK=1 to degrade to single-GPU instead."
            )
        return single(problem)

    return GpuConfig(
        mode=DUAL,
        world_size=world_size,
        rank=-1 if rank is None else rank,
        strategy=STRATEGY_ULYSSES_SP,
        **common,
    )


def _dual_blocker(*, device_count: int, world_size: int, num_heads: int) -> str | None:
    """Return the reason dual mode is impossible, or None when it is available."""
    if world_size < 2:
        return f"H3_SP_WORLD_SIZE={world_size} leaves nothing to parallelise."
    if device_count < world_size:
        return (
            f"the worker can see {device_count} CUDA device(s) but dual mode needs "
            f"{world_size}. Configure the RunPod endpoint for {world_size} GPUs per worker."
        )
    if num_heads % world_size != 0:
        return (
            f"the H3 transformer has {num_heads} attention heads, which does not divide "
            f"evenly across {world_size} ranks. Ulysses shards heads, so it cannot run."
        )
    return None


def transport_ladder() -> list[tuple[str, dict, str]]:
    """The NCCL transports this worker should try, in order.

    Returns a single entry when the operator has pinned one - either by naming it in
    H3_SP_TRANSPORT or by setting an NCCL variable directly on the endpoint. Walking a
    ladder over the top of somebody's deliberate choice would make the logs lie about what
    was actually tested.
    """
    pinned = [name for name in NCCL_PINNING_VARS if _env(name)]
    if pinned:
        return [("operator-pinned", {}, f"pinned by {', '.join(pinned)} on the endpoint")]

    requested = _env("H3_SP_TRANSPORT").lower()
    if requested and requested != "auto":
        for name, overlay, description in NCCL_TRANSPORTS:
            if name == requested:
                return [(name, overlay, description)]
        raise ConfigurationError(
            f"H3_SP_TRANSPORT={requested!r} is not a known transport; expected one of "
            + ", ".join(name for name, _, _ in NCCL_TRANSPORTS)
        )

    return list(NCCL_TRANSPORTS)


def resolve_for_rank() -> GpuConfig:
    """The plan for a ComfyUI process that was launched as a member of the group.

    Deliberately does *not* re-check the device count. Each rank is started with
    CUDA_VISIBLE_DEVICES narrowed to its own GPU, so it can only ever see one - the
    handler is the process that verified there were two before it launched anything.
    """
    mode = requested_mode()
    if mode == SINGLE:
        return resolve(device_count=1)

    rank = child_rank()
    world_size = _env_int("H3_SP_WORLD_SIZE", 2)
    if rank is None:
        raise ConfigurationError(
            "H3_GPU_MODE=dual but H3_SP_RANK is not set in this process. Ranks are "
            "launched by the handler; ComfyUI must not be started by hand in dual mode."
        )
    if not 0 <= rank < world_size:
        raise ConfigurationError(
            f"H3_SP_RANK={rank} is outside the world size {world_size}."
        )

    return GpuConfig(
        mode=DUAL,
        world_size=world_size,
        rank=rank,
        strategy=STRATEGY_ULYSSES_SP,
        master_addr=_env("H3_SP_MASTER_ADDR", "127.0.0.1"),
        master_port=_env_int("H3_SP_MASTER_PORT", DEFAULT_MASTER_PORT),
        parallel_vae=_env_flag("H3_SP_VAE", True),
        init_timeout=_env_int("H3_SP_INIT_TIMEOUT", 300),
        probe_timeout=_env_int("H3_SP_PROBE_TIMEOUT", 25),
        attempt_timeout=_env_int("H3_SP_ATTEMPT_TIMEOUT", 150),
        selftest=_env_flag("H3_SP_SELFTEST", True),
        allow_fallback=_env_flag("H3_SP_ALLOW_FALLBACK", False),
    )


def child_rank() -> int | None:
    """The rank this process was launched as, or None if it is not a group member."""
    raw = _env("H3_SP_RANK")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def comfy_port_for_rank(rank: int, base_port: int = DEFAULT_COMFY_PORT) -> int:
    """Each rank runs its own private ComfyUI, so each needs its own loopback port."""
    return base_port + rank
