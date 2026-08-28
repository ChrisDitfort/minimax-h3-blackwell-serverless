"""Collective backends for the sequence-parallel H3 path.

Two implementations of the same three-method interface:

    TorchDistCollective   real NCCL over torch.distributed, used in the container
    ThreadCollective      both ranks in one process, used by the CPU tests

The interface is deliberately tiny - an all-to-all and a broadcast - because everything
else Ulysses needs (variable-length gathers) is built on top of broadcast, which every
backend supports at arbitrary and unequal sizes. torch.distributed's `all_gather` is only
defined for equal shapes on NCCL, and the H3 sequence does not divide evenly, so relying
on it would have been a latent shape bug.
"""

from __future__ import annotations

import datetime
import os
import threading

import torch


class Collective:
    """Rank-aware collectives. Implementations must be safe to call from one thread."""

    rank: int = 0
    world_size: int = 1

    # -- primitives --------------------------------------------------------------------

    def all_to_all(
        self,
        send: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """Scatter `send` by `input_splits` to peers, gather their pieces back, flat."""
        raise NotImplementedError

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """In-place broadcast from `src`. Returns the tensor for convenience."""
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError

    # -- derived -----------------------------------------------------------------------

    def all_gather_varied(
        self, tensor: torch.Tensor, shapes: list[tuple[int, ...]]
    ) -> list[torch.Tensor]:
        """Gather one tensor per rank, each with its own shape, onto every rank.

        Built from broadcast so unequal shapes are legal. `shapes[i]` must be what rank i
        contributes, and every rank must already agree on the list - which they do here,
        because the shard plan and the VAE chunk plan are both computed identically from
        inputs every rank already has.
        """
        if self.world_size == 1:
            return [tensor]

        gathered: list[torch.Tensor] = []
        for src in range(self.world_size):
            if src == self.rank:
                buffer = tensor.contiguous()
            else:
                buffer = torch.empty(
                    shapes[src], dtype=tensor.dtype, device=tensor.device
                )
            gathered.append(self.broadcast(buffer, src))
        return gathered

    def all_gather_rows(self, tensor: torch.Tensor, row_counts: list[int]) -> torch.Tensor:
        """Gather shards that differ only in their leading dimension, concatenated."""
        if self.world_size == 1:
            return tensor
        trailing = tuple(tensor.shape[1:])
        shapes = [(count,) + trailing for count in row_counts]
        return torch.cat(self.all_gather_varied(tensor, shapes), dim=0)


class TorchDistCollective(Collective):
    """NCCL-backed collectives over an already-initialised torch.distributed group."""

    def __init__(self, rank: int, world_size: int, group=None) -> None:
        self.rank = rank
        self.world_size = world_size
        self._group = group

    def all_to_all(
        self,
        send: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        import torch.distributed as dist

        send = send.contiguous()
        received = torch.empty(sum(output_splits), dtype=send.dtype, device=send.device)
        dist.all_to_all_single(
            received,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self._group,
        )
        return received

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        import torch.distributed as dist

        dist.broadcast(tensor, src=src, group=self._group)
        return tensor

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier(group=self._group)


class ThreadGroup:
    """Rendezvous for ThreadCollective: one per simulated cluster."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.slots: list = [None] * world_size

    def exchange(self, rank: int, value):
        """Publish `value` from `rank`, then return every rank's value."""
        self.slots[rank] = value
        self.barrier.wait()
        published = list(self.slots)
        self.barrier.wait()
        return published


class ThreadCollective(Collective):
    """Runs every rank in the same process, for testing the algebra without NCCL."""

    def __init__(self, group: ThreadGroup, rank: int) -> None:
        self._group = group
        self.rank = rank
        self.world_size = group.world_size

    def all_to_all(
        self,
        send: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        offsets = [sum(input_splits[:i]) for i in range(len(input_splits))]
        outgoing = [
            send[offset : offset + size].clone()
            for offset, size in zip(offsets, input_splits)
        ]
        published = self._group.exchange(self.rank, outgoing)
        received = [published[src][self.rank] for src in range(self.world_size)]

        for src, piece in enumerate(received):
            if piece.numel() != output_splits[src]:
                raise ValueError(
                    f"rank {self.rank} expected {output_splits[src]} elements from rank "
                    f"{src} but got {piece.numel()}"
                )
        return torch.cat(received)

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        published = self._group.exchange(
            self.rank, tensor.clone() if self.rank == src else None
        )
        source = published[src]
        if self.rank != src:
            tensor.copy_(source)
        return tensor

    def barrier(self) -> None:
        self._group.barrier.wait()


# --------------------------------------------------------------------------------------
# Process group lifecycle
# --------------------------------------------------------------------------------------


def init_process_group(config, *, device_index: int = 0):
    """Join the worker's NCCL group. Returns a TorchDistCollective, or raises.

    Called once per rank at ComfyUI import time, not per job: NCCL bootstrap costs a
    second or two and a warm worker must not pay it again on every generation. The
    timeout is bounded so a peer that never arrives fails the boot loudly instead of
    hanging the endpoint forever.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return TorchDistCollective(dist.get_rank(), dist.get_world_size())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "dual mode needs a CUDA device in every rank, and this process can see none. "
            "Check that the endpoint really has 2 GPUs per worker and that "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
            "selects one of them."
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"rank {config.rank} was told to use cuda:{device_index} but only "
            f"{torch.cuda.device_count()} device(s) are visible to it."
        )

    os.environ.setdefault("MASTER_ADDR", config.master_addr)
    os.environ.setdefault("MASTER_PORT", str(config.master_port))
    # A collective that never returns must kill the job, not wedge the worker.
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.cuda.set_device(device_index)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{config.master_addr}:{config.master_port}",
        world_size=config.world_size,
        rank=config.rank,
        timeout=datetime.timedelta(seconds=config.init_timeout),
        device_id=torch.device("cuda", device_index),
    )
    return TorchDistCollective(config.rank, config.world_size)
