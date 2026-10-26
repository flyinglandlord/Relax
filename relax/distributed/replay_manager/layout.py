# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Move row coordinates through Megatron layout; materialize only at
consume."""

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from .data import ConsumerBatchLayout, ConsumerSource, ReplayRef


if TYPE_CHECKING:
    from .store import ConsumerReadSession


def pack_token_rows(
    args: Any,
    rows: list[torch.Tensor],
    lengths: list[int],
    pad: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    sequence_parallel: bool,
) -> torch.Tensor:
    """The existing token layout, independent of what each row contains."""
    from megatron.core import mpu

    from relax.backends.megatron.cp_utils import maybe_padded_total_lengths, slice_with_cp

    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    padded = maybe_padded_total_lengths(lengths, args.qkv_format, getattr(args, "is_vl_model", False))
    if padded is not None:
        rows = [pad(value, padded[i] - value.shape[0]) for i, value in enumerate(rows)]
        packed = torch.cat([slice_with_cp(value, pad) for value in rows], dim=0)
    else:
        rows = [pad(value, 1) for value in rows]
        alignment = tp * args.data_pad_size_multiplier
        if args.allgather_cp and cp > 1:
            packed = torch.cat(rows, dim=0)
            count = (-packed.shape[0]) % (cp * alignment)
            if count:
                packed = pad(packed, count)
            packed = packed.chunk(cp, dim=0)[mpu.get_context_parallel_rank()]
        else:
            packed = torch.cat([slice_with_cp(value, pad) for value in rows], dim=0)
            count = (-packed.shape[0]) % alignment
            if count:
                packed = pad(packed, count)
    if sequence_parallel:
        assert packed.shape[0] % tp == 0
        packed = packed.chunk(tp, dim=0)[mpu.get_tensor_model_parallel_rank()]
    return packed


def consumer_batch_layout(
    args: Any,
    refs: list[ReplayRef],
    lengths: list[int],
    *,
    sequence_parallel: bool,
) -> ConsumerBatchLayout:
    if any(m.rows != length - 1 for m, length in zip(refs, lengths, strict=True)):
        raise ValueError("Replay reference row count differs from training tokens")
    rows = [torch.stack((torch.full((m.rows,), i), torch.arange(m.rows)), dim=1) for i, m in enumerate(refs)]

    def pad(value: torch.Tensor, count: int) -> torch.Tensor:
        # Each padding operation restarts its offset, exactly as the original
        # MoE padding formula does. Do not replace this with a global offset.
        padding = torch.stack((torch.full((count,), -1), torch.arange(count)), dim=1)
        return torch.cat((value, padding), dim=0)

    coordinates = pack_token_rows(args, rows, lengths, pad, sequence_parallel=sequence_parallel).numpy()
    return ConsumerBatchLayout.from_coordinates(coordinates)


class ConsumerTensorCache:
    """One train batch/rank's bounded, immutable CPU materializations.

    Shared by both kinds, all layers and microbatches. Eviction drops only our
    tensor references, never reader leases. No GPU tensor or CUDA graph is
    kept. H2D still occurs per consumption; a miss may reread after budget
    eviction.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("Materialization budget must be nonnegative")
        self.capacity_bytes = capacity_bytes
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key: Any) -> Any:
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return value[0]

    def discard(self, key: Any) -> None:
        value = self.entries.pop(key, None)
        if value is not None:
            self.bytes -= value[1]

    def put(self, key: Any, value: tuple[torch.Tensor, torch.Tensor | None]) -> None:
        size = sum(t.numel() * t.element_size() for t in value if t is not None)
        if size > self.capacity_bytes:
            return
        while self.entries and self.bytes + size > self.capacity_bytes:
            self.discard(next(iter(self.entries)))
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


class ConsumerLayerReader(ConsumerSource):
    """Resolve one layer's final microbatch layout at the model's consumption
    boundary."""

    def __init__(
        self,
        reader: "ConsumerReadSession",
        batch_layout: ConsumerBatchLayout,
        *,
        layer: int,
        kind: str,
        device: Any,
        num_experts: int = 0,
        cache: ConsumerTensorCache | None = None,
    ) -> None:
        refs = reader.refs
        if not refs or any(m.kind != kind or m.topk != refs[0].topk or not 0 <= layer < m.layers for m in refs):
            raise ValueError("Incompatible replay manifests for a layer")
        if any(rows.sample_index >= len(refs) for rows in batch_layout.samples):
            raise ValueError("Replay row plan refers to a missing sample")
        if kind == "moe" and num_experts <= 0:
            raise ValueError("MoE replay requires a positive expert count")
        self.reader = reader
        self.batch_layout = batch_layout
        self.layer = layer
        self.kind = kind
        self.device = device
        self.num_experts = num_experts
        self.released = False
        self.cache = cache
        self._cache_key = object()
        reader.retain()

    def read(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.released:
            raise RuntimeError("Replay source has been released")
        cached = self.cache.get(self._cache_key) if self.cache is not None else None
        if cached is None:
            cached = self._materialize()
            if self.cache is not None:
                self.cache.put(self._cache_key, cached)
        staging, mask = cached
        device = torch.device(self.device)
        # Callers own their returned tensors, while cached CPU storage stays
        # immutable. CUDA gets an independent allocation on the current stream.
        values = staging.clone() if device.type == "cpu" else staging.to(device, non_blocking=True)
        if mask is None:
            return values
        return values, mask.clone() if device.type == "cpu" else mask.to(device, non_blocking=True)

    def _materialize(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        import numpy as np

        first = self.reader.refs[0]
        device = torch.device(self.device)
        # Assemble directly into the final dtype on the host. A pageable
        # as_tensor(..., device="cuda") serializes every layer's replay read
        # with the current CUDA stream. Pinned staging lets the copy enqueue
        # on that stream without waiting for preceding model kernels.
        staging = torch.empty(
            (len(self.batch_layout), first.topk),
            dtype=torch.long if self.kind == "moe" else torch.int32,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        output = staging.numpy()
        output.fill(-1)
        for rows in self.batch_layout.samples:
            self.reader.read_into(rows.sample_index, self.layer, rows.source_rows, output, rows.output_rows)
        if self.kind == "moe":
            output[~self.batch_layout.valid] = (
                (self.batch_layout.padding_offsets * first.layers + self.layer) * first.topk + np.arange(first.topk)
            ) % self.num_experts
            return staging, None
        mask = torch.empty(
            self.batch_layout.valid.shape, dtype=torch.bool, device="cpu", pin_memory=device.type == "cuda"
        )
        mask.numpy()[:] = self.batch_layout.valid
        # Never mutate/reuse a cached buffer; PyTorch also tracks outstanding
        # asynchronous copies if an eviction drops the final Python reference.
        return staging, mask

    def release(self) -> None:
        if not self.released:
            if self.cache is not None:
                self.cache.discard(self._cache_key)
            self.reader.release()
            self.released = True
