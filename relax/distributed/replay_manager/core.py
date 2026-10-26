# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay manager: wire data, train-region ownership and local consumer
state."""

from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any

from .config import ReplayConfig, get_store
from .data import (
    FIELDS,
    REFERENCE_FIELD,
    ConsumerChannelKey,
    ConsumerRecord,
    ConsumerSource,
    ReplayRef,
)


if TYPE_CHECKING:
    from .megatron import Consumer
    from .sglang import Producer
    from .store import ConsumerReadSession, SharedMemoryReplayStore
    from .transport import RoutedReplayStore


def refs_from_samples(samples: list[Any]) -> list[ReplayRef]:
    return [ReplayRef.from_dict(item) for sample in samples for item in (sample.rollout_replay_refs or {}).values()]


class ConsumerPhase:
    def __init__(self, *, environment: MutableMapping[str, str] | None = None, key: str | None = None) -> None:
        if (environment is None) != (key is None):
            raise ValueError("A phase environment and key must be supplied together")
        self._environment = environment
        self._key = key
        self._stage = "fallthrough"

    def get(self) -> str:
        # Do not silently initialize a missing legacy environment variable.
        return self._stage if self._environment is None else self._environment[self._key]

    def set(self, stage: str) -> None:
        if stage not in {"fallthrough", "record", "replay_forward", "replay_backward"}:
            raise ValueError(f"Invalid replay stage {stage!r}")
        if self._environment is None:
            self._stage = stage
        else:
            self._environment[self._key] = stage

    def enter_forward(self) -> str:
        previous = self.get()
        self.set("replay_forward")
        return previous


class ConsumerConsumptionError(RuntimeError):
    def __init__(self, phase: str, consumed: int, recorded: int) -> None:
        self.phase = phase
        self.consumed = consumed
        self.recorded = recorded
        super().__init__(f"Replay {phase} consumed {consumed}/{recorded} records")


@dataclass
class ConsumerChannel:
    entries: list[Any] = field(default_factory=list)
    microbatch_ids: dict[int, int] = field(default_factory=dict)
    forward_index: int = 0
    backward_index: int = 0

    def release_sources(self) -> None:
        # Inline tensors retain Python's original ownership semantics. A bound
        # source owns a consumer lease, which is released only at this boundary.
        error = None
        for payload in self.entries:
            if isinstance(payload, ConsumerSource):
                try:
                    payload.release()
                except Exception as exc:
                    error = error or exc
        if error is not None:
            # Keep failed leases reachable for retry, while still attempting
            # every release. A failed clear must not be reported as success.
            raise error

    def clear(self) -> None:
        self.release_sources()
        # Rebinding preserves the original MoE API's externally held lists.
        self.entries = []
        self.microbatch_ids = {}
        self.forward_index = self.backward_index = 0


class ReplayManager:
    """A model adapter owns this manager and decides when its records expire.

    Full/reference delivery follows one immutable configuration. Local channels
    own binding, repeated reads and consumer leases. Adapters retain model
    semantics, layout, capture placement and legacy error policy.
    """

    def __init__(
        self,
        *,
        config: ReplayConfig | None = None,
        phase: ConsumerPhase | None = None,
        store_factory: Callable[[], "SharedMemoryReplayStore | RoutedReplayStore"] | None = None,
    ) -> None:
        self.config = config if config is not None else ReplayConfig()
        self.phase = phase if phase is not None else ConsumerPhase()
        self._channels: dict[ConsumerChannelKey, ConsumerChannel] = {}
        self._closed = False
        self._store_factory = store_factory
        self._owns_run = False

    def start_run(self) -> None:
        """Called by Relax's training entry point, before constructing
        workers."""
        if self.uses_references:
            self.store().start_run()
            self._owns_run = True

    def finish_run(self) -> None:
        """Called by standard normal/error shutdown, never by an in-job
        restart."""
        if self._owns_run:
            self.store().close_run()
            self._owns_run = False

    @property
    def uses_references(self) -> bool:
        return self.config.delivery == "reference"

    def store(self) -> "SharedMemoryReplayStore | RoutedReplayStore":
        self._ensure_open()
        if not self.uses_references:
            raise RuntimeError("Full replay delivery does not use a body store")
        return self._store_factory() if self._store_factory is not None else get_store(self.config)

    def reader(
        self,
        refs: list["ReplayRef"],
        *,
        partition: str | None = None,
    ) -> AbstractContextManager["ConsumerReadSession"]:
        return self.store().reader(refs, partition=partition)

    def register_producer(self, kinds: Iterable[str]) -> "Producer":
        from .sglang import Producer

        self._ensure_open()
        kinds = frozenset(kinds)
        if not kinds.issubset(FIELDS):
            raise ValueError(f"Unsupported replay kinds: {kinds - FIELDS.keys()}")
        return Producer(self, kinds)

    def register_consumer(self, kinds: Iterable[str] = ()) -> "Consumer":
        from .megatron import Consumer

        self._ensure_open()
        kinds = frozenset(kinds)
        if not kinds.issubset(FIELDS):
            raise ValueError(f"Unsupported replay kinds: {kinds - FIELDS.keys()}")
        return Consumer(self, kinds)

    def region(self, partition: str) -> "ReplayRegion":
        self._ensure_open()
        if not isinstance(partition, str) or not partition:
            raise ValueError("Replay region requires an explicit TQ partition identity")
        return ReplayRegion(self, partition)

    def sample_refs(self, samples: Sequence[Any]) -> list["ReplayRef"]:
        return refs_from_samples(samples) if self.uses_references else []

    def discard(self, samples: Sequence[Any]) -> None:
        refs = self.sample_refs(samples)
        if refs:
            self.store().discard(refs)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Replay manager is closed")

    def build_tq_fields(
        self, samples: Sequence[Any], *, required_fields: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        self._ensure_open()
        if self.uses_references:
            from .data import REFERENCE_FIELD, validate_sample

            values = []
            for sample in samples:
                refs = sample.rollout_replay_refs
                if not refs:
                    raise RuntimeError("Reference mode requires SGLang-published replay manifests")
                validate_sample(refs, sample.tokens)
                for kind, field in FIELDS.items():
                    if field in required_fields and kind not in refs:
                        raise RuntimeError(f"Reference mode is missing {kind} replay")
                values.append(refs)
            return {REFERENCE_FIELD: values}
        import numpy as np

        unknown = required_fields - frozenset(FIELDS.values())
        if unknown:
            raise KeyError(f"Unsupported replay fields: {sorted(unknown)}")
        fields = {}
        if samples[0].rollout_routed_experts is not None:
            fields["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
        if "rollout_indexer_topk" in required_fields:
            values = [getattr(sample, "rollout_indexer_topk", None) for sample in samples]
            if any(value is None for value in values):
                raise RuntimeError("Every training sample must contain rollout_indexer_topk when replay is enabled")
            widths = set()
            for sample, value in zip(samples, values, strict=False):
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"rollout_indexer_topk must be a numpy array, got {type(value)}")
                if value.ndim != 2 or value.shape[0] != len(sample.tokens) - 1:
                    raise RuntimeError(
                        "rollout_indexer_topk must be a 2-D array with one row per input token except the last, "
                        f"got replay={value.shape}, tokens={len(sample.tokens)}"
                    )
                if value.dtype != np.int32:
                    raise RuntimeError(f"rollout_indexer_topk must use int32, got {value.dtype}")
                widths.add(value.shape[1])
            if len(widths) != 1:
                raise RuntimeError(f"Inconsistent rollout_indexer_topk widths in one batch: {sorted(widths)}")
            fields["rollout_indexer_topk"] = values
        return fields

    def handles(self, field: str) -> bool:
        return field in ({REFERENCE_FIELD} if self.uses_references else FIELDS.values())

    def pack(self, field: str, values: list[Any]) -> Any:
        self._ensure_open()
        if self.uses_references:
            import torch

            from .data import encode_refs

            if field not in {REFERENCE_FIELD}:
                raise KeyError(field)
            return torch.nested.as_nested_tensor(
                [torch.frombuffer(bytearray(encode_refs(value)), dtype=torch.uint8) for value in values],
                layout=torch.jagged,
            )
        import numpy as np
        import torch

        if field not in frozenset(FIELDS.values()):
            raise KeyError(field)
        # Preserve the original CPU/int32 jagged wire format, including when
        # other TensorDict fields are being packed onto a requested device.
        tensors = [
            torch.from_numpy(np.ascontiguousarray(value.reshape(value.shape[0], -1))).to(torch.int32)
            for value in values
        ]
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    def receive(self, field: str, values: Iterable[Any], *, device: Any) -> list[Any]:
        self._ensure_open()
        if self.uses_references:
            import json

            from .data import ReplayRef

            if field not in {REFERENCE_FIELD}:
                raise KeyError(field)
            result = []
            for value in values:
                # Descriptors remain CPU metadata: never copy replay contents to GPU here.
                decoded = json.loads(value.cpu().numpy().tobytes())
                for item in decoded.values():
                    ReplayRef.from_dict(item)
                result.append(decoded)
            return result
        import torch
        from tensordict.tensorclass import NonTensorData

        if field not in frozenset(FIELDS.values()):
            raise KeyError(field)
        dtype = torch.long if field == "rollout_routed_experts" else torch.int32
        return [
            torch.as_tensor(value.data if isinstance(value, NonTensorData) else value, dtype=dtype, device=device)
            for value in values
        ]

    def register_channel(self, key: ConsumerChannelKey) -> ConsumerChannel:
        self._ensure_open()
        if key in self._channels:
            raise ValueError(f"Replay channel already registered: {key}")
        channel = ConsumerChannel()
        self._channels[key] = channel
        return channel

    def _channel(self, key: ConsumerChannelKey) -> ConsumerChannel:
        self._ensure_open()
        return self._channels[key]

    def bind(self, key: ConsumerChannelKey, record: ConsumerRecord) -> None:
        """Bind an already laid-out layer/microbatch to its consumer
        channel."""
        channel = self._channel(key)
        if record.kind != key.kind or record.microbatch_id != len(channel.entries):
            raise ValueError("Replay kind or microbatch order does not match the channel")
        channel.microbatch_ids[len(channel.entries)] = record.microbatch_id
        channel.entries.append(record.payload)

    def consume(
        self,
        key: ConsumerChannelKey,
        phase: str | None = None,
        *,
        microbatch_id: int | None = None,
        advance_before_lookup: bool = False,
    ) -> Any:
        channel = self._channel(key)
        if phase is None:
            phase = self.phase.get()
        if phase not in ("replay_forward", "replay_backward"):
            raise ValueError(f"Cannot consume replay in stage {phase!r}")
        cursor = "forward_index" if phase == "replay_forward" else "backward_index"
        index = getattr(channel, cursor)
        if microbatch_id is not None:
            recorded_id = channel.microbatch_ids.get(index)
            if recorded_id is None or recorded_id != microbatch_id:
                raise RuntimeError(f"Replay microbatch mismatch: expected={microbatch_id}, cursor={index}")
        # Legacy Indexer advances even on overrun; MoE advances only on success.
        if advance_before_lookup:
            setattr(channel, cursor, index + 1)
        payload = channel.entries[index]
        if isinstance(payload, ConsumerSource):
            payload = payload.read()
        if not advance_before_lookup:
            setattr(channel, cursor, index + 1)
        return payload

    def reset_forward(self, key: ConsumerChannelKey, *, require_consumed: bool = False) -> None:
        channel = self._channel(key)
        if require_consumed and channel.forward_index != len(channel.entries):
            raise ConsumerConsumptionError("replay_forward", channel.forward_index, len(channel.entries))
        channel.forward_index = 0

    def check_consumed(self, key: ConsumerChannelKey, *, expect_backward: bool = False) -> None:
        channel = self._channel(key)
        cursors = ("forward_index", "backward_index") if expect_backward else ("forward_index",)
        for cursor in cursors:
            count = getattr(channel, cursor)
            if count != len(channel.entries):
                phase = "replay_forward" if cursor == "forward_index" else "replay_backward"
                raise ConsumerConsumptionError(phase, count, len(channel.entries))

    def clear(self) -> None:
        error = None
        for channel in self._channels.values():
            try:
                channel.clear()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def release_sources(self) -> None:
        """Release bound leases without clearing externally held inline
        views."""
        error = None
        for channel in self._channels.values():
            try:
                channel.release_sources()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def close(self) -> None:
        self.clear()
        self._channels.clear()
        self._closed = True


@dataclass(frozen=True)
class ReplayRegion:
    manager: "ReplayManager"
    partition: str

    @contextmanager
    def queue_ownership(self, samples: Sequence[Any]) -> Iterator[None]:
        """Hold reference bodies around the caller's queue write.

        Transfer publisher ownership only on success. An exception or
        cancellation keeps the holds because queue delivery may be uncertain.
        """
        self.manager._ensure_open()
        refs = self.manager.sample_refs(samples)
        if refs:
            self.manager.store().attach(refs, self.partition)
        yield
        if refs:
            self.manager.store().commit(refs)

    def retire(self) -> None:
        self.manager._ensure_open()
        if self.manager.uses_references:
            self.manager.store().retire_partition(self.partition)

    def reader(self, refs: list[ReplayRef]) -> AbstractContextManager["ConsumerReadSession"]:
        self.manager._ensure_open()
        return self.manager.reader(refs, partition=self.partition)

    def validate(self, batch: dict[str, Any] | None) -> None:
        self.manager._ensure_open()
        if batch is None or REFERENCE_FIELD not in batch:
            return
        refs = [ReplayRef.from_dict(item) for sample_refs in batch[REFERENCE_FIELD] for item in sample_refs.values()]
        self.manager.store().validate_partition(refs, self.partition)


replay_data_manager = ReplayManager(config=ReplayConfig.from_env())


def install_tq_cleanup_hook() -> None:
    """Link TQ partition cleanup to Replay once, before clients are created.

    TQ's synchronous client binds this same async method at construction.
    Partial sample deletion is not a partition boundary and stays unchanged.
    """
    if not replay_data_manager.uses_references:
        return

    from transfer_queue.client import AsyncTransferQueueClient

    original = AsyncTransferQueueClient.async_clear_partition
    if getattr(original, "_relax_replay_cleanup", False):
        return

    @wraps(original)
    async def clear_partition(client: Any, partition_id: str) -> Any:
        # A failed/cancelled TQ clear must keep its reference bodies live.
        result = await original(client, partition_id=partition_id)
        # Retirement removes the partition hold, never an active reader lease.
        replay_data_manager.region(partition_id).retire()
        return result

    clear_partition._relax_replay_cleanup = True
    AsyncTransferQueueClient.async_clear_partition = clear_partition
