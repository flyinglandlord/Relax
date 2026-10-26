# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay records, reference identity and row/layer layout descriptions."""

import hashlib
import json
import struct
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .core import ReplayManager

REFERENCE_FIELD = "rollout_replay_refs"
FIELDS = {"moe": "rollout_routed_experts", "indexer": "rollout_indexer_topk"}


@dataclass(frozen=True)
class ConsumerChannelKey:
    """One replay kind and layer; coordinates preserve the backend's layer
    numbering."""

    kind: str
    layer: int
    coordinate: Literal["global_layer", "local_registration"] = "global_layer"


@dataclass(frozen=True)
class ConsumerRecord:
    """Bind one layer/microbatch to a tensor, (tensor, mask), or lazy reader.

    The payload keeps its existing storage; this record adds no tensor copy.
    """

    kind: str
    microbatch_id: int
    payload: Any


class ConsumerSource(ABC):
    """A readable lease bound to one layer/microbatch's final layout.

    read must remain repeatable through forward and recomputation. release
    drops this consumer's lease, never unilaterally deletes shared data. It
    must be idempotent so a failed release can be retried safely. A reference
    implementation must provide both operations; the manager does not guess how
    a locator is resolved or when other ranks finish.
    """

    @abstractmethod
    def read(self) -> Any: ...

    @abstractmethod
    def release(self) -> None: ...


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}q", *tokens)).hexdigest()


@dataclass(frozen=True)
class ReplayRef:
    """Small wire descriptor locating immutable chunks through their run's
    registered publishing owner."""

    version: int
    owner: str
    run: str
    record: str
    kind: str
    rows: int
    layers: int
    topk: int
    chunk_rows: int
    tokens: str

    def __post_init__(self) -> None:
        if self.version != 1 or self.kind not in {"moe", "indexer"}:
            raise ValueError("Unsupported replay manifest version or kind")
        if min(self.rows, self.layers, self.topk, self.chunk_rows) <= 0:
            raise ValueError("Replay manifest dimensions must be positive")
        for name in ("owner", "run", "record", "tokens"):
            value = getattr(self, name)
            if not value or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid replay identity: {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReplayRef":
        return cls(**value)


def encode_refs(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_sample(value: dict[str, Any], tokens: list[int]) -> None:
    if not value:
        raise ValueError("Missing replay reference")
    for kind, item in value.items():
        ref = ReplayRef.from_dict(item)
        if kind != ref.kind or ref.rows != len(tokens) - 1 or ref.tokens != token_digest(tokens):
            raise ValueError("Replay reference does not match sample tokens or kind")


@dataclass(frozen=True)
class ConsumerSampleRowMap:
    """Map one sample's captured rows to this rank's microbatch rows."""

    sample_index: int
    output_rows: "NDArray[np.int64]"
    source_rows: "NDArray[np.int64]"


@dataclass(frozen=True)
class ConsumerBatchLayout:
    """Selected microbatch's final row positions, shared by all local layers.

    This describes existing packing/padding and rank slices, not TQ sampling.
    """

    size: int
    samples: tuple[ConsumerSampleRowMap, ...]
    valid: "NDArray[np.bool_]"
    padding_offsets: "NDArray[np.int64]"

    def __len__(self) -> int:
        return self.size

    @classmethod
    def from_coordinates(cls, coordinates: "NDArray[np.int64]") -> "ConsumerBatchLayout":
        import numpy as np

        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or np.any(coordinates[:, 0] < -1):
            raise ValueError("Replay coordinates must be [sample index, source row], with -1 for padding")
        sample_ids, source_rows = coordinates[:, 0], coordinates[:, 1]
        valid = sample_ids >= 0
        samples = []
        for sample_index in np.unique(sample_ids[valid]):
            positions = np.flatnonzero(sample_ids == sample_index)
            samples.append(ConsumerSampleRowMap(int(sample_index), positions, source_rows[positions]))
        return cls(len(coordinates), tuple(samples), valid, source_rows[~valid, None])


@dataclass(frozen=True)
class ConsumerLayerBinding:
    """Map a local model channel to its captured layer; no body reads occur
    here."""

    capture_layer: int
    manager: "ReplayManager"
    channel: ConsumerChannelKey


@dataclass(frozen=True)
class ConsumerModelLayout:
    """Captured shape and local model-layer destinations for one replay
    kind."""

    kind: str
    topk: int
    capture_layers: int
    bindings: tuple[ConsumerLayerBinding, ...]
