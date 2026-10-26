# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable node-local shared-memory objects with explicit ownership.

Only metadata is locked. Payloads are write-once, split by layer and row chunk.
A partition hold and reader leases are independent; deleting a queue partition
cannot invalidate a reader. Unadopted publications remain owned by this run and
are reclaimed by explicit run cleanup, never by a timer during training.
"""

import fcntl
import hashlib
import json
import os
import shutil
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, RLock
from typing import Any, Callable, Iterator

from .data import ReplayRef, token_digest


def owner_identity() -> str:
    boot = Path("/proc/sys/kernel/random/boot_id")
    mount = Path("/proc/self/ns/mnt")
    identity = socket.gethostname() + (boot.read_text() if boot.exists() else "")
    if mount.exists():
        identity += os.readlink(mount)
    return hashlib.sha256(identity.encode()).hexdigest()


def _process_start(pid: int) -> str | None:
    """A PID alone is insufficient: job-end cleanup must reject PID reuse."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


class SharedMemoryReplayStore:
    """Own bodies and leases; TQ transports references, never these files."""

    def __init__(
        self,
        root: str,
        run: str,
        *,
        capacity_bytes: int = 16 << 30,
        chunk_rows: int = 4096,
        publication_bytes: int = 64 << 20,
    ) -> None:
        if not run or any(c not in "0123456789abcdef" for c in run):
            raise ValueError("Replay run ID must be a hexadecimal identifier shared by all participants")
        if capacity_bytes <= 0 or chunk_rows <= 0:
            raise ValueError("Replay capacity and chunk rows must be positive")
        self.path = Path(root) / run
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run = run
        self.owner = owner_identity()
        self.capacity_bytes = capacity_bytes
        self.chunk_rows = chunk_rows
        self.publication_bytes = publication_bytes
        self._writer = None

    @contextmanager
    def _state(self, *, write: bool = True) -> Iterator[dict[str, Any]]:
        with (self.path / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            state_file = self.path / "state.json"
            state = (
                json.loads(state_file.read_text())
                if state_file.exists()
                else {"owner": self.owner, "objects": {}, "closed_partitions": [], "closed": False}
            )
            if state["owner"] != self.owner:
                raise RuntimeError("Replay shared-memory store belongs to another node or boot")
            try:
                yield state
            finally:
                if write:
                    tmp = self.path / "state.tmp"
                    tmp.write_text(json.dumps(state))
                    os.replace(tmp, state_file)
                fcntl.flock(lock, fcntl.LOCK_UN)

    def begin_publication(self, *, kind: str, layers: int, topk: int) -> "ProducerPublication":
        if self._writer is None:
            self._writer = _ProducerChunkWriter(self.publication_bytes)
        return ProducerPublication(self, self._writer, kind=kind, layers=layers, topk=topk)

    def start_run(self) -> None:
        """The standard training driver creates a fresh inventory before
        workers start."""
        with self._state() as state:
            if state.get("driver") or state["closed"] or state["objects"] or state["closed_partitions"]:
                raise RuntimeError("Replay run ID has previous state; use a fresh run ID")
            state["driver"] = {"pid": os.getpid(), "start": _process_start(os.getpid())}

    def _chunk_path(self, ref: ReplayRef, chunk: int, layer: int) -> Path:
        return self.path / ref.record / f"{chunk}-{layer}.bin"

    def _object(self, state: dict, ref: ReplayRef) -> dict:
        if ref.owner != self.owner or ref.run != self.run:
            raise RuntimeError("Replay reference belongs to another node or run; this backend is node-local")
        item = state["objects"].get(ref.record)
        if item is None or not item["ready"] or item["manifest"] != ref.to_dict():
            raise RuntimeError("Replay reference is missing, not ready, or has different metadata")
        return item

    def acquire_many(
        self, refs: list[ReplayRef], *, partition: str | None = None, leases: list[str] | None = None
    ) -> list[str]:
        """Validate first, then register one reader per sample in one
        transaction."""
        leases = [uuid.uuid4().hex for _ in refs] if leases is None else leases
        if len(leases) != len(refs) or any(not lease for lease in leases):
            raise ValueError("Every replay reference requires a nonempty reader identity")
        with self._state() as state:
            items = self._partition_items(state, refs, partition)
            for item, lease in zip(items, leases, strict=True):
                if lease not in item["readers"]:
                    item["readers"].append(lease)
        return leases

    @contextmanager
    def reader(self, refs: list[ReplayRef], *, partition: str | None = None) -> Iterator["ConsumerReadSession"]:
        """Keep a construction hold until every layer source has been bound."""
        session = ConsumerReadSession(self, refs, partition=partition)
        try:
            yield session
        except BaseException as primary_error:
            try:
                session.release()
            except Exception as cleanup_error:
                raise primary_error from cleanup_error
            raise
        else:
            session.release()

    def _read_into(
        self, ref: ReplayRef, layer: int, rows: Any, output: Any, destinations: Any, lease: str | None = None
    ) -> None:
        """Copy contiguous runs straight from mmap to caller-owned final
        storage.

        Both source and destination use basic slices: NumPy advanced indexing
        would allocate an entire extra gathered payload before assignment.
        """
        import numpy as np

        rows = np.asarray(rows, dtype=np.int64)
        destinations = np.asarray(destinations, dtype=np.int64)
        if (
            rows.ndim != 1
            or destinations.shape != rows.shape
            or not 0 <= layer < ref.layers
            or np.any(rows < 0)
            or np.any(rows >= ref.rows)
            or output.ndim != 2
            or output.shape[1] != ref.topk
            or output.dtype not in (np.int32, np.int64)
            or not output.flags.writeable
            or np.any(destinations < 0)
            or np.any(destinations >= len(output))
        ):
            raise ValueError("Replay row selection or destination is invalid")
        if not len(rows):
            return
        chunks = rows // ref.chunk_rows
        boundaries = np.flatnonzero((np.diff(rows) != 1) | (np.diff(destinations) != 1) | (np.diff(chunks) != 0)) + 1
        opened = {}
        for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(rows)], strict=True):
            chunk = int(chunks[start])
            path = self._chunk_path(ref, int(chunk), layer)
            count = min(ref.chunk_rows, ref.rows - int(chunk) * ref.chunk_rows)
            if chunk not in opened:
                if path.stat().st_size != count * ref.topk * np.dtype(np.int32).itemsize:
                    raise RuntimeError("Replay chunk size differs from manifest")
                opened[chunk] = np.memmap(path, mode="r", dtype=np.int32, shape=(count, ref.topk))
            src, dst, length = int(rows[start] % ref.chunk_rows), int(destinations[start]), int(end - start)
            np.copyto(output[dst : dst + length], opened[chunk][src : src + length], casting="safe")

    def _partition_items(self, state: dict, refs: list[ReplayRef], partition: str | None) -> list[dict]:
        if (
            state["closed"]
            or state.get("closing")
            or (partition is not None and partition in state["closed_partitions"])
        ):
            raise RuntimeError("Replay consumer requested a closed partition/run")
        items = [self._object(state, ref) for ref in refs]
        if partition is not None and any(partition not in item["partitions"] for item in items):
            raise RuntimeError("Replay reference does not belong to the requested train partition")
        return items

    def validate_partition(self, refs: list[ReplayRef], partition: str) -> None:
        """Early queue check; scoped acquire repeats it atomically with lease
        creation."""
        with self._state(write=False) as state:
            self._partition_items(state, refs, partition)

    def attach(self, refs: list[ReplayRef], partition: str) -> None:
        """Reserve a queue hold before put; retain it on an uncertain put."""
        with self._state() as state:
            if partition in state["closed_partitions"] or state["closed"] or state.get("closing"):
                raise RuntimeError("Cannot attach replay to a closed partition/run")
            items = [self._object(state, m) for m in refs]
            for item in items:
                if partition not in item["partitions"]:
                    item["partitions"].append(partition)

    def commit(self, refs: list[ReplayRef]) -> None:
        with self._state() as state:
            items = [self._object(state, m) for m in refs]
            if any(not item["partitions"] for item in items):
                raise RuntimeError("Cannot commit replay without a queue hold")
            for item in items:
                item["publisher"] = False

    def discard(self, refs: list[ReplayRef]) -> None:
        """Release known discarded samples before queue ownership transfers."""
        with self._state() as state:
            items = []
            for ref in refs:
                if ref.owner != self.owner or ref.run != self.run:
                    raise RuntimeError("Cannot discard replay from another node or run")
                if ref.record in state["objects"]:
                    item = self._object(state, ref)
                    if item["partitions"]:
                        raise RuntimeError("Cannot discard queued replay; retire its partition instead")
                    items.append(item)
            for item in items:
                item["publisher"] = False
            self._collect(state)

    def _collect(self, state: dict) -> None:
        for record, item in list(state["objects"].items()):
            if not item["publisher"] and not item["partitions"] and not item["readers"]:
                try:
                    shutil.rmtree(self.path / record)
                except FileNotFoundError:
                    pass
                del state["objects"][record]

    def release_many(self, refs: list[ReplayRef], leases: list[str]) -> None:
        """Idempotent batch release, including retries after partial
        collection."""
        with self._state() as state:
            items = []
            for ref, lease in zip(refs, leases, strict=True):
                if ref.owner != self.owner or ref.run != self.run:
                    raise RuntimeError("Cannot release replay from another node or run")
                item = state["objects"].get(ref.record)
                if item is not None and lease in item["readers"]:
                    items.append((self._object(state, ref), lease))
            for item, lease in items:
                if lease in item["readers"]:
                    item["readers"].remove(lease)
            self._collect(state)

    def retire_partition(self, partition: str) -> None:
        with self._state() as state:
            if partition not in state["closed_partitions"]:
                state["closed_partitions"].append(partition)
            for item in state["objects"].values():
                if partition in item["partitions"]:
                    item["partitions"].remove(partition)
            self._collect(state)

    def close_run(self) -> None:
        """Final job-end cleanup, after every producer and reader stopped.

        Refuse live leases/partitions. A crashed reader requires operator
        recovery, not unsafe timeout-based deletion during training.
        """
        with self._state() as state:
            # Stop new publications while existing writers/readers drain. A failed
            # finalization remains retryable, without reopening publication.
            state["closing"] = True
            if any(
                item.get("writing")
                and (
                    item.get("producer_start") is None
                    or _process_start(item["producer_pid"]) == item["producer_start"]
                )
                for item in state["objects"].values()
            ):
                raise RuntimeError("Replay run still has active publications")
            if any(item["readers"] or item["partitions"] for item in state["objects"].values()):
                raise RuntimeError("Replay run still has readers or queue holds")
            state["closed"] = True
            for item in state["objects"].values():
                item["publisher"] = False
            self._collect(state)
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def reclaim_stopped_publishers(self) -> int:
        """Reclaim orphan bodies after engine shutdown without closing a
        restartable run.

        Unknown/live producers, readers and TQ partitions always retain
        ownership.
        """
        with self._state() as state:
            for item in state["objects"].values():
                start = item.get("producer_start")
                if (
                    start is not None
                    and not item["readers"]
                    and not item["partitions"]
                    and _process_start(item["producer_pid"]) != start
                ):
                    item["publisher"] = False
            self._collect(state)
            return len(state["objects"])


class _ProducerChunkWriter:
    """One bounded queue per store; reserve bytes before taking a slot
    snapshot.

    Workers receive independent CPU arrays, never a KV-slot view. Backpressure
    blocks the scheduler only when the in-flight byte budget is exhausted.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("Publication byte budget must be positive")
        self.capacity_bytes = capacity_bytes
        self.pending_bytes = self.peak_bytes = 0
        self._error = None
        self._condition = Condition()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-publish")

    def submit(self, size: int, snapshot: Callable[[], Any], path: Path):
        if size > self.capacity_bytes:
            raise BufferError("Replay layer chunk exceeds publication budget")
        with self._condition:
            self._condition.wait_for(
                lambda: self._error is not None or self.pending_bytes + size <= self.capacity_bytes
            )
            if self._error is not None:
                raise self._error
            self.pending_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)

        def done() -> None:
            with self._condition:
                self.pending_bytes -= size
                self._condition.notify_all()

        try:
            values = snapshot()  # scheduler thread: copy before KV slots can be reused
            if values.nbytes != size or not values.flags.c_contiguous or not values.flags.owndata:
                raise ValueError("Publication requires an owned contiguous CPU chunk of the reserved size")

            def write():
                try:
                    values.tofile(path)
                except BaseException as exc:
                    with self._condition:
                        # Keep a lightweight sticky failure, not its traceback
                        # retaining the failed chunk. The Future keeps the
                        # original exception for the request's finish/abort.
                        self._error = RuntimeError(f"Replay writer failed: {type(exc).__name__}: {exc}")
                        self._condition.notify_all()
                    raise
                finally:
                    done()

            return self._executor.submit(write)
        except BaseException:
            done()
            raise

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class ProducerPublication:
    """One request/kind's write-once body, invisible to TQ until sealed.

    Full stable chunks can be submitted during generation. Only the last chunk
    may be short. Retraction/abort drains the writer before removing this body.
    """

    def __init__(
        self, store: SharedMemoryReplayStore, writer: _ProducerChunkWriter, *, kind: str, layers: int, topk: int
    ):
        if kind not in {"moe", "indexer"} or min(layers, topk) <= 0:
            raise ValueError("Invalid streaming replay shape/kind")
        self.store, self.writer = store, writer
        self.kind, self.layers, self.topk = kind, layers, topk
        self.chunk_rows = min(store.chunk_rows, writer.capacity_bytes // (topk * 4))
        if not self.chunk_rows:
            raise BufferError("One replay row exceeds publication budget")
        self.record = uuid.uuid4().hex
        self.path = store.path / self.record
        self.rows = 0
        self._futures = []
        self._sealed = self._aborted = False
        with store._state() as state:
            if state["closed"] or state.get("closing"):
                raise RuntimeError("Replay run is closed")
            self.path.mkdir(mode=0o700)
            state["objects"][self.record] = dict(
                manifest=None,
                bytes=0,
                ready=False,
                writing=True,
                producer_pid=os.getpid(),
                producer_start=_process_start(os.getpid()),
                publisher=True,
                partitions=[],
                readers=[],
            )

    def append(self, count: int, snapshot_layer: Callable[[int], Any]) -> None:
        import numpy as np

        if self._sealed or self._aborted or self.rows % self.chunk_rows or not 0 < count <= self.chunk_rows:
            raise ValueError("Replay publication is closed or chunk is not appendable")
        pending = []
        for future in self._futures:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self._futures = pending
        size = count * self.topk * 4
        with self.store._state() as state:
            used = sum(item["bytes"] for item in state["objects"].values())
            if used + size * self.layers > self.store.capacity_bytes:
                raise BufferError("Replay store capacity exhausted during publication")
            state["objects"][self.record]["bytes"] += size * self.layers

        def snapshot(layer: int) -> Any:
            values = snapshot_layer(layer)
            if not isinstance(values, np.ndarray) or values.dtype != np.int32 or values.shape != (count, self.topk):
                raise ValueError("Replay layer snapshot must be int32 [chunk rows, topk]")
            return values

        for layer in range(self.layers):
            self._futures.append(
                self.writer.submit(
                    size,
                    lambda layer=layer: snapshot(layer),
                    self.path / f"{self.rows // self.chunk_rows}-{layer}.bin",
                )
            )
        self.rows += count

    def finish(self, tokens: list[int]) -> ReplayRef:
        if self._sealed or self._aborted or self.rows != len(tokens) - 1:
            raise ValueError("Replay publication does not cover the finalized request")
        for future in self._futures:
            future.result()
        ref = ReplayRef(
            1,
            self.store.owner,
            self.store.run,
            self.record,
            self.kind,
            self.rows,
            self.layers,
            self.topk,
            self.chunk_rows,
            token_digest(tokens),
        )
        with self.store._state() as state:
            state["objects"][self.record].update(manifest=ref.to_dict(), ready=True, writing=False)
        self._futures.clear()
        self._sealed = True
        return ref

    def abort(self) -> None:
        if self._aborted:
            return
        # Drain every worker before deletion: no late writer may resurrect files.
        for future in self._futures:
            try:
                future.result()
            except Exception:
                pass  # The primary failure propagates through submit/finish.
        with self.store._state() as state:
            item = state["objects"].get(self.record)
            if item is not None:
                if item["partitions"] or item["readers"]:
                    raise RuntimeError("Cannot abort a replay body already adopted by training")
                shutil.rmtree(self.path)
                del state["objects"][self.record]
        self._futures.clear()
        self._aborted = True


class ConsumerReadSession:
    """One microbatch/rank's reader holds, shared by its layer sources.

    The builder owns the first reference. Each bound source retains one; only
    the final release updates the store. A local lock excludes close during
    reads, so hot-path reads need no repeated inventory transaction.
    """

    def __init__(self, store: SharedMemoryReplayStore, refs: list[ReplayRef], *, partition: str | None = None) -> None:
        self.store = store
        self.refs = tuple(refs)
        self._leases = store.acquire_many(refs, partition=partition)
        self._users = 1
        self._closing = False
        self._lock = RLock()

    def retain(self) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Replay reader is closing or released")
            self._users += 1

    def read_into(self, sample: int, layer: int, rows: Any, output: Any, destinations: Any) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Replay reader is closing or released")
            self.store._read_into(self.refs[sample], layer, rows, output, destinations, self._leases[sample])

    def release(self) -> None:
        with self._lock:
            if self._users == 0:
                return
            if self._users > 1:
                self._users -= 1
                return
            # Collection may partially succeed before raising. Forbid reads
            # from then on, but retain the final hold locally for a retry.
            self._closing = True
            self.store.release_many(list(self.refs), self._leases)
            self._users = 0
