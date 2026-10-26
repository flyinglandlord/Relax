# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Resolve replay bodies on their publishing node, without changing TQ flow.

Local reads keep the mmap-to-pinned path. Only remote reads use Ray, requesting
bounded layer/row slices. The run registry routes ownership operations; it
never receives tensor bodies or decides when a training step should run.
"""

import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .data import ReplayRef
from .store import ConsumerReadSession, SharedMemoryReplayStore, _process_start, owner_identity


_NAMESPACE = "relax-replay"


def _ray() -> Any:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Replay transport requires the existing Relax Ray connection")
    return ray


def _call(actor: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    return _ray().get(getattr(actor, method).remote(*args, **kwargs))


def _local_store(config: Any) -> SharedMemoryReplayStore:
    return SharedMemoryReplayStore(
        str(Path(config.shm_root).resolve() / owner_identity()),
        config.run_id,
        capacity_bytes=config.capacity_bytes,
        chunk_rows=config.chunk_rows,
        publication_bytes=config.publication_bytes,
    )


def _create_owner(config: Any, owner: str, node: str) -> Any:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return (
        _ray()
        .remote(num_cpus=0)(_ReplayOwner)
        .options(
            name=f"replay-{config.run_id}-{owner}",
            namespace=_NAMESPACE,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
        )
        .remote(config, owner, node)
    )


class _ReplayOwner:
    """Ray endpoint for one node's existing local body store."""

    def __init__(self, config: Any, owner: str, node: str) -> None:
        self.store = _local_store(config)
        if self.store.owner != owner or _ray().get_runtime_context().get_node_id() != node:
            raise RuntimeError("Replay owner was started on a different node or mount namespace")
        self.store.start_run()
        self.transfer_bytes = config.publication_bytes

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        # Descriptors may locate only registered replay operations, never an
        # arbitrary method, filesystem path or network endpoint.
        allowed = {
            "acquire_many",
            "release_many",
            "validate_partition",
            "attach",
            "commit",
            "discard",
            "retire_partition",
            "close_run",
            "reclaim_stopped_publishers",
        }
        if operation not in allowed:
            raise ValueError("Unknown replay ownership operation")
        return getattr(self.store, operation)(*args, **kwargs)

    def ready(self) -> str:
        return self.store.owner

    def read(self, ref: ReplayRef, lease: str, layer: int, rows: Any) -> Any:
        import numpy as np

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) * ref.topk * 4 > self.transfer_bytes:
            raise ValueError("Remote replay read exceeds the bounded transfer size")
        with self.store._state(write=False) as state:
            if lease not in self.store._object(state, ref)["readers"]:
                raise RuntimeError("Remote replay read requires an active reader lease")
        output = np.empty((len(rows), ref.topk), dtype=np.int32)
        self.store._read_into(ref, layer, rows, output, np.arange(len(rows)))
        return output


class _ReplayRegistry:
    """Run ownership and partition tombstones, independent of training mode."""

    def __init__(self, config: Any, driver: str) -> None:
        self.config, self.driver = config, driver
        self.owners: dict[str, Any] = {}
        self.retired: set[str] = set()
        self.closing = False

    def register(self, owner: str, node: str) -> Any:
        if self.closing:
            raise RuntimeError("Replay run is closing")
        if owner in self.owners:
            # Several Ray nodes may share the same kernel/mount namespace.
            # They share a body owner too; retain its verified endpoint.
            _call(self.owners[owner], "ready")
            return self.owners[owner]
        ray = _ray()
        actor = _create_owner(self.config, owner, node)
        try:
            if _call(actor, "ready") != owner:
                raise RuntimeError("Replay owner registration returned another identity")
            # Registration and retirement are serialized by this actor. A late
            # publishing node cannot reopen a partition that already retired.
            for partition in self.retired:
                _call(actor, "invoke", "retire_partition", partition)
        except BaseException:
            ray.kill(actor, no_restart=True)
            raise
        self.owners[owner] = actor
        return actor

    def locate(self, owner: str) -> Any:
        if owner not in self.owners:
            raise RuntimeError("Replay reference names an unregistered owner")
        return self.owners[owner]

    def attach(self, refs: list[ReplayRef], partition: str) -> None:
        if self.closing or partition in self.retired:
            raise RuntimeError("Cannot attach replay to a closed partition/run")
        groups = _group(refs)
        for owner in groups:
            self.locate(owner)
        # All owners stay registered even if a reply is lost after its hold
        # was written. A later partition retirement reaches every one.
        for owner, indices in groups.items():
            _call(self.owners[owner], "invoke", "attach", [refs[i] for i in indices], partition)

    def retire_partition(self, partition: str) -> None:
        self.retired.add(partition)
        error = None
        # Include owners with no attached body so late/stale reader requests
        # have the same tombstone on every registered node.
        for actor in self.owners.values():
            try:
                _call(actor, "invoke", "retire_partition", partition)
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def reclaim_stopped_publishers(self) -> int:
        remaining, error = 0, None
        for actor in self.owners.values():
            try:
                remaining += _call(actor, "invoke", "reclaim_stopped_publishers")
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error
        return remaining

    def close(self, driver: str) -> None:
        if driver != self.driver:
            raise RuntimeError("Only the owning Relax driver may close this replay run")
        self.closing = True
        error = None
        for actor in self.owners.values():
            try:
                _call(actor, "invoke", "close_run")
            except Exception as exc:
                error = error or exc
        if error is not None:
            # Keep actors and their leases reachable for evidence and retry.
            raise error
        for actor in self.owners.values():
            _ray().kill(actor, no_restart=True)


def _group(refs: list[ReplayRef]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, ref in enumerate(refs):
        result[ref.owner].append(index)
    return result


class RoutedReplayStore:
    """Use local files directly or route a reference to its registered
    owner."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.local = _local_store(config)
        self._registry_handle = None
        self._owners: dict[str, Any] = {}
        self._registered = False
        self._driver = None
        self._pending_releases: list[tuple[list[ReplayRef], list[str]]] = []

    def _registry(self) -> Any:
        if self._registry_handle is None:
            self._registry_handle = _ray().get_actor(f"replay-{self.config.run_id}", namespace=_NAMESPACE)
        return self._registry_handle

    def start_run(self) -> None:
        ray = _ray()
        driver = uuid.uuid4().hex
        # No get_if_exists: a second driver must never adopt an existing run.
        registry = (
            ray.remote(num_cpus=0)(_ReplayRegistry)
            .options(
                name=f"replay-{self.config.run_id}",
                namespace=_NAMESPACE,
            )
            .remote(self.config, driver)
        )
        self._registry_handle = registry
        try:
            self.register_local_owner()
        except BaseException:
            ray.kill(registry, no_restart=True)
            self._registry_handle = None
            raise
        self._driver = driver

    def register_local_owner(self) -> str:
        if not self._registered:
            node = _ray().get_runtime_context().get_node_id()
            self._owners[self.local.owner] = _call(self._registry(), "register", self.local.owner, node)
            self._registered = True
        return self.local.owner

    def begin_publication(self, **kwargs: Any) -> Any:
        # Check the registered local endpoint first. This also avoids touching
        # an inherited Ray client if a CUDA scheduler was created with fork.
        with self.local._state(write=False) as state:
            driver = state.get("driver")
            registered = driver and _process_start(driver["pid"]) == driver["start"] and not state["closed"]
        if not registered:
            import ray

            if not ray.is_initialized():
                raise RuntimeError("Replay publisher's parent has not registered this node")
            self._registered = False
            self.register_local_owner()
        return self.local.begin_publication(**kwargs)

    def _owner(self, owner: str) -> Any:
        if owner not in self._owners:
            self._owners[owner] = _call(self._registry(), "locate", owner)
        return self._owners[owner]

    def _invoke(self, owner: str, operation: str, *args: Any, **kwargs: Any) -> Any:
        if owner == self.local.owner:
            return getattr(self.local, operation)(*args, **kwargs)
        return _call(self._owner(owner), "invoke", operation, *args, **kwargs)

    def _check(self, refs: list[ReplayRef]) -> None:
        if any(ref.run != self.config.run_id for ref in refs):
            raise RuntimeError("Replay reference belongs to another run")

    def attach(self, refs: list[ReplayRef], partition: str) -> None:
        self._check(refs)
        _call(self._registry(), "attach", refs, partition)

    def _apply(self, operation: str, refs: list[ReplayRef], *args: Any) -> None:
        self._check(refs)
        error = None
        for owner, indices in _group(refs).items():
            try:
                self._invoke(owner, operation, [refs[i] for i in indices], *args)
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def commit(self, refs: list[ReplayRef]) -> None:
        self._apply("commit", refs)

    def discard(self, refs: list[ReplayRef]) -> None:
        self._apply("discard", refs)

    def validate_partition(self, refs: list[ReplayRef], partition: str) -> None:
        self._apply("validate_partition", refs, partition)

    def acquire_many(self, refs: list[ReplayRef], *, partition: str | None = None) -> list[str]:
        self._check(refs)
        leases = [uuid.uuid4().hex for _ in refs]
        attempted = []
        try:
            for owner, indices in _group(refs).items():
                group, identities = [refs[i] for i in indices], [leases[i] for i in indices]
                if owner != self.local.owner:
                    self._owner(owner)
                attempted.append((group, identities))
                self._invoke(owner, "acquire_many", group, partition=partition, leases=identities)
        except BaseException as primary:
            for group, identities in attempted:
                try:
                    self.release_many(group, identities)
                except Exception as cleanup:
                    self._pending_releases.append((group, identities))
                    primary.add_note(f"Replay acquire rollback remains pending: {cleanup!r}")
            raise
        return leases

    def release_many(self, refs: list[ReplayRef], leases: list[str]) -> None:
        self._check(refs)
        if len(refs) != len(leases):
            raise ValueError("Replay release requires one lease per reference")
        error = None
        for owner, indices in _group(refs).items():
            try:
                self._invoke(owner, "release_many", [refs[i] for i in indices], [leases[i] for i in indices])
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    @contextmanager
    def reader(self, refs: list[ReplayRef], *, partition: str | None = None) -> Iterator[ConsumerReadSession]:
        session = ConsumerReadSession(self, refs, partition=partition)
        try:
            yield session
        except BaseException as primary:
            try:
                session.release()
            except Exception as cleanup:
                raise primary from cleanup
            raise
        else:
            session.release()

    def _read_into(
        self, ref: ReplayRef, layer: int, rows: Any, output: Any, destinations: Any, lease: str | None = None
    ) -> None:
        self._check([ref])
        if ref.owner == self.local.owner:
            self.local._read_into(ref, layer, rows, output, destinations)
            return
        import numpy as np

        rows, destinations = np.asarray(rows, dtype=np.int64), np.asarray(destinations, dtype=np.int64)
        if (
            rows.ndim != 1
            or destinations.shape != rows.shape
            or output.ndim != 2
            or output.shape[1] != ref.topk
            or output.dtype not in (np.int32, np.int64)
            or not output.flags.writeable
            or np.any(destinations < 0)
            or np.any(destinations >= len(output))
            or not lease
        ):
            raise ValueError("Invalid remote replay read layout or lease")
        count = max(1, self.config.publication_bytes // (ref.topk * 4))
        for start in range(0, len(rows), count):
            selected = rows[start : start + count]
            target = destinations[start : start + count]
            values = _call(self._owner(ref.owner), "read", ref, lease, layer, selected)
            if values.dtype != np.int32 or values.shape != (len(selected), ref.topk):
                raise RuntimeError("Remote replay owner returned a different layer shape or dtype")
            boundaries = np.r_[0, np.flatnonzero(np.diff(target) != 1) + 1, len(target)]
            for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
                dst = int(target[first])
                np.copyto(output[dst : dst + int(last - first)], values[first:last], casting="safe")

    def retire_partition(self, partition: str) -> None:
        _call(self._registry(), "retire_partition", partition)

    def reclaim_stopped_publishers(self) -> int:
        return _call(self._registry(), "reclaim_stopped_publishers")

    def close_run(self) -> None:
        if self._driver is None:
            raise RuntimeError("This process does not own the replay run")
        pending, self._pending_releases = self._pending_releases, []
        for group, leases in pending:
            try:
                self.release_many(group, leases)
            except Exception:
                self._pending_releases.append((group, leases))
        if self._pending_releases:
            raise RuntimeError("Replay reader rollback is still incomplete")
        _call(self._registry(), "close", self._driver)
        _ray().kill(self._registry(), no_restart=True)
        self._driver = None
