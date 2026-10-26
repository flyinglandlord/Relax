# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang/rollout integration: request, capture, response and publication."""

import ast
import importlib
import importlib.util
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pybase64

from .core import ReplayManager, replay_data_manager
from .data import FIELDS, REFERENCE_FIELD, ReplayRef, validate_sample


_CAPTURE_FIELDS = {"moe": "routed_experts", "indexer": "indexer_topk"}


@dataclass(frozen=True)
class Producer:
    """Rollout-facing adapter: request capture, accept references, build TQ
    fields.

    In reference mode SGLang writes bodies directly through the store; TQ only
    carries references. Full mode preserves the original tensor wire format.
    """

    manager: "ReplayManager"
    kinds: frozenset[str]

    def request_options(self, *, evaluation: bool = False) -> dict[str, bool]:
        self.manager._ensure_open()
        options = {}
        if "moe" in self.kinds and not (evaluation and self.manager.uses_references):
            options["return_routed_experts"] = True
        if "indexer" in self.kinds and not evaluation:
            options["return_indexer_topk"] = True
        return options

    def accept_response(self, sample: Any, meta: dict, args: Any, *, evaluation: bool = False) -> None:
        self.manager._ensure_open()
        if self.manager.uses_references:
            if evaluation or not self.kinds:
                return
            refs = {}
            for kind in self.kinds:
                field = _CAPTURE_FIELDS[kind]
                value = meta[field]
                if not isinstance(value, dict):
                    # External/unpatched SGLang keeps its original HTTP payload.
                    # Publish it at ingress; TQ still carries only a reference.
                    values = (
                        decode_routed_experts(meta, len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
                        if kind == "moe"
                        else decode_indexer_topk(meta, len(sample.tokens) - 1)
                    )
                    if kind == "indexer":
                        topk = getattr(args, "dsa_indexer_topk", None)
                        if topk is None:
                            from relax.utils.misc import get_hf_config

                            config = get_hf_config(args.hf_checkpoint)
                            topk = getattr(config, "text_config", config).index_topk
                        values = values.reshape(len(sample.tokens) - 1, -1, topk)
                    value = self._publish_response(values, kind=kind, tokens=sample.tokens)
                    # The enclosing response guard now owns cleanup on failure.
                    meta[field] = value
                refs[kind] = value
            validate_sample(refs, sample.tokens)
            sample.rollout_replay_refs = refs
        else:
            if "routed_experts" in meta:
                sample.rollout_routed_experts = decode_routed_experts(
                    meta, len(sample.tokens) - 1, args.num_layers, args.moe_router_topk
                )
            if "indexer" in self.kinds and not evaluation:
                sample.rollout_indexer_topk = decode_indexer_topk(meta, len(sample.tokens) - 1)

    def _publish_response(self, values: np.ndarray, *, kind: str, tokens: list[int]) -> dict:
        """Adapt an existing tensor response without changing the producer
        protocol."""
        publication = self.manager.store().begin_publication(kind=kind, layers=values.shape[1], topk=values.shape[2])
        try:
            for start in range(0, len(values), publication.chunk_rows):
                stop = min(start + publication.chunk_rows, len(values))
                publication.append(
                    stop - start,
                    lambda layer: np.array(values[start:stop, layer], copy=True, order="C"),
                )
            return publication.finish(tokens).to_dict()
        except BaseException as error:
            try:
                publication.abort()
            except Exception as cleanup:
                error.add_note(f"Replay ingress cleanup failed: {cleanup!r}")
            raise

    @contextmanager
    def response(self, meta: dict, *, evaluation: bool = False):
        """Hold response ownership until parsing succeeds; failed responses
        never reach TQ."""
        try:
            yield
        except BaseException as error:
            if self.manager.uses_references and not evaluation:
                for field in _CAPTURE_FIELDS.values():
                    value = meta.get(field)
                    if isinstance(value, dict):
                        try:
                            self.manager.store().discard([ReplayRef.from_dict(value)])
                        except Exception as cleanup:
                            error.add_note(f"Replay response cleanup failed: {cleanup!r}")
            raise

    def build_tq_fields(self, samples: Sequence[Any]) -> dict[str, Any]:
        self.manager._ensure_open()
        if not samples or (self.manager.uses_references and not self.kinds):
            return {}
        fields = frozenset(FIELDS[kind] for kind in self.kinds)
        result = self.manager.build_tq_fields(samples, required_fields=fields)
        if self.manager.uses_references:
            for refs in result[REFERENCE_FIELD]:
                if not self.kinds.issubset(refs):
                    raise ValueError("Registered producer is missing replay kinds")
        return result


def decode_indexer_topk(meta_info: dict[str, Any], num_rows: int) -> np.ndarray:
    """Decode SGLang's shape-less DSV4 indexer payload.

    SGLang returns one int32 row for every input token except the final one.
    The remaining width is ``num_c4_layers * index_topk`` and is intentionally
    kept flat here. Megatron's concrete model config validates and selects the
    C4 columns that are built on this PP rank.
    """
    encoded = meta_info.get("indexer_topk")
    if encoded is None:
        raise RuntimeError(
            "SGLang did not return meta_info['indexer_topk'] while --use-rollout-indexer-replay is enabled."
        )
    if num_rows <= 0:
        raise ValueError(f"indexer replay requires at least one captured row, got {num_rows}")

    values = np.frombuffer(pybase64.b64decode(encoded.encode("ascii")), dtype=np.int32)
    if values.size == 0 or values.size % num_rows != 0:
        raise ValueError(
            "Invalid SGLang indexer_topk payload: "
            f"num_values={values.size} is not a positive multiple of num_rows={num_rows}."
        )
    return values.reshape(num_rows, values.size // num_rows)


def decode_routed_experts(meta_info: dict[str, Any], num_rows: int, num_layers: int, topk: int) -> np.ndarray:
    return np.frombuffer(pybase64.b64decode(meta_info["routed_experts"].encode("ascii")), dtype=np.int32).reshape(
        num_rows, num_layers, topk
    )


def _scheduler_contract() -> tuple[str, str, str]:
    root = Path(importlib.util.find_spec("sglang").origin).parent
    if (root / "srt/managers/scheduler_output_processor_mixin.py").is_file():
        return "sglang.srt.managers.scheduler_output_processor_mixin", "SchedulerOutputProcessorMixin", ""
    return (
        "sglang.srt.managers.scheduler_components.batch_result_processor",
        "SchedulerBatchResultProcessor",
        "_",
    )


def _validate_capture_api(kinds: frozenset[str]) -> None:
    """Check installed source without importing SGLang's CUDA runtime."""
    root = Path(importlib.util.find_spec("sglang").origin).parent
    module, classname, prefix = _scheduler_contract()

    def methods(path, name):
        if not path.is_file():
            return set()
        tree = ast.parse(path.read_text())
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), None)
        return {node.name for node in cls.body if isinstance(node, ast.FunctionDef)} if cls else set()

    collectors = methods(root.joinpath(*module.split(".")[1:]).with_suffix(".py"), classname)
    encoders = methods(root / "srt/managers/detokenizer_manager.py", "DetokenizerManager")
    for kind in kinds:
        field = _CAPTURE_FIELDS[kind]
        if prefix + "maybe_collect_" + field not in collectors or not (
            "_b64_encode_per_request" in encoders or "_extract_" + field in encoders
        ):
            raise RuntimeError(f"Installed SGLang lacks the {kind} replay capture/publication API")


def _abort_request_publication(req: Any) -> None:
    """Drop only this request's unpublished bodies, after all writes drain."""
    error = None
    for publication in getattr(req, "_relax_replay_streams", {}).values():
        try:
            publication.abort()
        except Exception as exc:
            error = error or exc
    if error is not None:
        raise error
    req._relax_replay_streams = {}


def _request_aborted(req: Any) -> bool:
    return any(type(getattr(req, name, None)).__name__ == "FINISH_ABORT" for name in ("finished_reason", "to_finish"))


def _install_abort_patch() -> None:
    """Waiting/chunked requests can disappear without a finished collector."""
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler.abort_request
    if getattr(original, "_relax_replay_cleanup", False):
        return

    @wraps(original)
    def abort(self, recv_req, *args, **kwargs):
        requests = list(self.waiting_queue)
        if getattr(self, "chunked_req", None) is not None:
            requests.append(self.chunked_req)
        for name in ("running_batch", "last_batch"):
            batch = getattr(self, name, None)
            if batch is not None:
                requests.extend(batch.reqs)
        targets = {id(req): req for req in requests if recv_req.abort_all or req.rid.startswith(recv_req.rid)}

        def cleanup():
            error = None
            for req in targets.values():
                try:
                    _abort_request_publication(req)
                except Exception as exc:
                    error = error or exc
            if error is not None:
                raise error

        try:
            result = original(self, recv_req, *args, **kwargs)
        except BaseException as primary:
            try:
                cleanup()
            except Exception as secondary:
                primary.add_note(f"Replay abort cleanup failed: {secondary!r}")
            raise
        cleanup()
        return result

    abort._relax_replay_cleanup = True
    Scheduler.abort_request = abort


def _snapshot_layer(buffer: Any, slots: Any, layer: int) -> np.ndarray:
    """Gather bounded stable rows into one owned chunk before yielding
    scheduler.

    Basic slices avoid a hidden all-layer gather or NumPy advanced-index
    buffer. Background I/O never dereferences a live/recyclable SGLang KV slot.
    """
    output = np.empty((len(slots), buffer.shape[2]), dtype=np.int32)
    boundaries = np.flatnonzero(np.diff(slots) != 1) + 1
    for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(slots)], strict=True):
        src = int(slots[start])
        np.copyto(output[start:end], buffer[src : src + int(end - start), layer], casting="no")
    return output


def _advance_publication(processor: Any, req: Any, target: int, capturers: dict[str, Any], *, final: bool) -> dict:
    """Publish newly stable whole chunks; seal the remaining tail only at
    finish."""
    generation = getattr(req, "retraction_count", 0)
    if getattr(req, "_relax_replay_generation", generation) != generation:
        _abort_request_publication(req)
    req._relax_replay_generation = generation
    if not hasattr(req, "_relax_replay_streams"):
        req._relax_replay_streams = {}
    streams = req._relax_replay_streams
    try:
        for kind, getter in capturers.items():
            field = _CAPTURE_FIELDS[kind]
            if not getattr(req, "return_" + field, False):
                continue
            capturer = getter()
            if capturer is None:
                raise RuntimeError(f"SGLang {kind} capturer is unavailable")
            host = capturer.host_cache if hasattr(capturer, "host_cache") else capturer.get_host_cache()
            buffer = host.buffer
            if buffer.device.type != "cpu" or str(buffer.dtype) != "torch.int32" or buffer.ndim != 3:
                raise RuntimeError("Replay streaming requires completed int32 CPU capture storage")
            if kind not in streams:
                streams[kind] = replay_data_manager.store().begin_publication(
                    kind=kind,
                    layers=buffer.shape[1],
                    topk=buffer.shape[2],
                )
            publication = streams[kind]
            if target < publication.rows:
                raise RuntimeError("Finalized token range overlaps already published replay chunks")
            end = target if final else target // publication.chunk_rows * publication.chunk_rows
            if end > publication.rows:
                # Relax's older async D2H patch defers the CPU slot scatter until
                # the legacy getter. Streaming bypasses that getter, so drain it here.
                flush = getattr(capturer, "_flush_pending_scatter", None)
                if flush is not None:
                    flush()
            while publication.rows < end:
                start = publication.rows
                stop = min(start + publication.chunk_rows, end)
                # Only lightweight slot indices cross D2H here; the capture body
                # already crossed at SGLang's own result-copy boundary.
                slots = processor.req_to_token_pool.req_to_token[req.req_pool_idx][start:stop].cpu().numpy()
                if len(slots) != stop - start or np.any(slots < 0) or np.any(slots >= len(buffer)):
                    raise ValueError("Replay capture slot range is invalid")
                publication.append(stop - start, lambda layer: _snapshot_layer(buffer.numpy(), slots, layer))
        if not final:
            return {}
        output_ids = req.output_ids_through_stop if hasattr(req, "output_ids_through_stop") else req.output_ids
        tokens = list(req.origin_input_ids) + list(output_ids)
        refs = {kind: publication.finish(tokens).to_dict() for kind, publication in streams.items()}
        return refs
    except BaseException as primary:
        try:
            _abort_request_publication(req)
        except Exception as cleanup:
            raise primary from cleanup
        raise


def install_scheduler_patch() -> None:
    """Intercept completed batch copies and finished responses in SGLang only.

    The intermediate layer owns publication. SGLang still owns capture, token
    acceptance, cache slots and scheduling; TQ receives only sealed references.
    """
    module, classname, prefix = _scheduler_contract()
    upstream = importlib.import_module(module)
    cls = getattr(upstream, classname)
    if getattr(cls, "_relax_replay_references", False):
        return
    capturers = {
        kind: getattr(upstream, name)
        for kind, name in (("moe", "get_global_experts_capturer"), ("indexer", "get_global_indexer_capturer"))
        if hasattr(upstream, name)
    }
    collectors = {kind: getattr(cls, prefix + "maybe_collect_" + _CAPTURE_FIELDS[kind]) for kind in capturers}

    def native_request(req):
        # Keep upstream's partial-range and old speculative-stop semantics.
        # These requests use its collector and are published at rollout ingress.
        return bool(getattr(req, "routed_experts_start_len", 0)) or (
            not prefix
            and hasattr(req, "output_ids_through_stop")
            and len(req.output_ids_through_stop) != len(req.output_ids)
        )

    def owner(self):
        if prefix:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
        else:
            parallel = self
        # Match the response-producing rank within each attention-DP replica.
        # Global TP0 would omit every other DP replica's requests.
        rank = getattr(parallel, "attn_tp_rank", parallel.tp_rank)
        return rank == 0 and getattr(parallel, "attn_cp_rank", 0) == 0 and getattr(parallel, "pp_rank", 0) == 0

    def collect(self, req):
        if getattr(req, "_relax_replay_published", False) or getattr(req, "_relax_replay_native_collected", False):
            return
        kinds = {kind for kind, field in _CAPTURE_FIELDS.items() if getattr(req, "return_" + field, False)}
        if not kinds:
            return
        if not kinds <= capturers.keys():
            raise RuntimeError("SGLang lacks a requested replay capturer")
        if _request_aborted(req):
            _abort_request_publication(req)
            return
        if not owner(self):
            for kind in kinds:
                setattr(req, _CAPTURE_FIELDS[kind], None)
            return
        if native_request(req):
            _abort_request_publication(req)
            for kind in kinds:
                collectors[kind](self, req)
            req._relax_replay_native_collected = True
            return
        output_ids = req.output_ids_through_stop if hasattr(req, "output_ids_through_stop") else req.output_ids
        refs = _advance_publication(self, req, len(req.origin_input_ids) + len(output_ids) - 1, capturers, final=True)
        for kind, ref in refs.items():
            setattr(req, _CAPTURE_FIELDS[kind], ref)
        # Ownership now belongs to the response/rollout handoff, not the live Req.
        req._relax_replay_streams = {}
        req._relax_replay_published = True

    def wrap_result(original):
        @wraps(original)
        def process(self, batch, result, *args, **kwargs):
            if not owner(self):
                return original(self, batch, result, *args, **kwargs)
            try:
                # These are the upstream method's existing copy/finalize actions,
                # moved before publication. Clear holders so upstream won't redo.
                if getattr(result, "copy_done", None) is not None:
                    result.copy_done.synchronize()
                for field in ("routed_experts_output", "indexer_topk_output"):
                    capture = getattr(result, field, None)
                    if capture is not None:
                        capture.finalize()
                        setattr(result, field, None)
                for i, req in enumerate(batch.reqs):
                    if _request_aborted(req):
                        _abort_request_publication(req)
                    elif not req.finished() and not req.is_retracted and not native_request(req):
                        # Only the already accepted prefix is stable. A verify
                        # batch may also contain speculative tokens not accepted yet.
                        accepted = len(req.origin_input_ids) + len(req.output_ids)
                        target = min(int(batch.seq_lens_cpu[i]), accepted) - 1
                        _advance_publication(self, req, max(0, target), capturers, final=False)
                return original(self, batch, result, *args, **kwargs)
            except BaseException as primary:
                for req in batch.reqs:
                    try:
                        _abort_request_publication(req)
                    except Exception as cleanup:
                        primary.add_note(f"Replay publication cleanup failed: {cleanup!r}")
                raise

        return process

    for kind in capturers:
        setattr(cls, prefix + "maybe_collect_" + _CAPTURE_FIELDS[kind], collect)
    for method in ("process_batch_result_prefill", "process_batch_result_decode"):
        setattr(cls, method, wrap_result(getattr(cls, method)))
    from sglang.srt.managers.schedule_batch import Req

    _install_abort_patch()
    for method in ("reset_for_retract", "set_finish_with_abort"):
        original = getattr(Req, method)

        def reset(req, *args, _original=original, **kwargs):
            _abort_request_publication(req)
            return _original(req, *args, **kwargs)

        if not getattr(original, "_relax_replay_cleanup", False):
            reset._relax_replay_cleanup = True
            setattr(Req, method, reset)
    cls._relax_replay_references = True


def _encode_responses(values: Any, encode: Any) -> Any:
    """Keep reference metadata and encode native payloads in their original
    order."""
    if values is None or not any(isinstance(value, dict) for value in values):
        return encode(values)
    for value in values:
        if isinstance(value, dict):
            ReplayRef.from_dict(value)
    native = [value for value in values if not isinstance(value, dict)]
    if not native:
        return values
    encoded = iter(encode(native))
    return [value if isinstance(value, dict) else next(encoded) for value in values]


def install_detokenizer_patch() -> None:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

    cls = DetokenizerManager
    if getattr(cls, "_relax_replay_references", False):
        return
    if hasattr(cls, "_b64_encode_per_request"):
        original = cls._b64_encode_per_request

        def encode(values):
            return _encode_responses(values, original)

        cls._b64_encode_per_request = staticmethod(encode)
    else:

        def wrap(original, field):
            def extract(self, recv_obj):
                values = getattr(recv_obj, field)

                def encode(native):
                    response = copy(recv_obj)
                    setattr(response, field, native)
                    return original(self, response)

                return _encode_responses(values, encode)

            return extract

        for field in _CAPTURE_FIELDS.values():
            name = "_extract_" + field
            if hasattr(cls, name):
                setattr(cls, name, wrap(getattr(cls, name), field))
    cls._relax_replay_references = True


def run_reference_scheduler(*args, **kwargs):
    from relax.utils.env import Envs

    if Envs.RELAX_OPTIMIZE_ROUTING_REPLAY:
        from relax.backends.sglang.routing_replay_patch import apply_patch

        apply_patch()
    install_scheduler_patch()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_reference_detokenizer(*args, **kwargs):
    install_detokenizer_patch()
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    return run_detokenizer_process(*args, **kwargs)


def validate_configuration(args: Any, server_args: dict) -> None:
    from .config import rollout_kinds

    kinds = rollout_kinds(args)
    # Algorithm compatibility stays in Relax's existing replay-flag validation.
    # This adapter checks only the installed publication API, never training modes.
    _validate_capture_api(kinds)
