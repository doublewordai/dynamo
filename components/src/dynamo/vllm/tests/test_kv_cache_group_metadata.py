# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV event block-size discovery across vLLM releases.

Some vLLM releases expose KV cache group metadata as an engine-core utility and
some do not, so Dynamo reads it through its own worker extension. These tests
use stand-in engine objects and do not need vLLM installed.
"""

import importlib
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    KvCacheGroupMetadataError,
    configure_kv_event_block_size,
    validate_worker_group_metadata,
)
from dynamo.vllm.worker_extension import (
    DYNAMO_WORKER_EXTENSION_CLS,
    KV_CACHE_GROUP_METADATA_METHOD,
    DynamoWorkerExtension,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

HYBRID_GROUPS = [
    {"group_idx": 0, "kind": "sliding_window_mla", "block_size": 64},
    {"group_idx": 1, "kind": "mla_attention", "block_size": 256},
]


class SubclassedExtension(DynamoWorkerExtension):
    pass


def _vllm_config(worker_extension_cls="", dcp_size=1, block_size=64):
    return SimpleNamespace(
        additional_config=None,
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            worker_extension_cls=worker_extension_cls,
            decode_context_parallel_size=dcp_size,
        ),
    )


class _Engine:
    """Stand-in for AsyncLLM recording which metadata path was used."""

    def __init__(self, worker_groups=None, utility_groups=None):
        self.calls: list[str] = []
        self._worker_groups = worker_groups
        self._utility_groups = utility_groups
        self.engine_core = SimpleNamespace(call_utility_async=self._call_utility)

    async def collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"collective_rpc:{method}")
        if self._worker_groups is None:
            raise RuntimeError("worker has no such method")
        # One result per worker rank.
        return [self._worker_groups, self._worker_groups]

    async def _call_utility(self, method, *args):
        self.calls.append(f"utility:{method}")
        if self._utility_groups is None:
            raise RuntimeError(f"Call to {method} method failed")
        return self._utility_groups


async def test_worker_extension_supplies_main_attention_block_size():
    """Release without the engine-core utility: the worker extension answers."""
    engine = _Engine(worker_groups=HYBRID_GROUPS)
    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS)

    assert await configure_kv_event_block_size(engine, vllm_config) == 256
    assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 256
    assert engine.calls == [f"collective_rpc:{KV_CACHE_GROUP_METADATA_METHOD}"]


async def test_worker_extension_block_size_spans_decode_context_parallel_ranks():
    engine = _Engine(worker_groups=HYBRID_GROUPS)
    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS, dcp_size=2)

    assert await configure_kv_event_block_size(engine, vllm_config) == 512
    # The worker's answer is not mutated.
    assert HYBRID_GROUPS[1]["block_size"] == 256


async def test_subclassed_worker_extension_is_recognised():
    engine = _Engine(worker_groups=HYBRID_GROUPS)
    vllm_config = _vllm_config(f"{__name__}.SubclassedExtension")

    assert await configure_kv_event_block_size(engine, vllm_config) == 256


async def test_foreign_worker_extension_uses_engine_core_utility():
    """Extension slot taken by the user, on a release that has the utility."""
    engine = _Engine(utility_groups=HYBRID_GROUPS)
    vllm_config = _vllm_config("some.other.Extension")

    assert await configure_kv_event_block_size(engine, vllm_config) == 256
    assert engine.calls == ["utility:get_kv_cache_group_metadata"]


async def test_no_metadata_source_falls_back_to_cache_config():
    engine = _Engine()
    vllm_config = _vllm_config("some.other.Extension", block_size=64)

    assert await configure_kv_event_block_size(engine, vllm_config) == 64
    assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 64


async def test_worker_rpc_failure_falls_back_to_cache_config():
    engine = _Engine()
    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS, block_size=16)

    assert await configure_kv_event_block_size(engine, vllm_config) == 16


class _Kind(str, Enum):
    MLA = "mla_attention"
    SWA_MLA = "sliding_window_mla"
    FULL = "full_attention"


@pytest.fixture
def fake_kv_cache_interface(monkeypatch):
    """Stand-in for vllm.v1.kv_cache_interface, keyed on a spec's ``kind``."""
    module = ModuleType("vllm.v1.kv_cache_interface")
    module.get_kv_cache_spec_kind = lambda spec: spec.kind
    for name in ("vllm", "vllm.v1"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", module)
    return module


def _worker(groups):
    worker = DynamoWorkerExtension()
    worker.model_runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in groups]
        )
    )
    return worker


def test_worker_extension_reports_plain_serializable_groups(fake_kv_cache_interface):
    worker = _worker(
        [
            SimpleNamespace(kind=_Kind.SWA_MLA, block_size=64),
            SimpleNamespace(kind=_Kind.MLA, block_size=256),
        ]
    )

    metadata = worker.dynamo_get_kv_cache_group_metadata()

    assert metadata == HYBRID_GROUPS
    for group in metadata:
        assert {type(v) for v in group.values()} <= {int, str}


def test_worker_extension_uses_first_layer_spec_of_uniform_group(
    fake_kv_cache_interface,
):
    merged = SimpleNamespace(
        kind=_Kind.FULL,
        block_size=16,
        kv_cache_specs={
            "layers.0": SimpleNamespace(kind=_Kind.FULL, block_size=32),
            "layers.1": SimpleNamespace(kind=_Kind.FULL, block_size=32),
        },
    )

    assert _worker([merged]).dynamo_get_kv_cache_group_metadata() == [
        {"group_idx": 0, "kind": "full_attention", "block_size": 32}
    ]


def test_worker_extension_without_kv_cache_config_reports_no_groups(
    fake_kv_cache_interface,
):
    worker = DynamoWorkerExtension()
    worker.model_runner = SimpleNamespace()

    assert worker.dynamo_get_kv_cache_group_metadata() == []


def test_worker_extension_adds_only_prefixed_public_names():
    """vLLM asserts the extension shares no attribute with its worker class."""
    names = [n for n in dir(DynamoWorkerExtension) if not n.startswith("__")]
    assert names == [KV_CACHE_GROUP_METADATA_METHOD]


# ---------------------------------------------------------------------------
# Strict validation of the worker result, and a loud fallback
# ---------------------------------------------------------------------------

_GOOD = {"group_idx": 0, "kind": "mla_attention", "block_size": 256}


@pytest.mark.parametrize(
    "rank_results",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="no-ranks"),
        pytest.param({"0": [_GOOD]}, id="not-a-list"),
        pytest.param([None], id="rank-returned-none"),
        pytest.param(["mla_attention"], id="rank-returned-str"),
        pytest.param([[_GOOD], []], id="ranks-disagree-empty"),
        pytest.param([[_GOOD], [{**_GOOD, "block_size": 64}]], id="ranks-disagree"),
        pytest.param([[["mla_attention", 256]]], id="group-not-a-dict"),
        pytest.param([[{"kind": "mla_attention", "block_size": 256}]], id="no-idx"),
        pytest.param([[{**_GOOD, "group_idx": 1}]], id="idx-not-position"),
        pytest.param([[{**_GOOD, "group_idx": "0"}]], id="idx-str"),
        pytest.param([[{"group_idx": 0, "block_size": 256}]], id="no-kind"),
        pytest.param([[{**_GOOD, "kind": ""}]], id="kind-empty"),
        pytest.param([[{**_GOOD, "kind": 3}]], id="kind-int"),
        pytest.param([[{"group_idx": 0, "kind": "mla_attention"}]], id="no-size"),
        pytest.param([[{**_GOOD, "block_size": "256"}]], id="size-str"),
        pytest.param([[{**_GOOD, "block_size": 256.0}]], id="size-float"),
        pytest.param([[{**_GOOD, "block_size": True}]], id="size-bool"),
        pytest.param([[{**_GOOD, "block_size": 0}]], id="size-zero"),
        pytest.param([[{**_GOOD, "block_size": -16}]], id="size-negative"),
    ],
)
def test_malformed_worker_results_are_rejected(rank_results):
    with pytest.raises(KvCacheGroupMetadataError):
        validate_worker_group_metadata(rank_results)


def test_valid_worker_results_return_a_copy_of_rank_zero():
    rank_results = [list(HYBRID_GROUPS), list(HYBRID_GROUPS)]

    validated = validate_worker_group_metadata(rank_results)

    assert validated == HYBRID_GROUPS
    assert validated[0] is not HYBRID_GROUPS[0]


def test_worker_without_kv_cache_groups_is_valid():
    assert validate_worker_group_metadata([[], []]) == []


async def test_malformed_worker_result_falls_back_with_greppable_error(caplog):
    engine = _Engine(worker_groups=[{**_GOOD, "block_size": "256"}])
    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS, block_size=64)

    with caplog.at_level(logging.ERROR, logger="dynamo.vllm.cache_info"):
        assert await configure_kv_event_block_size(engine, vllm_config) == 64

    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    message = record.getMessage()
    assert "KV_EVENT_BLOCK_SIZE_UNVERIFIED" in message
    assert "KvCacheGroupMetadataError" in message
    assert "hybrid/MLA" in message
    assert record.exc_info is not None


async def test_worker_rpc_failure_logs_greppable_error(caplog):
    engine = _Engine()
    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS, block_size=16)

    with caplog.at_level(logging.ERROR, logger="dynamo.vllm.cache_info"):
        assert await configure_kv_event_block_size(engine, vllm_config) == 16

    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "KV_EVENT_BLOCK_SIZE_UNVERIFIED" in record.getMessage()
    assert "worker has no such method" in record.getMessage()


async def test_successful_fetch_logs_no_error(caplog):
    engine = _Engine(worker_groups=HYBRID_GROUPS)
    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS)

    with caplog.at_level(logging.ERROR, logger="dynamo.vllm.cache_info"):
        await configure_kv_event_block_size(engine, vllm_config)

    assert not caplog.records


# ---------------------------------------------------------------------------
# The extension loaded the way vLLM loads it
#
# No vLLM >= 0.28 is importable where these tests were written, so the stand-ins
# below are transcribed from vLLM v0.30.0: the dataclass field layout of
# vllm/v1/kv_cache_interface.py (KVCacheSpec, AttentionSpec, FullAttentionSpec,
# MLAAttentionSpec, SlidingWindowSpec, SlidingWindowMLASpec,
# UniformTypeKVCacheSpecs, KVCacheGroupSpec, KVCacheConfig), the isinstance
# ladder of get_kv_cache_spec_kind, and the mixin injection performed by
# WorkerWrapperBase.init_worker in vllm/v1/worker/worker_base.py.
# ---------------------------------------------------------------------------


@dataclass
class _KVCacheSpec:
    block_size: int


@dataclass
class _AttentionSpec(_KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: str  # torch.dtype in vLLM: not msgpack-serializable on purpose


@dataclass
class _FullAttentionSpec(_AttentionSpec):
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


@dataclass
class _MLAAttentionSpec(_FullAttentionSpec):
    cache_dtype_str: str | None = None
    storage_block_size: int | None = None


@dataclass
class _SlidingWindowSpec(_AttentionSpec):
    sliding_window: int = 0


@dataclass
class _SlidingWindowMLASpec(_SlidingWindowSpec):
    cache_dtype_str: str | None = None


@dataclass
class _UniformTypeKVCacheSpecs(_KVCacheSpec):
    kv_cache_specs: dict = field(default_factory=dict)


@dataclass
class _KVCacheGroupSpec:
    layer_names: list
    kv_cache_spec: _KVCacheSpec
    is_eagle_group: bool = False
    host_resident: bool = False
    enable_kv_transfer: bool = True


@dataclass
class _KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list
    kv_cache_layout: str | None = None


class _SpecKind(str, Enum):
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    UNKNOWN = "unknown"


def _get_kv_cache_spec_kind(spec):
    if isinstance(spec, _UniformTypeKVCacheSpecs):
        kinds = {_get_kv_cache_spec_kind(s) for s in spec.kv_cache_specs.values()}
        return next(iter(kinds)) if len(kinds) == 1 else _SpecKind.UNKNOWN
    if isinstance(spec, _SlidingWindowMLASpec):
        return _SpecKind.SLIDING_WINDOW_MLA
    if isinstance(spec, _MLAAttentionSpec):
        return _SpecKind.MLA_ATTENTION
    if isinstance(spec, _FullAttentionSpec):
        return _SpecKind.FULL_ATTENTION
    if isinstance(spec, _SlidingWindowSpec):
        return _SpecKind.SLIDING_WINDOW
    return _SpecKind.UNKNOWN


def _inject_worker_extension(worker_class, qualname):
    """WorkerWrapperBase.init_worker's handling of worker_extension_cls."""
    module_name, obj_name = qualname.rsplit(".", 1)
    extension_cls = getattr(importlib.import_module(module_name), obj_name)
    extended_calls = []
    if extension_cls not in worker_class.__bases__:
        for attr in dir(extension_cls):
            if attr.startswith("__"):
                continue
            assert not hasattr(worker_class, attr), attr
            if callable(getattr(extension_cls, attr)):
                extended_calls.append(attr)
        worker_class.__bases__ = worker_class.__bases__ + (extension_cls,)
    return extended_calls


def _run_method(worker, method, args=(), kwargs=None):
    """vLLM's run_method for a string method name, as collective_rpc uses it."""
    return getattr(worker, method)(*args, **(kwargs or {}))


def _hybrid_mla_kv_cache_config():
    mla = dict(num_kv_heads=1, head_size=576, dtype="bfloat16")
    return _KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[
            _KVCacheGroupSpec(
                ["model.layers.0.attn"],
                _SlidingWindowMLASpec(block_size=64, sliding_window=128, **mla),
            ),
            _KVCacheGroupSpec(
                ["model.layers.1.attn", "model.layers.2.attn"],
                _UniformTypeKVCacheSpecs(
                    block_size=256,
                    kv_cache_specs={
                        "model.layers.1.attn": _MLAAttentionSpec(block_size=256, **mla),
                        "model.layers.2.attn": _MLAAttentionSpec(block_size=256, **mla),
                    },
                ),
            ),
        ],
    )


@pytest.fixture
def vllm_style_worker(monkeypatch):
    """A stub worker class with the extension mixed in the way vLLM does it."""
    module = ModuleType("vllm.v1.kv_cache_interface")
    module.get_kv_cache_spec_kind = _get_kv_cache_spec_kind
    for name in ("vllm", "vllm.v1"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", module)

    class WorkerBase:
        pass

    class Worker(WorkerBase):
        def __init__(self):
            self.model_runner = SimpleNamespace(
                kv_cache_config=_hybrid_mla_kv_cache_config()
            )

        def get_kv_cache_spec(self):
            return {}

    extended_calls = _inject_worker_extension(Worker, DYNAMO_WORKER_EXTENSION_CLS)
    return Worker, extended_calls


def test_extension_mixes_into_a_vllm_style_worker(vllm_style_worker):
    worker_class, extended_calls = vllm_style_worker

    assert extended_calls == [KV_CACHE_GROUP_METADATA_METHOD]
    assert issubclass(worker_class, DynamoWorkerExtension)
    # A second engine in the same process must not inject twice.
    assert _inject_worker_extension(worker_class, DYNAMO_WORKER_EXTENSION_CLS) == []


async def test_mixed_in_extension_result_passes_validation_end_to_end(
    vllm_style_worker,
):
    """Real-shaped kv_cache_config -> mixin -> msgpack wire -> validation."""
    import msgspec

    worker_class, _ = vllm_style_worker
    workers = [worker_class(), worker_class()]

    class Engine:
        engine_core = None

        async def collective_rpc(self, method, *args, **kwargs):
            results = [_run_method(worker, method) for worker in workers]
            # The utility channel carries plain msgpack with no custom types.
            return msgspec.msgpack.decode(msgspec.msgpack.encode(results))

    vllm_config = _vllm_config(DYNAMO_WORKER_EXTENSION_CLS, block_size=64)

    assert await configure_kv_event_block_size(Engine(), vllm_config) == 256
