# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV event block-size discovery across vLLM releases.

Some vLLM releases expose KV cache group metadata as an engine-core utility and
some do not, so Dynamo reads it through its own worker extension. These tests
use stand-in engine objects and do not need vLLM installed.
"""

import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
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
