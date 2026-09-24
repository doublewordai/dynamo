# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "dynamo.llm", types.SimpleNamespace(KvRouter=Mock())
    )
    return load_module("test_generation_router", "generations.py")


def make_router(module):
    return module.GenerationRouter(
        None, "pool.backend.generate", 2, None, None, "/unused"
    )


def generation(module, name, ids, cost):
    backend = Mock()
    backend.preview_request = AsyncMock(
        return_value=None if cost is None else (ids[0], 0, cost)
    )
    return module.Generation(name, Mock(instance_ids=lambda: ids), backend)


@pytest.mark.asyncio
async def test_compare_native_kv_and_load_costs(module):
    router = make_router(module)
    old = generation(module, "old", [1, 2, 3], 0)
    new = generation(module, "new", [4], 8)
    router.generations = {"old": old, "new": new}
    request = {"token_ids": [1, 2], "routing": {"cache_salt": "tenant"}}
    assert await router._choose(request) is old  # warm prefix wins
    old.router.preview_request.return_value = (1, 0, 12)
    assert await router._choose(request) is new  # old load outweighs overlap
    new.router.preview_request.assert_awaited_with(request)


@pytest.mark.asyncio
async def test_equal_cost_weights_follow_membership(module, monkeypatch):
    router = make_router(module)
    old = generation(module, "old", [1, 2, 3], 2)
    new = generation(module, "new", [4], 2)
    router.generations = {"old": old, "new": new}
    choices = Mock(return_value=[old])
    monkeypatch.setattr(module.random, "choices", choices)
    await router._choose({})
    choices.assert_called_once_with([old, new], weights=[3, 1], k=1)


@pytest.mark.asyncio
async def test_pins_and_allowlists_do_not_cross_generations(module):
    router = make_router(module)
    old = generation(module, "old", [1, 2], 99)
    new = generation(module, "new", [3], 0)
    router.generations = {"old": old, "new": new}
    assert await router._choose({"routing": {"backend_instance_id": 1}}) is old
    assert await router._choose({"routing": {"allowed_worker_ids": [2]}}) is old
    new.router.preview_request.assert_not_awaited()
    with pytest.raises(RuntimeError, match="No eligible"):
        await router._choose({"routing": {"backend_instance_id": 4}})


@pytest.mark.asyncio
async def test_dead_or_failed_generation_does_not_hide_survivor(module):
    router = make_router(module)
    old = generation(module, "old", [1], 1)
    new = generation(module, "new", [2], 0)
    router.generations = {"old": old, "new": new}
    new.router.preview_request.side_effect = RuntimeError("removed during preview")
    assert await router._choose({}) is old
    new.client.instance_ids = list
    assert await router._choose({}) is old


@pytest.mark.asyncio
async def test_overload_uses_normal_admission_and_invalid_request_surfaces(module):
    router = make_router(module)
    old = generation(module, "old", [1], None)
    router.generations = {"old": old}
    assert await router._choose({}) is old
    old.router.preview_request.side_effect = ValueError("bad routing constraint")
    with pytest.raises(ValueError, match="bad routing"):
        await router._choose({})


@pytest.mark.asyncio
async def test_removed_generation_finishes_existing_stream(module):
    router = make_router(module)
    old = generation(module, "old", [1], 0)
    router.generations = {"old": old}

    async def output():
        yield {"token_ids": [1]}
        yield {"token_ids": [2]}

    old.router.generate_from_request = AsyncMock(return_value=output())
    stream = await router.generate_from_request({"token_ids": [1]})
    assert await anext(stream) == {"token_ids": [1]}
    router.generations = {}
    assert await anext(stream) == {"token_ids": [2]}
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_atomic_update_preserves_warm_router_and_rejects_invalid_snapshot(
    module, tmp_path
):
    router = make_router(module)
    old = generation(module, "pool-old.backend.generate", [1], 0)
    router.generations = {old.endpoint: old}
    router.path = tmp_path / "generations.json"
    router.path.write_text(
        json.dumps({"block_size": 2, "endpoints": [old.endpoint, old.endpoint]})
    )
    await router.refresh()
    assert list(router.generations.values()) == [old]
    router.path.write_text('{"endpoints":')
    with pytest.raises(ValueError):
        await router.refresh()
    assert list(router.generations.values()) == [old]
    router.path.write_text(json.dumps({"block_size": 2, "endpoints": []}))
    await router.refresh()
    assert router.generations == {}


@pytest.mark.parametrize(
    "snapshot",
    [
        [],
        {"block_size": 4, "endpoints": []},
        {"block_size": 2, "endpoints": ["unrelated-new.backend.generate"]},
        {"block_size": 2, "endpoints": ["pool-new.prefill.generate"]},
        {"block_size": 2, "endpoints": "pool-new.backend.generate"},
    ],
)
def test_reject_incomparable_or_unowned_generations(module, tmp_path, snapshot):
    path = tmp_path / "generations.json"
    path.write_text(json.dumps(snapshot))
    with pytest.raises((ValueError, TypeError)):
        module.read_endpoints(path, "pool.backend.generate", 2)


def test_supervisor_retains_old_ready_generation():
    supervisor = load_module("test_generation_supervisor", "supervisor.py")
    graph = {"status": {"components": {"Worker": {"componentNames": ["old", "new"]}}}}

    def deployment(name, ready):
        return {
            "metadata": {
                "name": name,
                "labels": {
                    "nvidia.com/dynamo-namespace": "pool",
                    "nvidia.com/dynamo-worker-hash": name,
                },
            },
            "status": {"readyReplicas": ready},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "env": [
                                    {"name": "DYN_NAMESPACE", "value": "pool"},
                                    {
                                        "name": "DYN_NAMESPACE_WORKER_SUFFIX",
                                        "value": name,
                                    },
                                ]
                            }
                        ]
                    }
                }
            },
        }

    old, new = deployment("old", 3), deployment("new", 1)
    assert supervisor.ready_runtime_namespaces(graph, "Worker", "pool", [new, old]) == [
        "pool-new",
        "pool-old",
    ]
    old["status"]["readyReplicas"] = 0
    assert supervisor.ready_runtime_namespaces(graph, "Worker", "pool", [new, old]) == [
        "pool-new"
    ]


@pytest.mark.asyncio
async def test_watcher_retries_generic_binding_failure(module, monkeypatch):
    import asyncio

    router = make_router(module)
    old = generation(module, "old", [1], 0)
    router.generations = {"old": old}
    router.refresh = AsyncMock(
        side_effect=[
            Exception("native discovery unavailable"),
            asyncio.CancelledError(),
        ]
    )
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await router._watch()
    assert router.refresh.await_count == 2
    assert router.generations == {"old": old}
