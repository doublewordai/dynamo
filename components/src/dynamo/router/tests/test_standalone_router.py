# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def load_standalone_router_handler():
    placeholder_type = type("Placeholder", (), {})
    stubs = {
        "uvloop": stub_module("uvloop", run=lambda coroutine: coroutine),
        "dynamo": stub_module("dynamo"),
        "dynamo.llm": stub_module(
            "dynamo.llm",
            AicPerfConfig=placeholder_type,
            KvRouter=placeholder_type,
            KvRouterConfig=placeholder_type,
        ),
        "dynamo.router": stub_module("dynamo.router"),
        "dynamo.router.args": stub_module(
            "dynamo.router.args",
            DynamoRouterConfig=placeholder_type,
            build_aic_perf_config=lambda config: config,
            build_kv_router_config=lambda config: config,
            parse_args=lambda argv=None: argv,
        ),
        "dynamo.runtime": stub_module(
            "dynamo.runtime",
            Client=placeholder_type,
            DistributedRuntime=placeholder_type,
            dynamo_worker=lambda: lambda function: function,
        ),
        "dynamo.runtime.logging": stub_module(
            "dynamo.runtime.logging", configure_dynamo_logging=lambda: None
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).parents[1] / "__main__.py"
        spec = importlib.util.spec_from_file_location(
            "standalone_router_main", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


router_module = load_standalone_router_handler()
StandaloneRouterHandler = router_module.StandaloneRouterHandler


def handler_with_router():
    handler = StandaloneRouterHandler.__new__(StandaloneRouterHandler)
    router = AsyncMock()
    handler.kv_router = router
    return handler, router


@pytest.mark.asyncio
async def test_best_worker_id_forwards_cache_namespace() -> None:
    handler, router = handler_with_router()
    router.best_worker.return_value = (7, 0, 3)

    results = [
        worker_id
        async for worker_id in handler.best_worker_id(
            [1, 2, 3, 4],
            {"temperature": 0.0},
            cache_namespace="tenant-a",
        )
    ]

    assert results == [7]
    router.best_worker.assert_awaited_once_with(
        [1, 2, 3, 4],
        {"temperature": 0.0},
        cache_namespace="tenant-a",
    )


@pytest.mark.asyncio
async def test_get_overlap_scores_forwards_cache_namespace() -> None:
    handler, router = handler_with_router()
    router.get_overlap_scores.return_value = {"workers": []}
    request = {
        "token_ids": [1, 2, 3, 4],
        "router_config_override": {"temperature": 0.0},
        "block_mm_infos": None,
        "lora_name": "adapter-a",
        "include_shared": False,
        "cache_namespace": "tenant-a",
    }

    results = [scores async for scores in handler.get_overlap_scores(request)]

    assert results == [{"workers": []}]
    router.get_overlap_scores.assert_awaited_once_with(
        [1, 2, 3, 4],
        {"temperature": 0.0},
        None,
        "adapter-a",
        False,
        "tenant-a",
    )


async def generate_request(handler, router, request):
    async def worker_stream():
        yield {"token_ids": [42], "text": "answer"}

    router.generate_from_request.return_value = worker_stream()
    results = [output async for output in handler.generate(request)]
    router.generate_from_request.assert_awaited_once()
    assert results[0]["token_ids"] == [42]
    return router.generate_from_request.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("require_reasoning", [True, False, None])
async def test_generate_preserves_guided_reasoning(require_reasoning):
    handler, router = handler_with_router()
    request = {
        "token_ids": [1, 2],
        "sampling_options": {
            "guided_decoding": {
                "json": '{"type":"object","properties":{"answer":{"type":"number"}}}'
            }
        },
        "extra_args": {
            "reasoning_parser_kwargs": {"chat_template_kwargs": {"thinking": True}}
        },
    }
    if require_reasoning is not None:
        request["require_reasoning"] = require_reasoning

    forwarded = await generate_request(handler, router, request)

    assert forwarded["require_reasoning"] is (require_reasoning is True)
    assert forwarded["sampling_options"] == request["sampling_options"]
    assert forwarded["extra_args"] == request["extra_args"]


@pytest.mark.asyncio
@pytest.mark.parametrize("disaggregated", [False, True])
async def test_generate_preserves_multimodal_payload_and_routing(disaggregated):
    handler, router = handler_with_router()
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {
            "image": [{"Url": "https://example.com/image.png"}, {"UuidOnly": "cached"}]
        },
        "multi_modal_uuids": {"image": [None, "cached"]},
        "mm_routing_info": {
            "routing_token_ids": [1, 2, 3, 4],
            "block_mm_infos": [None],
            "expanded_prompt_len": 4,
        },
        "media_io_kwargs": {"video": {"fps": 2, "custom_option": "opaque"}},
        "encoder_result": {
            "embedding_handle": {"uri": "nixl://encoder/embedding", "shape": [1, 4]},
            "processed_token_ids": [1, 2, 3, 4],
        },
        "routing": {"dp_rank": 1},
    }
    if disaggregated:
        request["prefill_result"] = {"disaggregated_params": {"opaque": "handoff"}}
        request["bootstrap_info"] = {
            "bootstrap_host": "prefill-worker",
            "bootstrap_port": 12345,
            "bootstrap_room": 7,
        }

    forwarded = await generate_request(handler, router, request)

    for key, value in request.items():
        assert forwarded[key] == value, key
    # Routing may use expanded tokens, but the worker must receive the original input.
    assert forwarded["token_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_generate_preserves_prompt_embeddings_with_empty_tokens():
    handler, router = handler_with_router()
    request = {"token_ids": [], "prompt_embeds": "b3BhcXVlLXRlbnNvci1ieXRlcw=="}

    forwarded = await generate_request(handler, router, request)

    assert forwarded["prompt_embeds"] == request["prompt_embeds"]
    assert forwarded["token_ids"] == []


@pytest.mark.asyncio
async def test_generate_text_only_defaults_and_legacy_dp_rank():
    handler, router = handler_with_router()
    request = {"token_ids": [1, 2], "dp_rank": 3}

    forwarded = await generate_request(handler, router, request)

    assert forwarded["require_reasoning"] is False
    for key in (
        "prompt_embeds",
        "multi_modal_data",
        "multi_modal_uuids",
        "mm_routing_info",
        "media_io_kwargs",
        "encoder_result",
    ):
        assert forwarded.get(key) is None
    assert forwarded["routing"] == {"dp_rank": 3}
    assert request == {"token_ids": [1, 2], "dp_rank": 3}


@pytest.mark.asyncio
async def test_drain_withdraws_before_waiting_for_active_stream():
    import asyncio
    from unittest.mock import Mock

    drain = router_module.RouterDrain()
    release = asyncio.Event()

    async def stream(request):
        yield "first"
        await release.wait()
        yield "last"

    iterator = drain.track(stream)({})
    assert await anext(iterator) == "first"
    endpoints = [
        types.SimpleNamespace(unregister_endpoint_instance=AsyncMock())
        for _ in range(3)
    ]
    serving = asyncio.get_running_loop().create_future()
    runtime = types.SimpleNamespace(
        set_health_status=Mock(),
        shutdown=Mock(side_effect=lambda: serving.set_result(None)),
    )
    task = asyncio.create_task(drain.finish(runtime, endpoints, serving, 0))
    for _ in range(10):
        await asyncio.sleep(0)
    for ep in endpoints:
        ep.unregister_endpoint_instance.assert_awaited_once()
    runtime.set_health_status.assert_called_once_with(False)
    runtime.shutdown.assert_not_called()
    # A client with a stale discovery snapshot still gets a complete response.
    other = drain.track(stream)({})
    assert await anext(other) == "first"
    release.set()
    assert [x async for x in iterator] == ["last"]
    assert not drain.idle.is_set()
    assert [x async for x in other] == ["last"]
    await asyncio.wait_for(task, 1)
    runtime.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_drain_tracking_releases_cancelled_or_failed_streams():
    drain = router_module.RouterDrain()

    async def stream(request):
        yield "first"
        raise RuntimeError("backend failed")

    iterator = drain.track(stream)({})
    assert await anext(iterator) == "first"
    await iterator.aclose()
    assert drain.active == 0 and drain.idle.is_set()
    with pytest.raises(RuntimeError, match="backend failed"):
        _ = [x async for x in drain.track(stream)({})]
    assert drain.active == 0 and drain.idle.is_set()
