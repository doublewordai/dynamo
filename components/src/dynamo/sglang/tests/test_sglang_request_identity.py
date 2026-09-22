# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch identity and trace propagation, using real handlers and fake engines."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.request_handlers.embedding.embedding_handler import (
    EmbeddingWorkerHandler,
)
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.diffusion_handler import DiffusionWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler
from dynamo.sglang.request_handlers.multimodal import worker_handler as mm
from dynamo.sglang.request_identity import new_engine_request_id

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.fixture(params=["sampled", "unsampled", "missing"])
def context(request):
    span = Mock()
    flags = "01" if request.param == "sampled" else "00"
    headers = (
        None
        if request.param == "missing"
        else {
            "traceparent": f"00-0123456789abcdef0123456789abcdef-0123456789abcdef-{flags}",
            "tracestate": "test=value",
        }
    )
    return SimpleNamespace(
        id=lambda: "caller-supplied-request-id",
        trace_id="0123456789abcdef0123456789abcdef" if headers else None,
        trace_headers=lambda: headers,
        current_span=lambda: span,
        is_stopped=lambda: False,
    )


async def empty_stream():
    if False:
        yield


@pytest.fixture
def engine():
    async def generate(**kwargs):
        return empty_stream()

    return SimpleNamespace(async_generate=AsyncMock(side_effect=generate))


def configure_handler(handler, engine, enabled, text_mode=False):
    handler.engine = engine
    handler.enable_trace = enabled
    handler.use_sglang_tokenizer = text_mode
    handler.shutdown_event = None
    handler._routed_experts_kwargs = {}
    handler._enable_frontend_decoding = False
    handler._mm_hashes_supported = False
    handler._get_input_param = lambda request: {"input_ids": [1, 2]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    handler._metadata_uploader_from_request = lambda request: None
    handler._build_logprob_kwargs = lambda request: {}
    handler._build_sampling_params = lambda request: {}
    handler.bootstrap_host = "test-prefill"
    handler.bootstrap_port = 0
    handler._generate_bootstrap_room = lambda: 42
    handler._consume_tasks = set()


@pytest.mark.parametrize("mode", ["aggregated", "decode", "prefill", "diffusion"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("text_mode", [False, True])
@pytest.mark.asyncio
async def test_dispatch_keeps_identity_and_parent_independent(
    mode, enabled, text_mode, context, engine
):
    cls = {
        "aggregated": DecodeWorkerHandler,
        "decode": DecodeWorkerHandler,
        "prefill": PrefillWorkerHandler,
        "diffusion": DiffusionWorkerHandler,
    }[mode]
    handler = cls.__new__(cls)
    configure_handler(handler, engine, enabled, text_mode)
    handler.serving_mode = (
        DisaggregationMode.DECODE if mode == "decode" else DisaggregationMode.AGGREGATED
    )
    cancellation_ids = []

    async def process(stream, ctx, *args, request_id=None, **kwargs):
        cancellation_ids.append(request_id)
        async for item in stream:
            yield item

    async def consume(stream, ctx, request_id=None):
        cancellation_ids.append(request_id)
        async for _ in stream:
            pass

    handler._process_token_stream = process
    handler._process_text_stream = process
    handler._consume_results = consume
    request = {
        "token_ids": [1, 2],
        "bootstrap_info": {
            "bootstrap_host": "test-prefill",
            "bootstrap_port": 0,
            "bootstrap_room": 42,
        },
    }

    async def dispatch():
        return [item async for item in handler.generate(request, context)]

    # Same trace AND context, overlapping dispatches: neither can be the rid.
    await asyncio.gather(dispatch(), dispatch())
    calls = [call.kwargs for call in engine.async_generate.call_args_list]
    ids = [call["rid"] for call in calls]
    assert len(set(ids)) == 2
    assert all(UUID(rid).version == 4 for rid in ids)
    assert sorted(cancellation_ids) == sorted(ids)
    for call in calls:
        assert call["external_trace_header"] == (
            context.trace_headers() if enabled else None
        )
        context.current_span().set_attribute.assert_any_call(
            "sglang.request_id", call["rid"]
        )


@pytest.mark.parametrize("prompt", ["hello", ["hello"], ["hello", "world"]])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_embedding_dispatch_matches_input_shape(prompt, enabled, context):
    handler = EmbeddingWorkerHandler.__new__(EmbeddingWorkerHandler)
    handler.enable_trace = enabled
    handler.engine = SimpleNamespace(async_encode=AsyncMock(return_value=[]))
    handler._transform_response = Mock(return_value={"data": []})
    for _ in range(2):
        assert [
            item
            async for item in handler.generate(
                {"model": "test", "input": prompt}, context
            )
        ] == [{"data": []}]
    calls = [call.kwargs for call in handler.engine.async_encode.call_args_list]
    for call in calls:
        assert call["prompt"] == prompt
        assert call["external_trace_header"] == (
            context.trace_headers() if enabled else None
        )
        if isinstance(prompt, list):
            ids = call["rid"]
            assert len(ids) == len(prompt)
            prefix = ids[0].rsplit("-", 1)[0]
            assert UUID(prefix).version == 4
            assert ids == [f"{prefix}-{i}" for i in range(len(prompt))]
            context.current_span().set_attribute.assert_any_call(
                "sglang.request_id_prefix", prefix
            )
        else:
            assert UUID(call["rid"]).version == 4
    assert calls[0]["rid"] != calls[1]["rid"]


@pytest.mark.parametrize("mode", ["aggregated", "decode", "prefill"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_dedicated_multimodal_dispatch_ids(
    mode, enabled, context, engine, monkeypatch
):
    handler_cls = (
        mm.MultimodalPrefillWorkerHandler
        if mode == "prefill"
        else mm.MultimodalWorkerHandler
    )
    handler = handler_cls.__new__(handler_cls)
    handler.engine = engine
    handler.enable_trace = enabled
    handler.embeddings_processor = Mock()
    handler.bootstrap_host = "test-prefill"
    handler.bootstrap_port = 0
    handler._get_bootstrap_from_prefill = AsyncMock(
        return_value={
            "bootstrap_host": "test-prefill",
            "bootstrap_port": 0,
            "bootstrap_room": 42,
        }
    )
    monkeypatch.setattr(mm.SglangUtils, "build_sampling_params", lambda request: {})
    monkeypatch.setattr(
        mm, "_build_mm_items", AsyncMock(return_value=([], None, None, None))
    )
    consumer_done = asyncio.Event()

    async def consume(stream, tensor_id):
        async for _ in stream:
            pass
        consumer_done.set()

    handler._consume_results = consume
    request = SimpleNamespace(request=SimpleNamespace(token_ids=[1, 2]))
    for _ in range(2):
        if mode == "prefill":
            consumer_done.clear()
            await handler._process_prefill_generation(
                SimpleNamespace(request=request, sampling_params={}), 42, context
            )
            await consumer_done.wait()
        else:
            method = (
                handler._generate_disaggregated
                if mode == "decode"
                else handler._generate_aggregated
            )
            assert [item async for item in method(request, lambda: None, context)] == []
    calls = [call.kwargs for call in engine.async_generate.call_args_list]
    assert len({call["rid"] for call in calls}) == 2
    for call in calls:
        assert UUID(call["rid"]).version == 4
        assert call["external_trace_header"] == (
            context.trace_headers() if enabled else None
        )


@pytest.mark.asyncio
async def test_shared_trace_cancellation_targets_only_selected_invocation(context):
    # Two live invocations on the same trace and context: aborting one must
    # not touch the other, so the abort key has to be the per-dispatch rid.
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.shutdown_event = None
    aborted = asyncio.Event()
    abort_request = Mock(side_effect=lambda **kwargs: aborted.set())
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(abort_request=abort_request)
    )
    loop = asyncio.get_running_loop()
    killed = [loop.create_future(), loop.create_future()]
    contexts = [
        SimpleNamespace(**vars(context), async_killed_or_stopped=lambda i=i: killed[i])
        for i in range(2)
    ]
    ids = [new_engine_request_id(ctx) for ctx in contexts]
    tasks = [
        asyncio.create_task(handler._handle_cancellation(rid, ctx))
        for rid, ctx in zip(ids, contexts)
    ]
    try:
        await asyncio.sleep(0)  # let both monitors arm
        killed[0].set_result(None)
        await aborted.wait()
        abort_request.assert_called_once_with(rid=ids[0], abort_all=False)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
