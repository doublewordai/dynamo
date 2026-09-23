# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang embedding input dispatch."""

from typing import Any

import pytest

pytest.importorskip(
    "sglang.srt.managers.io_struct", reason="sglang not installed in this container"
)

from sglang.srt.managers.io_struct import EmbeddingReqInput  # noqa: E402

from dynamo.sglang.request_handlers.embedding import (  # noqa: E402
    embedding_handler as eh,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _TokenizerManager:
    def __init__(self) -> None:
        self.requests: list[tuple[EmbeddingReqInput, Any]] = []

    async def generate_request(self, request: EmbeddingReqInput, context: Any):
        request.normalize_batch_and_arguments()
        self.requests.append((request, context))
        yield {"embedding": [0.1, 0.2], "meta_info": {"prompt_tokens": 2}}


class _Engine:
    def __init__(self) -> None:
        self.tokenizer_manager = _TokenizerManager()
        self.async_encode_calls: list[dict[str, Any]] = []

    async def async_encode(self, **kwargs: Any):
        self.async_encode_calls.append(kwargs)
        return {"embedding": [0.1, 0.2], "meta_info": {"prompt_tokens": 2}}


class _Context:
    trace_id = "embedding-trace"

    def id(self) -> str:
        return "embedding-request"

    def trace_headers(self) -> dict[str, str]:
        return {"traceparent": "00-test"}


def _assert_engine_id(request_id: Any) -> None:
    """Engine IDs are fresh, never the shared trace or caller ID."""
    assert isinstance(request_id, str)
    assert "embedding-trace" not in request_id
    assert request_id != "embedding-request"


def _assert_batch_ids(request_ids: Any, batch_size: int) -> None:
    assert isinstance(request_ids, list)
    assert len(request_ids) == batch_size
    prefixes = {request_id.rsplit("-", 1)[0] for request_id in request_ids}
    assert len(prefixes) == 1
    _assert_engine_id(prefixes.pop())
    assert [request_id.rsplit("-", 1)[1] for request_id in request_ids] == [
        str(index) for index in range(batch_size)
    ]


def _handler(*, enable_trace: bool = True) -> eh.EmbeddingWorkerHandler:
    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)
    handler.engine = _Engine()
    handler.enable_trace = enable_trace
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding_input", ["hello", ["hello", "world"], ["one"]])
async def test_text_inputs_use_async_encode(embedding_input):
    handler = _handler()

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": embedding_input}, _Context()
        )
    ]

    assert len(outputs) == 1
    [call] = handler.engine.async_encode_calls
    assert call["prompt"] == embedding_input
    assert call["external_trace_header"] == {"traceparent": "00-test"}
    if isinstance(embedding_input, list):
        # A list is always a batch, including a one-item list.
        _assert_batch_ids(call["rid"], len(embedding_input))
    else:
        _assert_engine_id(call["rid"])
    assert handler.engine.tokenizer_manager.requests == []


@pytest.mark.asyncio
async def test_requests_sharing_a_trace_get_distinct_engine_ids():
    handler = _handler()

    for _ in range(2):
        async for _output in handler.generate(
            {"model": "embedding-model", "input": "hello"}, _Context()
        ):
            pass

    first, second = (call["rid"] for call in handler.engine.async_encode_calls)
    _assert_engine_id(first)
    _assert_engine_id(second)
    assert first != second


@pytest.mark.asyncio
async def test_empty_batch_is_rejected_before_dispatch():
    handler = _handler()

    with pytest.raises(ValueError, match="must not be empty"):
        async for _output in handler.generate(
            {"model": "embedding-model", "input": []}, _Context()
        ):
            pass

    assert handler.engine.async_encode_calls == []
    assert handler.engine.tokenizer_manager.requests == []


@pytest.mark.asyncio
async def test_single_tokenized_input_uses_native_input_ids():
    handler = _handler()

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": [11, 22, 33]}, _Context()
        )
    ]

    assert len(outputs) == 1
    assert handler.engine.async_encode_calls == []
    [(request, context)] = handler.engine.tokenizer_manager.requests
    assert context is None
    assert request.text is None
    assert request.input_ids == [11, 22, 33]
    _assert_engine_id(request.rid)
    assert request.external_trace_header == {"traceparent": "00-test"}


@pytest.mark.asyncio
async def test_batched_tokenized_input_gets_unique_request_ids():
    handler = _handler(enable_trace=False)
    token_ids = [[11, 22], [33, 44]]

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": token_ids}, _Context()
        )
    ]

    assert len(outputs) == 1
    [(request, context)] = handler.engine.tokenizer_manager.requests
    assert context is None
    assert request.text is None
    assert request.input_ids == token_ids
    _assert_batch_ids(request.rid, 2)
    assert request.external_trace_header is None
