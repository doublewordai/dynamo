# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang request identity for the diffusion LM worker."""

from typing import Any

import pytest

pytest.importorskip("sglang", reason="sglang not installed in this container")

from dynamo.sglang.request_handlers.llm import diffusion_handler as dh  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _Engine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_generate(self, **kwargs: Any):
        self.calls.append(kwargs)

        async def stream():
            if False:
                yield {}

        return stream()


class _Context:
    trace_id = "shared-trace"

    def id(self) -> str:
        return "caller-request"

    def trace_headers(self) -> dict[str, str]:
        return {"traceparent": "00-test"}


def _handler() -> dh.DiffusionWorkerHandler:
    handler = dh.DiffusionWorkerHandler.__new__(dh.DiffusionWorkerHandler)
    handler.engine = _Engine()
    handler.enable_trace = True
    handler.use_sglang_tokenizer = False
    handler._supports_ordered_cancellation = True
    handler.submitted: list[str | None] = []
    handler._get_input_param = lambda request: {"input_ids": [1, 2]}
    handler._build_sampling_params = lambda request: {"max_new_tokens": 1}

    async def process(stream, context, submitted_request_id=None):
        handler.submitted.append(submitted_request_id)
        async for output in stream:
            yield output

    handler._process_token_stream = process
    return handler


@pytest.mark.asyncio
async def test_requests_sharing_a_trace_get_distinct_engine_ids():
    handler = _handler()

    for _ in range(2):
        async for _output in handler.generate({"token_ids": [1, 2]}, _Context()):
            pass

    first, second = (call["rid"] for call in handler.engine.calls)
    assert first != second
    for request_id in (first, second):
        assert request_id not in ("shared-trace", "caller-request")
    assert [call["external_trace_header"] for call in handler.engine.calls] == [
        {"traceparent": "00-test"}
    ] * 2
    # Cancellation aborts the ID SGLang was given, before any output.
    assert handler.submitted == [first, second]
