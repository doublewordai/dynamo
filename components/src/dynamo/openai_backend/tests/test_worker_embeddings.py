# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import Any, Optional

import httpx
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _config(embedding_worker: bool):
    from dynamo.openai_backend.worker import Config

    return Config(
        model="Qwen/Qwen3-Embedding-8B",
        served_model_name=None,
        upstream_base_url="http://127.0.0.1:30000/v1",
        upstream_health_path="/health",
        connect_timeout_seconds=1.0,
        write_timeout_seconds=1.0,
        embedding_worker=embedding_worker,
    )


class _StubHttpClient:
    """Stands in for httpx.AsyncClient, recording the single POST it receives."""

    def __init__(self, response: httpx.Response):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        json: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._response


def _json_response(payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "http://127.0.0.1:30000/v1/embeddings"),
    )


def _client(response: httpx.Response, *, embedding_worker: bool = True):
    from dynamo.openai_backend.worker import UpstreamClient

    upstream = UpstreamClient(_config(embedding_worker))
    upstream._client = _StubHttpClient(response)  # type: ignore[assignment]
    return upstream


def _upstream_payload() -> dict[str, Any]:
    return {
        "object": "list",
        "model": "Qwen/Qwen3-Embedding-8B",
        "data": [{"object": "embedding", "index": 0, "embedding": "AACAPwAAAEA="}],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }


def test_embedding_worker_routes_input_requests_to_embeddings():
    upstream = _client(_json_response(_upstream_payload()))

    assert upstream._resolve_upstream_path({"input": "hello"}) == "/embeddings"


def test_embedding_worker_rejects_a_generation_request():
    from dynamo.openai_backend.worker import HttpError

    upstream = _client(_json_response(_upstream_payload()))

    with pytest.raises(HttpError) as excinfo:
        upstream._resolve_upstream_path({"messages": [{"role": "user"}]})

    assert excinfo.value.code == 400


def test_generation_worker_ignores_input_requests():
    from dynamo.openai_backend.worker import HttpError

    upstream = _client(_json_response(_upstream_payload()), embedding_worker=False)

    with pytest.raises(HttpError):
        upstream._resolve_upstream_path({"input": "hello"})


async def test_forward_yields_the_upstream_body_once():
    upstream = _client(_json_response(_upstream_payload()))

    chunks = [chunk async for chunk in upstream.forward({"input": "hello"})]

    assert chunks == [_upstream_payload()]


async def test_forward_asks_upstream_for_base64_whatever_the_client_wanted():
    """The frontend decodes base64 to float at the HTTP boundary but never
    encodes in the other direction, so the worker hop must always be base64."""
    upstream = _client(_json_response(_upstream_payload()))

    async for _ in upstream.forward({"input": "hello", "encoding_format": "float"}):
        pass

    sent = upstream._client.calls[0]  # type: ignore[attr-defined]
    assert sent["url"] == "http://127.0.0.1:30000/v1/embeddings"
    assert sent["json"]["encoding_format"] == "base64"


async def test_forward_does_not_add_generation_only_fields():
    upstream = _client(_json_response(_upstream_payload()))

    async for _ in upstream.forward({"input": "hello"}):
        pass

    sent = upstream._client.calls[0]["json"]  # type: ignore[attr-defined]
    assert "stream" not in sent
    assert "stream_options" not in sent
    assert "rid" not in sent


async def test_forward_surfaces_an_upstream_error_status():
    from dynamo.openai_backend.worker import HttpError

    upstream = _client(
        _json_response({"error": {"message": "model not found"}}, status_code=404)
    )

    with pytest.raises(HttpError) as excinfo:
        async for _ in upstream.forward({"input": "hello"}):
            pass

    assert excinfo.value.code == 404
    assert "model not found" in str(excinfo.value)


@pytest.mark.parametrize("missing", ["object", "data", "model", "usage"])
async def test_forward_names_a_missing_response_field(missing: str):
    from dynamo.openai_backend.worker import HttpError

    payload = _upstream_payload()
    del payload[missing]
    upstream = _client(_json_response(payload))

    with pytest.raises(HttpError) as excinfo:
        async for _ in upstream.forward({"input": "hello"}):
            pass

    assert excinfo.value.code == 502
    assert missing in str(excinfo.value)


def test_launcher_forwards_the_embedding_flag():
    from dynamo.openai_backend.vllm import _worker_command

    args = SimpleNamespace(
        model="Qwen/Qwen3-Embedding-8B",
        served_model_name=None,
        engine_host="127.0.0.1",
        engine_port=30000,
        api_prefix="/v1",
        health_path="/health",
        embedding_worker=True,
    )

    assert _worker_command(args)[-1] == "--embedding-worker"
