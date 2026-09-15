# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import httpx
import pytest
from dynamo.openai_backend import worker
from dynamo.openai_backend.context_contract import verify_context_contract
from dynamo.openai_backend.launcher_common import build_worker_command
from dynamo.openai_backend.sglang import _build_parser

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def mock_engine(monkeypatch, info, models=None):
    original = httpx.AsyncClient

    def handle(request):
        if request.url.path == "/get_server_info":
            return httpx.Response(404 if info is None else 200, json=info)
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": models or []})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handle)),
    )


@pytest.mark.parametrize("limit", [65536, None, True])
async def test_rejects_undersized_or_unknown_engine(monkeypatch, limit):
    mock_engine(monkeypatch, {"context_length": limit})
    with pytest.raises(RuntimeError, match="Refusing worker registration"):
        await verify_context_contract("http://engine:30000/v1", {"model"}, 1048576)


async def test_accepts_matching_sglang_engine(monkeypatch):
    mock_engine(monkeypatch, {"context_length": 1048576})
    await verify_context_contract("http://engine:30000", {"model"}, 1048576)


async def test_vllm_limit_must_belong_to_the_registered_model(monkeypatch):
    mock_engine(
        monkeypatch,
        None,
        [
            {"id": "other", "max_model_len": 1048576},
            {"id": "model", "max_model_len": 65536},
        ],
    )
    with pytest.raises(RuntimeError, match="engine reports 65536"):
        await verify_context_contract("http://engine:30000", {"model"}, 1048576)


async def test_accepts_matching_vllm_alias(monkeypatch):
    mock_engine(monkeypatch, None, [{"id": "alias", "max_model_len": 1048576}])
    await verify_context_contract("http://engine:30000", {"model", "alias"}, 1048576)


async def test_context_failure_prevents_discovery_registration(monkeypatch):
    config = worker.cmd_line_args(
        ["--model", "model", "--expected-context-length", "1048576"]
    )
    upstream = AsyncMock()
    monkeypatch.setattr(worker, "UpstreamClient", lambda config: upstream)
    monkeypatch.setattr(
        worker,
        "verify_context_contract",
        AsyncMock(side_effect=RuntimeError("context mismatch")),
    )
    register = AsyncMock()
    monkeypatch.setattr(worker, "register_model", register)
    with pytest.raises(RuntimeError, match="context mismatch"):
        await worker.init(None, config, None, "namespace.backend.generate")
    register.assert_not_awaited()
    upstream.aclose.assert_awaited_once()


def test_launcher_preserves_the_context_contract():
    args = _build_parser().parse_args(
        ["--model", "model", "--expected-context-length", "1048576"]
    )
    command = build_worker_command(args)
    assert command[command.index("--expected-context-length") + 1] == "1048576"
