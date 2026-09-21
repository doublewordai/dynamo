# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public model identity and metadata at the GlobalRouter registration boundary."""

import argparse
import os
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.global_router import __main__ as entrypoint
from dynamo.global_router.backend_args import (
    DynamoGlobalRouterArgGroup,
    DynamoGlobalRouterConfig,
)
from dynamo.llm import ModelType, WorkerType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DYN_GLOBAL_ROUTER_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("DYN_NAMESPACE", raising=False)


def parse_config(*args):
    parser = argparse.ArgumentParser()
    DynamoGlobalRouterArgGroup().add_arguments(parser)
    config = DynamoGlobalRouterConfig.from_cli_args(parser.parse_args(list(args)))
    config.validate()
    return config


def test_legacy_cli_defaults():
    config = parse_config("--config", "pools.json", "--model-name", "served-alias")
    assert config.model_name == "served-alias"
    assert config.model_path is None


def test_environment_and_cli_registration_options(monkeypatch):
    values = {
        "CONFIG": "pools.json",
        "MODEL_NAME": "env-alias",
        "MODEL_PATH": "example/model",
        "REVISION": "pinned-revision",
        "KV_CACHE_BLOCK_SIZE": "256",
        "CONTEXT_LENGTH": "4096",
        "REASONING_PARSER": "deepseek_v4",
        "TOOL_CALL_PARSER": "deepseek_v4",
    }
    for suffix, value in values.items():
        monkeypatch.setenv(f"DYN_GLOBAL_ROUTER_{suffix}", value)
    config = parse_config()
    assert config.model_name == "env-alias"
    assert config.model_path == "example/model"
    assert config.revision == "pinned-revision"
    assert config.kv_cache_block_size == 256
    assert config.context_length == 4096
    assert config.reasoning_parser == config.tool_call_parser == "deepseek_v4"
    override = parse_config("--model-name", "cli-alias", "--model-path", "other/model")
    assert override.model_name == "cli-alias"
    assert override.model_path == "other/model"


@pytest.mark.parametrize(
    "flag", ["--model-path", "--revision", "--reasoning-parser", "--tool-call-parser"]
)
def test_empty_optional_metadata_is_rejected(flag):
    with pytest.raises(ValueError, match="must not be empty"):
        parse_config("--config", "pools.json", "--model-name", "alias", flag, "  ")


@pytest.mark.parametrize("flag", ["--kv-cache-block-size", "--context-length"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_sizes_are_rejected(flag, value):
    with pytest.raises(ValueError, match="must be positive"):
        parse_config("--config", "pools.json", "--model-name", "alias", flag, value)


@pytest.mark.asyncio
async def test_legacy_name_is_still_the_metadata_source(monkeypatch):
    download = Mock()
    monkeypatch.setattr(entrypoint, "snapshot_download", download)
    config = parse_config("--config", "pools.json", "--model-name", "example/model")
    registration = await entrypoint._registration_kwargs(config)
    assert registration["model_path"] == registration["model_name"] == "example/model"
    assert registration["self_host_metadata"] is None
    assert registration["ignore_weights"] is True
    download.assert_not_called()


@pytest.mark.asyncio
async def test_pinned_metadata_and_parsers_are_independent_of_alias(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "metadata-snapshot"
    snapshot.mkdir()
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr(entrypoint, "snapshot_download", download)
    config = parse_config(
        "--config",
        "pools.json",
        "--model-name",
        "served-alias",
        "--model-path",
        "example/model",
        "--revision",
        "pinned-revision",
        "--kv-cache-block-size",
        "256",
        "--context-length",
        "4096",
        "--reasoning-parser",
        "deepseek_v4",
        "--tool-call-parser",
        "deepseek_v4",
    )
    registration = await entrypoint._registration_kwargs(config)
    assert registration["model_path"] == str(snapshot)
    assert registration["model_name"] == "served-alias"
    assert registration["self_host_metadata"] is True
    assert registration["kv_cache_block_size"] == 256
    runtime = registration["runtime_config"]
    assert runtime.context_length == 4096
    assert runtime.reasoning_parser == runtime.tool_call_parser == "deepseek_v4"
    assert runtime.kv_event_publishing_enabled is False
    assert download.call_args.kwargs["repo_id"] == "example/model"
    assert download.call_args.kwargs["revision"] == "pinned-revision"
    patterns = download.call_args.kwargs["allow_patterns"]
    assert "*.json" in patterns
    assert "*.safetensors" not in patterns and "*.bin" not in patterns


@pytest.mark.asyncio
async def test_local_metadata_directory_never_downloads(monkeypatch, tmp_path):
    download = Mock()
    monkeypatch.setattr(entrypoint, "snapshot_download", download)
    config = parse_config(
        "--config", "pools.json", "--model-name", "alias", "--model-path", str(tmp_path)
    )
    registration = await entrypoint._registration_kwargs(config)
    assert registration["model_path"] == str(tmp_path)
    download.assert_not_called()
    config.revision = "some-revision"
    with pytest.raises(ValueError, match="only supported for Hugging Face"):
        await entrypoint._registration_kwargs(config)
    download.assert_not_called()


@pytest.mark.asyncio
async def test_metadata_failure_is_not_retried_with_an_unpinned_source(monkeypatch):
    monkeypatch.setattr(
        entrypoint, "snapshot_download", Mock(side_effect=OSError("download failed"))
    )
    config = parse_config(
        "--config",
        "pools.json",
        "--model-name",
        "alias",
        "--model-path",
        "example/model",
        "--revision",
        "missing",
    )
    with pytest.raises(OSError, match="download failed"):
        await entrypoint._registration_kwargs(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["agg", "disagg"])
async def test_all_endpoints_register_alias_and_preserve_worker_roles(
    monkeypatch, mode
):
    registration = AsyncMock()
    monkeypatch.setattr(entrypoint, "register_model", registration)
    endpoint = Mock(serve_endpoint=AsyncMock())
    runtime = Mock(endpoint=Mock(return_value=endpoint))
    handler = Mock()
    config = parse_config(
        "--config",
        "pools.json",
        "--model-name",
        "alias",
        "--model-path",
        "example/model",
    )
    await getattr(entrypoint, f"_serve_{mode}")(runtime, config, handler)
    calls = [call.kwargs for call in registration.await_args_list]
    assert len(calls) == (2 if mode == "disagg" else 1)
    for call in calls:
        assert call["model_name"] == "alias"
        assert call["model_path"] == "example/model"
        assert call["ignore_weights"] is True
    if mode == "disagg":
        assert calls[0]["worker_type"] == WorkerType.Prefill
        assert calls[0]["needs"] == [[WorkerType.Decode]]
        assert calls[0]["model_type"] == ModelType.Prefill
        assert calls[1]["worker_type"] == WorkerType.Decode
        assert calls[1]["needs"] == [[WorkerType.Prefill]]
        assert calls[0]["runtime_config"] is calls[1]["runtime_config"]
    else:
        assert calls[0]["worker_type"] == WorkerType.Aggregated
        assert "needs" not in calls[0]
