# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend metadata overrides must not replace the engine or its capacity."""

from types import SimpleNamespace

import pytest

from dynamo.sglang import register

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.fixture
def registration(monkeypatch):
    monkeypatch.delenv("DYN_MODEL_METADATA_SOURCE", raising=False)
    captured = {}
    runtime = SimpleNamespace(data_parallel_size=1, max_num_seqs=128)

    async def runtime_config(*args):
        return runtime

    async def publish(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)

    monkeypatch.setattr(register, "_get_runtime_config", runtime_config)
    monkeypatch.setattr(register, "register_model", publish)
    server = SimpleNamespace(
        model_path="engine-weights",
        served_model_name="shared-model",
        page_size=64,
    )
    config = SimpleNamespace(
        use_sglang_tokenizer=False,
        frontend_decoding=False,
        custom_jinja_template=None,
        served_model_aliases=["shared-alias"],
    )

    async def run():
        return await register._register_model_with_runtime_config(
            SimpleNamespace(),
            SimpleNamespace(),
            server,
            config,
            worker_type=register.WorkerType.Aggregated,
        )

    return run, captured, runtime, server, config


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, "frontend-metadata"])
async def test_metadata_override_preserves_engine_and_runtime(
    monkeypatch, registration, override
):
    run, captured, runtime, server, _ = registration
    if override:
        monkeypatch.setenv("DYN_MODEL_METADATA_SOURCE", override)
    assert await run()
    assert captured["args"][3] == (override or "engine-weights")
    assert captured["args"][4] == "shared-model"
    assert captured["kwargs"]["ignore_weights"] is bool(override)
    assert captured["kwargs"]["runtime_config"] is runtime
    assert captured["kwargs"]["kv_cache_block_size"] == 64
    assert captured["kwargs"]["model_aliases"] == ["shared-alias"]
    assert server.model_path == "engine-weights"


@pytest.mark.asyncio
async def test_metadata_override_rejects_empty_source(monkeypatch, registration):
    run, captured, *_ = registration
    monkeypatch.setenv("DYN_MODEL_METADATA_SOURCE", "  ")
    with pytest.raises(ValueError, match="must not be empty"):
        await run()
    assert not captured


@pytest.mark.asyncio
async def test_metadata_override_rejects_engine_tokenization(monkeypatch, registration):
    run, captured, _, _, config = registration
    monkeypatch.setenv("DYN_MODEL_METADATA_SOURCE", "frontend-metadata")
    config.use_sglang_tokenizer = True
    with pytest.raises(ValueError, match="requires token input"):
        await run()
    assert not captured
