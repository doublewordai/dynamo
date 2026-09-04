# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the aggregated SGLang worker startup warmup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.sglang.warmup import warmup_engine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _server_args(**overrides):
    values = {"skip_server_warmup": False, "dp_size": 8}
    values.update(overrides)
    return SimpleNamespace(**values)


def _engine(calls, consumed, is_generation=True):
    async def async_generate(**kwargs):
        calls.append(kwargs)

        async def results():
            consumed.append(kwargs["routed_dp_rank"])
            yield {"meta_info": {"dp_rank": kwargs["routed_dp_rank"]}}

        return results()

    return SimpleNamespace(
        async_generate=async_generate,
        tokenizer_manager=SimpleNamespace(is_generation=is_generation),
    )


@pytest.mark.asyncio
async def test_warmup_covers_every_dp_rank():
    calls, consumed = [], []

    await warmup_engine(_engine(calls, consumed), _server_args())

    assert len(calls) == 8
    for call in calls:
        assert call["input_ids"] == [10, 11, 12]
        assert call["stream"] is True
        assert call["sampling_params"] == {"temperature": 0, "max_new_tokens": 8}
    assert sorted(call["routed_dp_rank"] for call in calls) == list(range(8))
    assert sorted(consumed) == list(range(8))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skip_server_warmup", "is_generation"),
    [(True, True), (False, False)],
)
async def test_warmup_skips_disabled_or_non_generation_models(
    skip_server_warmup, is_generation
):
    calls, consumed = [], []

    await warmup_engine(
        _engine(calls, consumed, is_generation=is_generation),
        _server_args(skip_server_warmup=skip_server_warmup),
    )

    assert calls == []


@pytest.mark.asyncio
async def test_warmup_raises_when_a_request_fails():
    async def async_generate(**kwargs):
        raise RuntimeError("engine rejected the request")

    engine = SimpleNamespace(
        async_generate=async_generate,
        tokenizer_manager=SimpleNamespace(is_generation=True),
    )

    with pytest.raises(RuntimeError, match="engine rejected"):
        await warmup_engine(engine, _server_args(dp_size=2))
