# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo.common import model_fetch

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def no_sleep(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(model_fetch.asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture
def fetch_once(monkeypatch):
    calls: list[tuple[str, bool]] = []
    outcomes: list = []

    async def fake_once(remote_name, ignore_weights):
        calls.append((remote_name, ignore_weights))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(model_fetch, "_fetch_model_once", fake_once)
    return calls, outcomes


@pytest.mark.asyncio
async def test_retries_then_succeeds(monkeypatch, no_sleep, fetch_once):
    calls, outcomes = fetch_once
    monkeypatch.setenv("DYN_MODEL_FETCH_ATTEMPTS", "5")
    monkeypatch.setenv("DYN_MODEL_FETCH_RETRY_BASE_SECS", "2")
    monkeypatch.setenv("DYN_MODEL_FETCH_RETRY_MAX_SECS", "5")
    outcomes.extend(
        [
            Exception("429 Too Many Requests"),
            Exception("429 again"),
            Exception("429 again"),
            "/cache/model",
        ]
    )

    assert (
        await model_fetch.fetch_model("org/model", ignore_weights=True)
        == "/cache/model"
    )

    assert calls == [("org/model", True)] * 4
    assert no_sleep == [2.0, 4.0, 5.0]


@pytest.mark.asyncio
async def test_raises_after_last_attempt(monkeypatch, no_sleep, fetch_once):
    calls, outcomes = fetch_once
    monkeypatch.setenv("DYN_MODEL_FETCH_ATTEMPTS", "3")
    outcomes.extend([Exception("a"), Exception("b"), Exception("c"), "/never"])

    with pytest.raises(Exception, match="c"):
        await model_fetch.fetch_model("org/model")

    assert len(calls) == 3
    assert len(no_sleep) == 2


@pytest.mark.asyncio
async def test_import_error_is_not_retried(monkeypatch, no_sleep, fetch_once):
    calls, outcomes = fetch_once
    monkeypatch.setenv("DYN_MODEL_FETCH_ATTEMPTS", "5")
    outcomes.extend([ImportError("no dynamo.llm"), "/never"])

    with pytest.raises(ImportError):
        await model_fetch.fetch_model("org/model")

    assert len(calls) == 1
    assert no_sleep == []


@pytest.mark.asyncio
async def test_single_attempt_does_not_retry(monkeypatch, no_sleep, fetch_once):
    calls, outcomes = fetch_once
    monkeypatch.setenv("DYN_MODEL_FETCH_ATTEMPTS", "1")
    outcomes.extend([Exception("boom"), "/never"])

    with pytest.raises(Exception, match="boom"):
        await model_fetch.fetch_model("org/model")

    assert len(calls) == 1
    assert no_sleep == []
