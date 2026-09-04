# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang worker's shutdown drain.

The shutdown module is loaded directly from its file with the native
`dynamo._core` extension and the shared shutdown helper stubbed, so the tests
run without CUDA or the compiled bindings.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_SHUTDOWN_PATH = Path(__file__).parent.parent / "shutdown.py"


async def _noop_shutdown(*args, **kwargs):
    return None


def _load_shutdown():
    """Load shutdown.py from its path with the native runtime and the shared
    helper stubbed only for the duration of the import, so nothing leaks into
    sys.modules for other tests."""
    core_stub = types.ModuleType("dynamo._core")
    core_stub.DistributedRuntime = object
    helper_stub = types.ModuleType("dynamo.common.utils.graceful_shutdown")
    helper_stub.graceful_shutdown_with_discovery = _noop_shutdown
    stubs = {
        "dynamo": types.ModuleType("dynamo"),
        "dynamo._core": core_stub,
        "dynamo.common": types.ModuleType("dynamo.common"),
        "dynamo.common.utils": types.ModuleType("dynamo.common.utils"),
        "dynamo.common.utils.graceful_shutdown": helper_stub,
    }
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "dynamo.sglang.shutdown", _SHUTDOWN_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_shutdown = _load_shutdown()


class _FakeEngine:
    """Engine whose in-flight set shrinks by one on every poll."""

    def __init__(self, in_flight: int):
        self.tokenizer_manager = types.SimpleNamespace(
            rid_to_state={f"r{i}": object() for i in range(in_flight)}
        )

    def tick(self):
        states = self.tokenizer_manager.rid_to_state
        if states:
            states.pop(next(iter(states)))


class _FakeEndpoint:
    """Endpoint whose accepted-request count is set by the test."""

    def __init__(self, in_flight: int):
        self.in_flight = in_flight

    def inflight_requests(self) -> int:
        return self.in_flight


@pytest.fixture(autouse=True)
def reset_registered_drain_sources(monkeypatch):
    monkeypatch.setattr(_shutdown, "_drain_engine", None)
    monkeypatch.setattr(_shutdown, "_drain_endpoints", [])
    monkeypatch.setattr(_shutdown, "_DRAIN_POLL_SECS", 0.001)
    monkeypatch.setattr(_shutdown, "_DRAIN_QUIET_SECS", 0.01)
    yield


def test_in_flight_count_reads_tokenizer_manager():
    engine = _FakeEngine(3)
    assert _shutdown.in_flight_request_count(engine) == 3
    assert _shutdown.in_flight_request_count(object()) == 0


def test_endpoint_count_sums_registered_endpoints_once():
    first = _FakeEndpoint(2)
    second = _FakeEndpoint(1)
    _shutdown.register_drain_endpoint(first)
    _shutdown.register_drain_endpoint(first)
    _shutdown.register_drain_endpoint(second)
    assert _shutdown.endpoint_in_flight_count() == 3


def test_drain_returns_immediately_with_nothing_registered():
    asyncio.run(asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0))


def test_drain_waits_until_engine_in_flight_reaches_zero(monkeypatch):
    engine = _FakeEngine(5)
    _shutdown.register_drain_engine(engine)

    real_sleep = asyncio.sleep

    async def sleep_and_finish_one(_secs):
        engine.tick()
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep_and_finish_one)
    asyncio.run(asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0))
    assert _shutdown.in_flight_request_count(engine) == 0


def test_drain_waits_for_requests_the_endpoint_accepted(monkeypatch):
    """A request the request plane accepted but no handler has started yet
    is invisible to the engine and must still hold the drain."""
    engine = _FakeEngine(0)
    endpoint = _FakeEndpoint(2)
    _shutdown.register_drain_engine(engine)
    _shutdown.register_drain_endpoint(endpoint)

    real_sleep = asyncio.sleep

    async def sleep_and_finish_one(_secs):
        endpoint.in_flight = max(0, endpoint.in_flight - 1)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep_and_finish_one)
    asyncio.run(asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0))
    assert endpoint.in_flight == 0


def test_drain_waits_for_handler_only_workers_without_engine(monkeypatch):
    """Handler-only workers register no engine; their endpoints alone keep
    the drain waiting."""
    endpoint = _FakeEndpoint(1)
    _shutdown.register_drain_endpoint(endpoint)

    real_sleep = asyncio.sleep
    polls = 0

    async def sleep_then_finish(_secs):
        nonlocal polls
        polls += 1
        if polls >= 3:
            endpoint.in_flight = 0
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep_then_finish)
    asyncio.run(asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0))
    assert polls >= 3


def test_drain_restarts_quiet_period_when_a_request_arrives(monkeypatch):
    """A request accepted during the quiet period resets it."""
    endpoint = _FakeEndpoint(0)
    _shutdown.register_drain_endpoint(endpoint)
    monkeypatch.setattr(_shutdown, "_DRAIN_QUIET_SECS", 0.05)

    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()

        async def late_arrival():
            await asyncio.sleep(0.02)
            endpoint.in_flight = 1
            await asyncio.sleep(0.02)
            endpoint.in_flight = 0

        arrival = asyncio.create_task(late_arrival())
        await asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0)
        await arrival
        return loop.time() - started

    # The quiet period restarts after the late request finishes 0.04 s in,
    # so the drain ends no earlier than 0.04 + 0.05 s.
    assert asyncio.run(run()) >= 0.09


def test_drain_is_bounded_by_caller_timeout():
    engine = _FakeEngine(2)
    _shutdown.register_drain_engine(engine)

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_shutdown.drain_in_flight(), timeout=0.05)

    asyncio.run(run())
    assert _shutdown.in_flight_request_count(engine) == 2
