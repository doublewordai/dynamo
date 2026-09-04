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
        self.polls = 0

    def tick(self):
        states = self.tokenizer_manager.rid_to_state
        if states:
            states.pop(next(iter(states)))


@pytest.fixture(autouse=True)
def reset_registered_engine(monkeypatch):
    monkeypatch.setattr(_shutdown, "_drain_engine", None)
    monkeypatch.setattr(_shutdown, "_DRAIN_POLL_SECS", 0.001)
    monkeypatch.setattr(_shutdown, "_DRAIN_QUIET_SECS", 0.01)
    monkeypatch.setattr(_shutdown, "_handler_in_flight", 0)
    yield


def test_in_flight_count_reads_tokenizer_manager():
    engine = _FakeEngine(3)
    assert _shutdown.in_flight_request_count(engine) == 3
    assert _shutdown.in_flight_request_count(object()) == 0


def test_drain_returns_immediately_without_engine():
    asyncio.run(asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0))


def test_drain_waits_for_handler_side_work_without_engine():
    """Handler-only workers register no engine; tracked handlers alone keep
    the drain waiting."""
    started = asyncio.Event()

    async def slow_generate(request, context):
        started.set()
        await asyncio.sleep(0.05)
        yield "done"

    tracked = _shutdown.track_in_flight(slow_generate)

    async def run():
        async def consume():
            async for _ in tracked({}, None):
                pass

        consumer = asyncio.create_task(consume())
        await started.wait()
        assert _shutdown.handler_in_flight_count() == 1
        await asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0)
        assert _shutdown.handler_in_flight_count() == 0
        await consumer

    asyncio.run(run())


def test_drain_waits_until_in_flight_reaches_zero(monkeypatch):
    engine = _FakeEngine(5)
    _shutdown.register_drain_engine(engine)

    real_sleep = asyncio.sleep

    async def sleep_and_finish_one(_secs):
        engine.tick()
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep_and_finish_one)
    asyncio.run(asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0))
    assert _shutdown.in_flight_request_count(engine) == 0


def test_drain_is_bounded_by_caller_timeout(monkeypatch):
    engine = _FakeEngine(2)
    _shutdown.register_drain_engine(engine)

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_shutdown.drain_in_flight(), timeout=0.05)

    asyncio.run(run())
    assert _shutdown.in_flight_request_count(engine) == 2


def test_tracked_handler_counts_requests_until_the_stream_ends():
    async def generate(request, context):
        yield 1
        assert _shutdown.handler_in_flight_count() == 1
        yield 2

    tracked = _shutdown.track_in_flight(generate)

    async def run():
        items = []
        async for item in tracked({}, None):
            items.append(item)
        return items

    assert asyncio.run(run()) == [1, 2]
    assert _shutdown.handler_in_flight_count() == 0


def test_drain_waits_for_handler_side_work(monkeypatch):
    """A request accepted by the endpoint but not yet in the engine keeps
    the drain waiting."""
    engine = _FakeEngine(0)
    _shutdown.register_drain_engine(engine)
    started = asyncio.Event()

    async def slow_generate(request, context):
        started.set()
        await asyncio.sleep(0.05)
        yield "done"

    tracked = _shutdown.track_in_flight(slow_generate)

    async def run():
        async def consume():
            async for _ in tracked({}, None):
                pass

        consumer = asyncio.create_task(consume())
        await started.wait()
        assert _shutdown.handler_in_flight_count() == 1
        await asyncio.wait_for(_shutdown.drain_in_flight(), timeout=1.0)
        assert _shutdown.handler_in_flight_count() == 0
        await consumer

    asyncio.run(run())
