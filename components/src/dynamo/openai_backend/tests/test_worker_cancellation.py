# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime cancellation of forwarded streams: one watcher task per request.

The runtime drives each ``__anext__`` of the response generator as its own
asyncio task (see ``demand_driven_python_stream`` in the bindings), so the
tests that matter run in both consumer shapes: a fresh task per step and one
long-lived task.
"""

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_BLOCK = "<block>"


def _config():
    from dynamo.openai_backend.worker import Config

    return Config(
        model="test-model",
        served_model_name=None,
        upstream_base_url="http://127.0.0.1:30000/v1",
        upstream_health_path="/health",
        connect_timeout_seconds=1.0,
        write_timeout_seconds=1.0,
        embedding_worker=False,
    )


class _FakeContext:
    """The runtime Context surface the worker touches."""

    def __init__(self) -> None:
        self._killed_or_stopped = asyncio.get_running_loop().create_future()
        self.stopped = False

    def is_stopped(self) -> bool:
        return self.stopped

    def is_killed(self) -> bool:
        return self._killed_or_stopped.done()

    def async_killed_or_stopped(self) -> asyncio.Future[bool]:
        return self._killed_or_stopped

    def kill(self) -> None:
        self._killed_or_stopped.set_result(True)


class _FakeEventSource:
    """Stands in for httpx_sse.EventSource; ``_BLOCK`` entries park the reader."""

    def __init__(self, events: list[str]) -> None:
        self.response = SimpleNamespace(status_code=200)
        self._events = events
        self.gate: Optional[asyncio.Future[None]] = None
        self.reads_cancelled = 0
        self.closed = False

    async def aiter_sse(self):
        for data in self._events:
            if data == _BLOCK:
                self.gate = asyncio.get_running_loop().create_future()
                try:
                    await self.gate
                except asyncio.CancelledError:
                    self.reads_cancelled += 1
                    raise
                continue
            yield SimpleNamespace(data=data)


def _chunk(index: int) -> str:
    return json.dumps(
        {"id": "c", "choices": [{"index": 0, "delta": {"content": str(index)}}]}
    )


@pytest.fixture
def shutdown_event(monkeypatch):
    """A per-test shutdown event: asyncio.Event binds to the loop it first waits on."""
    from dynamo.openai_backend import worker

    event = asyncio.Event()
    monkeypatch.setattr(worker, "_SHUTDOWN_EVENT", event)
    return event


@pytest.fixture
def upstream(monkeypatch, shutdown_event):
    from dynamo.openai_backend import worker

    client = worker.UpstreamClient(_config())

    def install(event_source: _FakeEventSource) -> worker.UpstreamClient:
        @contextlib.asynccontextmanager
        async def aconnect_sse(client_, method, url, **kwargs):
            try:
                yield event_source
            finally:
                event_source.closed = True

        monkeypatch.setattr(worker, "aconnect_sse", aconnect_sse)
        return client

    return install


class _Driver:
    """Consume an async generator either one task per step or in the current task."""

    def __init__(self, agen, per_step_task: bool) -> None:
        self._agen = agen
        self._per_step_task = per_step_task

    async def step(self) -> Any:
        if self._per_step_task:
            return await asyncio.ensure_future(self._agen.__anext__())
        return await self._agen.__anext__()

    async def drain(self) -> list[Any]:
        items = []
        while True:
            try:
                items.append(await self.step())
            except StopAsyncIteration:
                return items


def _baseline_tasks() -> set:
    return set(asyncio.all_tasks())


def _cancel_args(exc: BaseException) -> tuple:
    """A CancelledError raised inside a task reaches the task's awaiter as a
    fresh CancelledError with the original chained as ``__context__``; one
    hop per task boundary."""
    seen: Optional[BaseException] = exc
    while isinstance(seen, asyncio.CancelledError):
        if seen.args:
            return seen.args
        seen = seen.__context__
    return ()


async def _parked(source: _FakeEventSource) -> None:
    while source.gate is None:
        await asyncio.sleep(0)


@pytest.mark.parametrize("per_step_task", [True, False])
async def test_normal_stream_yields_chunks_and_done_ends_it(upstream, per_step_task):
    source = _FakeEventSource([_chunk(0), _chunk(1), _chunk(2), "[DONE]", _chunk(3)])
    client = upstream(source)
    context = _FakeContext()
    baseline = _baseline_tasks()

    chunks = await _Driver(
        client._stream_request("/chat/completions", {"messages": []}, context),
        per_step_task,
    ).drain()

    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["0", "1", "2"]
    assert source.closed
    assert _baseline_tasks() == baseline


@pytest.mark.parametrize("per_step_task", [True, False])
async def test_context_kill_mid_stream_raises_cancelled(upstream, per_step_task):
    source = _FakeEventSource([_chunk(0), _BLOCK, _chunk(1)])
    client = upstream(source)
    context = _FakeContext()
    baseline = _baseline_tasks()
    driver = _Driver(
        client._stream_request("/chat/completions", {"messages": []}, context),
        per_step_task,
    )

    await driver.step()
    pending = asyncio.ensure_future(driver.step())
    await _parked(source)
    context.kill()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await pending

    assert _cancel_args(excinfo.value) == ("request was cancelled",)
    assert source.reads_cancelled == 1
    assert source.closed
    assert _baseline_tasks() == baseline
    current = asyncio.current_task()
    if hasattr(current, "cancelling"):
        assert current.cancelling() == 0


@pytest.mark.parametrize("per_step_task", [True, False])
async def test_shutdown_mid_stream_raises_generator_exit(
    upstream, shutdown_event, per_step_task
):
    source = _FakeEventSource([_chunk(0), _BLOCK, _chunk(1)])
    client = upstream(source)
    context = _FakeContext()
    baseline = _baseline_tasks()
    driver = _Driver(
        client._stream_request("/chat/completions", {"messages": []}, context),
        per_step_task,
    )

    await driver.step()
    pending = asyncio.ensure_future(driver.step())
    await _parked(source)
    shutdown_event.set()

    with pytest.raises(GeneratorExit) as excinfo:
        await pending

    assert excinfo.value.args == ("worker shutting down; request can be migrated",)
    assert source.reads_cancelled == 1
    assert source.closed
    assert _baseline_tasks() == baseline


async def test_kill_between_steps_is_seen_before_the_next_read(upstream):
    source = _FakeEventSource([_chunk(0), _chunk(1), _chunk(2)])
    client = upstream(source)
    context = _FakeContext()
    driver = _Driver(
        client._stream_request("/chat/completions", {"messages": []}, context),
        per_step_task=True,
    )

    await driver.step()
    context.kill()
    for _ in range(3):
        await asyncio.sleep(
            0
        )  # the watcher fires while the generator sits at its yield

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await driver.step()

    assert _cancel_args(excinfo.value) == ("request was cancelled",)
    assert source.reads_cancelled == 0
    assert source.closed


async def test_external_cancel_propagates_and_closes_upstream(upstream):
    source = _FakeEventSource([_chunk(0), _BLOCK, _chunk(1)])
    client = upstream(source)
    context = _FakeContext()
    baseline = _baseline_tasks()
    driver = _Driver(
        client._stream_request("/chat/completions", {"messages": []}, context),
        per_step_task=True,
    )

    await driver.step()
    pending = asyncio.ensure_future(driver.step())
    await _parked(source)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await pending

    assert _cancel_args(excinfo.value) == ()
    assert source.reads_cancelled == 1
    assert source.closed
    assert _baseline_tasks() == baseline


async def test_stream_creates_a_constant_number_of_tasks(upstream, monkeypatch):
    from dynamo.openai_backend import worker

    async def count_tasks(events: int) -> int:
        source = _FakeEventSource([_chunk(i) for i in range(events)] + ["[DONE]"])
        client = upstream(source)
        created: list[Any] = []
        real_create_task = asyncio.create_task
        real_ensure_future = asyncio.ensure_future

        def create_task(coro, **kwargs):
            task = real_create_task(coro, **kwargs)
            created.append(task)
            return task

        def ensure_future(obj, **kwargs):
            future = real_ensure_future(obj, **kwargs)
            created.append(future)
            return future

        monkeypatch.setattr(worker.asyncio, "create_task", create_task)
        monkeypatch.setattr(worker.asyncio, "ensure_future", ensure_future)
        try:
            chunks = await _Driver(
                client._stream_request(
                    "/chat/completions", {"messages": []}, _FakeContext()
                ),
                per_step_task=False,
            ).drain()
        finally:
            monkeypatch.setattr(worker.asyncio, "create_task", real_create_task)
            monkeypatch.setattr(worker.asyncio, "ensure_future", real_ensure_future)
        assert len(chunks) == events
        return len({id(t) for t in created})

    few = await count_tasks(4)
    many = await count_tasks(64)

    assert few == many
    assert few <= 3


async def test_forward_ends_the_stream_and_aborts_upstream_on_kill(
    upstream, monkeypatch
):
    source = _FakeEventSource([_chunk(0), _BLOCK, _chunk(1)])
    client = upstream(source)
    aborted: list[str] = []
    monkeypatch.setattr(client, "_schedule_abort", aborted.append)
    context = _FakeContext()
    driver = _Driver(
        client.forward({"messages": [], "rid": "rid-1"}, context), per_step_task=True
    )

    await driver.step()
    pending = asyncio.ensure_future(driver.step())
    await _parked(source)
    context.kill()

    with pytest.raises(StopAsyncIteration):
        await pending

    assert aborted == ["rid-1"]
    assert source.closed


async def test_embed_request_cancels_on_context_kill(shutdown_event, monkeypatch):
    from dynamo.openai_backend import worker

    config = _config()
    config.embedding_worker = True
    client = worker.UpstreamClient(config)
    gate = asyncio.get_running_loop().create_future()
    reads_cancelled = 0

    async def post(url, json=None, headers=None):
        nonlocal reads_cancelled
        try:
            await gate
        except asyncio.CancelledError:
            reads_cancelled += 1
            raise

    monkeypatch.setattr(client, "_client", SimpleNamespace(post=post))
    context = _FakeContext()
    baseline = _baseline_tasks()
    pending = asyncio.ensure_future(
        client._embed_request("/embeddings", {"input": "x"}, context)
    )
    await asyncio.sleep(0)
    context.kill()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await pending

    assert _cancel_args(excinfo.value) == ("request was cancelled",)
    assert reads_cancelled == 1
    assert _baseline_tasks() == baseline
