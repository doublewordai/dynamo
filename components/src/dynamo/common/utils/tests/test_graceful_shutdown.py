# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for graceful_shutdown.py

Tests the callback ordering and failure handling used during graceful shutdown.
The drain callback prevents decode worker segfaults when a prefill worker scales
down before in-flight NIXL KV transfers complete (issue #7319). The cleanup
callback releases engine resources before the runtime shuts down.

These tests import graceful_shutdown directly (bypassing the dynamo package hierarchy)
so they work without GPU, NIXL, or TensorRT-LLM installed.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# ---------------------------------------------------------------------------
# Module loading: import graceful_shutdown without triggering the full dynamo
# package (which requires dynamo.llm, CUDA, etc.)
#
# We cannot do `from dynamo.common.utils import graceful_shutdown` because the
# dynamo package __init__ transitively imports dynamo._core, which is a native
# extension (PyO3) requiring CUDA/NIXL libraries that are not available in
# unit test environments. Instead, we stub dynamo._core and load the module
# directly from its file path via importlib.
# ---------------------------------------------------------------------------

_GRACEFUL_SHUTDOWN_PATH = Path(__file__).parent.parent / "graceful_shutdown.py"

# Provide a minimal dynamo._core stub so the module can be loaded
_dynamo_stub = types.ModuleType("dynamo")
_dynamo_core_stub = types.ModuleType("dynamo._core")
_dynamo_core_stub.DistributedRuntime = object
sys.modules.setdefault("dynamo", _dynamo_stub)
sys.modules.setdefault("dynamo._core", _dynamo_core_stub)


def _load_graceful_shutdown():
    spec = importlib.util.spec_from_file_location(
        "dynamo.common.utils.graceful_shutdown",
        _GRACEFUL_SHUTDOWN_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gs = _load_graceful_shutdown()
graceful_shutdown_with_discovery = _gs.graceful_shutdown_with_discovery
install_signal_handlers = _gs.install_signal_handlers


# ---------------------------------------------------------------------------
# Helper: reset the module-level _shutdown_started event between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    _gs._shutdown_started.clear()
    yield
    _gs._shutdown_started.clear()


@pytest.mark.timeout(2)
def test_endpoint_drain_finishes_stream_before_abort_event(monkeypatch):
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_ENDPOINTS", "true")
    monkeypatch.setattr(_gs, "_DRAIN_POLL_SECS", 0.005)
    monkeypatch.setattr(_gs, "_DRAIN_QUIET_SECS", 0.01)

    async def run():
        event = asyncio.Event()
        endpoint = AsyncMock()
        endpoint.inflight_requests = AsyncMock(side_effect=[1, 1, 0, 0, 0, 0])
        runtime = MagicMock()
        task = asyncio.create_task(
            graceful_shutdown_with_discovery(runtime, [endpoint], event, 0)
        )
        await asyncio.sleep(0.005)
        endpoint.unregister_endpoint_instance.assert_awaited_once()
        assert not event.is_set()
        runtime.shutdown.assert_not_called()
        await task
        assert event.is_set()
        runtime.shutdown.assert_called_once()

    asyncio.run(run())


@pytest.mark.timeout(2)
@pytest.mark.parametrize("counter", [1, RuntimeError("unreadable")])
def test_endpoint_drain_is_bounded_when_busy_or_unreadable(monkeypatch, counter):
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_ENDPOINTS", "true")
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS", "0.03")
    monkeypatch.setattr(_gs, "_DRAIN_POLL_SECS", 0.005)

    async def run():
        endpoint = AsyncMock()
        endpoint.inflight_requests = AsyncMock(
            side_effect=counter if isinstance(counter, Exception) else None,
            return_value=counter,
        )
        runtime = MagicMock()
        event = asyncio.Event()
        started = asyncio.get_running_loop().time()
        await graceful_shutdown_with_discovery(runtime, [endpoint], event, 0)
        assert asyncio.get_running_loop().time() - started >= 0.025
        assert event.is_set()
        runtime.shutdown.assert_called_once()

    asyncio.run(run())


def test_parent_launcher_allows_full_worker_drain(monkeypatch):
    monkeypatch.delenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_ENDPOINTS", raising=False)
    assert _gs.worker_shutdown_timeout_seconds() == 20
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_ENDPOINTS", "true")
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS", "295")
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "5")
    assert _gs.worker_shutdown_timeout_seconds() == 340


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_drain_callback_called_before_shutdown():
    """Drain callback must be awaited before runtime.shutdown().

    This is the key regression test for issue #7319: prefill workers holding
    active NIXL RDMA references must drain in-flight transfers before their
    process exits, otherwise decode workers segfault accessing freed GPU memory.
    """
    call_order = []

    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("shutdown"))

    async def mock_drain():
        call_order.append("drain")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=mock_drain,
        )

    asyncio.run(_run())

    assert "drain" in call_order, "drain_callback was not called"
    assert "shutdown" in call_order, "runtime.shutdown was not called"
    drain_idx = call_order.index("drain")
    shutdown_idx = call_order.index("shutdown")
    assert drain_idx < shutdown_idx, (
        "drain_callback must be called before runtime.shutdown() to ensure "
        "in-flight NIXL transfers complete before GPU memory is freed"
    )


def test_no_drain_callback_still_shuts_down():
    """Backward compatibility: shutdown still works without drain_callback."""
    mock_runtime = MagicMock()

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=None,
        )

    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


def test_drain_callback_exception_does_not_block_shutdown():
    """Drain callback exceptions must not block shutdown.

    Even if draining fails (e.g., timeout), the shutdown must still proceed
    so the process exits cleanly.
    """
    mock_runtime = MagicMock()

    async def failing_drain():
        raise RuntimeError("drain timed out")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=failing_drain,
        )

    # Should not raise
    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


def test_cleanup_callback_runs_after_drain():
    """Drain, cleanup, and runtime shutdown must run in that order."""
    call_order = []

    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("shutdown"))

    async def mock_drain():
        call_order.append("drain")

    async def mock_cleanup():
        call_order.append("cleanup")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=mock_drain,
            cleanup_callback=mock_cleanup,
        )

    asyncio.run(_run())

    assert call_order == ["drain", "cleanup", "shutdown"]


def test_pre_shutdown_callback_withdraws_before_worker_teardown():
    """Lease-owned records disappear before the worker shutdown event is set."""
    call_order = []
    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("runtime"))

    async def withdraw():
        assert not shutdown_event.is_set()
        call_order.append("withdraw")

    async def cleanup():
        assert shutdown_event.is_set()
        call_order.append("engine")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)
        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=shutdown_event,
            grace_period_s=0,
            pre_shutdown_callback=withdraw,
            cleanup_callback=cleanup,
        )

    shutdown_event = asyncio.Event()
    asyncio.run(_run())
    assert call_order == ["withdraw", "engine", "runtime"]


def test_cleanup_callback_exception_does_not_block_shutdown():
    """A cleanup failure must not prevent runtime shutdown."""
    mock_runtime = MagicMock()

    async def failing_cleanup():
        raise RuntimeError("engine cleanup failed")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            cleanup_callback=failing_cleanup,
        )

    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


@pytest.mark.timeout(1)
def test_cleanup_callback_timeout_does_not_block_shutdown(monkeypatch):
    """A hanging cleanup callback must time out before runtime shutdown."""
    monkeypatch.setattr(_gs, "_DEFAULT_CLEANUP_TIMEOUT_SECS", 0.05)
    mock_runtime = MagicMock()

    async def hanging_cleanup():
        await asyncio.sleep(10)

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            cleanup_callback=hanging_cleanup,
        )

    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


def test_drain_timeout_from_env(monkeypatch):
    monkeypatch.delenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS", raising=False)
    assert _gs.get_drain_timeout_seconds() == 30.0
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS", "120")
    assert _gs.get_drain_timeout_seconds() == 120.0
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS", "nope")
    assert _gs.get_drain_timeout_seconds() == 30.0


def test_drain_timeout_rejects_non_finite(monkeypatch):
    for bad in ("nan", "inf", "-inf"):
        monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS", bad)
        assert _gs.get_drain_timeout_seconds() == 30.0
