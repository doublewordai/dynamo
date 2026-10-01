# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang worker's shutdown drain on a non-leader node.

The shutdown module is loaded from its file with the native `dynamo._core`
extension and the shared shutdown helper stubbed, so the tests run without
CUDA or the compiled bindings.
"""

import asyncio
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import psutil  # imported first so the stubbed import below leaves it in sys.modules
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
    core_stub = types.ModuleType("dynamo._core")
    core_stub.DistributedRuntime = object
    helper_stub = types.ModuleType("dynamo.common.utils.graceful_shutdown")
    helper_stub.graceful_shutdown_with_discovery = _noop_shutdown
    helper_stub.get_grace_period_seconds = lambda: 5.0
    helper_stub.get_drain_timeout_seconds = lambda: 30.0
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
    assert mod.psutil is psutil
    return mod


_shutdown = _load_shutdown()


def _engine(node_rank: int, child_pids: list[int]):
    return types.SimpleNamespace(
        server_args=types.SimpleNamespace(node_rank=node_rank),
        get_all_child_pids=lambda: list(child_pids),
    )


@pytest.fixture(autouse=True)
def reset_peers(monkeypatch):
    monkeypatch.setattr(_shutdown, "_peer_schedulers", [])
    monkeypatch.setattr(_shutdown, "_PEER_POLL_SECS", 0.005)
    yield


def test_only_a_non_leader_node_registers_its_schedulers():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _shutdown.register_drain_engine(_engine(0, [child.pid]))
        assert _shutdown._peer_schedulers == []
        _shutdown.register_drain_engine(_engine(1, [child.pid]))
        assert [p.pid for p in _shutdown._peer_schedulers] == [child.pid]
    finally:
        child.kill()
        child.wait()


@pytest.mark.timeout(5)
def test_drain_waits_until_the_schedulers_exit():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    _shutdown.register_drain_engine(_engine(1, [child.pid]))

    async def run():
        waiting = asyncio.create_task(_shutdown.wait_for_peer_schedulers())
        await asyncio.sleep(0.05)
        assert not waiting.done()
        child.kill()
        child.wait()
        await asyncio.wait_for(waiting, timeout=2.0)

    asyncio.run(run())


def test_drain_returns_at_once_with_no_schedulers_registered():
    asyncio.run(asyncio.wait_for(_shutdown.wait_for_peer_schedulers(), timeout=1.0))


def test_a_non_leader_waits_for_the_leaders_whole_sequence():
    assert _shutdown.peer_wait_budget_seconds() == 5 + 5 + 30 + 30
