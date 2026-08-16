# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only end-to-end coverage for reported-load text KV routing."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from tests.router.router_process import FrontendRouterProcess
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

MODEL_PATH = "Qwen/Qwen3-0.6B"
MODEL_NAME = "text-kv-e2e"
FAKE_ENGINE = Path(__file__).with_name("fake_openai_text_engine.py")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(MODEL_PATH),
    pytest.mark.timeout(90),
]


def _fake_engine_process(
    request: pytest.FixtureRequest,
    *,
    label: str,
    port: int,
    loads: str,
    queues: str | None = None,
    total_kv_blocks: int = 1000,
) -> ManagedProcess:
    command = [
        sys.executable,
        str(FAKE_ENGINE),
        "--port",
        str(port),
        "--label",
        label,
        "--loads",
        loads,
        "--total-kv-blocks",
        str(total_kv_blocks),
    ]
    if queues is not None:
        command.extend(["--queues", queues])
    return ManagedProcess(
        command=command,
        health_check_urls=[f"http://127.0.0.1:{port}/health"],
        timeout=15,
        display_output=True,
        terminate_all_matching_process_names=False,
        display_name=f"fake-openai-{label}",
        log_dir=request.node.name,
    )


def _worker_process(
    request: pytest.FixtureRequest,
    *,
    label: str,
    namespace: str,
    engine_port: int,
    system_port: int,
    load_report_interval_seconds: float = 0.05,
    extra_env: dict[str, str] | None = None,
) -> ManagedProcess:
    env = os.environ.copy()
    env.update(
        {
            "DYN_NAMESPACE": namespace,
            "DYN_REQUEST_PLANE": "nats",
            "DYN_SYSTEM_PORT": str(system_port),
            "DYN_OPENAI_BACKEND_LOAD_REPORT_INTERVAL_SECS": str(
                load_report_interval_seconds
            ),
        }
    )
    env.update(extra_env or {})
    return ManagedProcess(
        command=[
            sys.executable,
            "-m",
            "dynamo.openai_backend._worker",
            "--model",
            MODEL_PATH,
            "--served-model-name",
            MODEL_NAME,
            "--upstream-base-url",
            f"http://127.0.0.1:{engine_port}/v1",
            "--upstream-health-path",
            "/health",
            "--abort-base-url",
            f"http://127.0.0.1:{engine_port}",
        ],
        env=env,
        health_check_ports=[system_port],
        timeout=30,
        display_output=True,
        terminate_all_matching_process_names=False,
        display_name=f"text-worker-{label}",
        log_dir=request.node.name,
    )


def _wait_for_model(frontend_port: int) -> None:
    deadline = time.monotonic() + 20
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"http://127.0.0.1:{frontend_port}/v1/models", timeout=1
            )
            model_ids = {
                model["id"]
                for model in response.json().get("data", [])
                if "id" in model
            }
            if MODEL_NAME in model_ids:
                return
            last_error = f"available models: {sorted(model_ids)}"
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise AssertionError(
        f"model {MODEL_NAME!r} did not appear on the frontend: {last_error}"
    )


def _completion(frontend_port: int, session_id: str) -> str:
    response = requests.post(
        f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
        headers={"x-dynamo-session-id": session_id},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "route this"}],
            "max_tokens": 1,
            "stream": True,
        },
        stream=True,
        timeout=10,
    )
    response.raise_for_status()

    content = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line.startswith("data: "):
            continue
        data = raw_line.removeprefix("data: ")
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if text:
                content.append(text)
    return "".join(content)


def _wait_for_stable_target(frontend_port: int, expected: str, prefix: str) -> str:
    deadline = time.monotonic() + 10
    observed = []
    consecutive = 0
    while time.monotonic() < deadline:
        session_id = f"{prefix}-{uuid.uuid4().hex}"
        actual = _completion(frontend_port, session_id)
        observed.append(actual)
        if actual == expected:
            consecutive += 1
            if consecutive == 3:
                return session_id
        else:
            consecutive = 0
        time.sleep(0.1)
    raise AssertionError(
        f"never selected {expected!r} three consecutive times; observed {observed!r}"
    )


def _wait_for_session_target(
    frontend_port: int, session_id: str, expected: str
) -> None:
    deadline = time.monotonic() + 15
    observed: list[str] = []
    errors: list[str] = []
    while time.monotonic() < deadline:
        try:
            actual = _completion(frontend_port, session_id)
        except requests.RequestException as error:
            errors.append(str(error))
            time.sleep(0.1)
            continue
        observed.append(actual)
        if actual == expected:
            return
        time.sleep(0.1)
    raise AssertionError(
        f"session {session_id!r} never selected {expected!r}; "
        f"observed={observed!r}, errors={errors!r}"
    )


def _set_loads(engine_port: int, loads: list[float]) -> None:
    response = requests.post(
        f"http://127.0.0.1:{engine_port}/admin/loads",
        json={"loads": loads},
        timeout=2,
    )
    response.raise_for_status()


def _set_queue(engine_port: int, queue: list[int]) -> None:
    response = requests.post(
        f"http://127.0.0.1:{engine_port}/admin/queue",
        json={"queue": queue},
        timeout=2,
    )
    response.raise_for_status()


def _set_metrics_enabled(engine_port: int, enabled: bool) -> None:
    response = requests.post(
        f"http://127.0.0.1:{engine_port}/admin/metrics",
        json={"enabled": enabled},
        timeout=2,
    )
    response.raise_for_status()


def test_text_kv_routes_new_sessions_by_rank_and_reuses_affinity(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8000)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        # Queue/capacity drives initial placement. Deliberately make the
        # preferred rank's KV usage high to prove usage percentage is not the
        # placement signal.
        _set_loads(engine_a_port, [0.10, 0.10])
        _set_loads(engine_b_port, [0.90, 0.90])
        _set_queue(engine_a_port, [100, 100])
        _set_queue(engine_b_port, [100, 0])
        sticky_session = _wait_for_stable_target(
            frontend_port, "worker-b:rank-1", "initial"
        )

        concurrent_session = f"concurrent-{uuid.uuid4().hex}"
        with ThreadPoolExecutor(max_workers=8) as executor:
            concurrent_targets = list(
                executor.map(
                    lambda _: _completion(frontend_port, concurrent_session),
                    range(8),
                )
            )
        assert set(concurrent_targets) == {"worker-b:rank-1"}

        _set_loads(engine_a_port, [0.99, 0.99])
        _set_loads(engine_b_port, [0.01, 0.01])
        _set_queue(engine_a_port, [0, 100])
        _set_queue(engine_b_port, [100, 100])
        _wait_for_stable_target(frontend_port, "worker-a:rank-0", "changed")

        assert _completion(frontend_port, sticky_session) == "worker-b:rank-1"


def test_text_kv_zero_queue_burst_tracks_advertised_capacity(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8050)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-capacity-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request,
                label="worker-a",
                port=engine_a_port,
                loads="0.90",
                queues="0",
                total_kv_blocks=500,
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request,
                label="worker-b",
                port=engine_b_port,
                loads="0.10",
                queues="0",
                total_kv_blocks=100,
            )
        )
        for label, engine_port, system_port in (
            ("a", engine_a_port, system_a_port),
            ("b", engine_b_port, system_b_port),
        ):
            stack.enter_context(
                _worker_process(
                    request,
                    label=label,
                    namespace=namespace,
                    engine_port=engine_port,
                    system_port=system_port,
                    # Keep a single report window around the burst so this
                    # exercises frontend-side dispatch accounting directly.
                    load_report_interval_seconds=30,
                )
            )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        session_ids = [f"capacity-{uuid.uuid4().hex}" for _ in range(60)]
        with ThreadPoolExecutor(max_workers=16) as executor:
            targets = list(
                executor.map(
                    lambda session_id: _completion(frontend_port, session_id),
                    session_ids,
                )
            )

        # 500:100 advertised blocks is exactly 5:1. The high-capacity worker
        # also advertises much higher KV usage, which initial placement ignores.
        assert Counter(targets) == {
            "worker-a:rank-none": 50,
            "worker-b:rank-none": 10,
        }


def test_text_kv_discovers_all_ranks_for_late_subscribing_frontend(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8000)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
            )
        )
        # Let the workers' bootstrap and first scrape reports drain into the
        # non-durable event plane before the frontend subscribes.
        time.sleep(1.0)
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        # Fresh periodic observations must restore every rank. Give one rank a
        # large queue and verify traffic uses the other configured ranks.
        _set_queue(engine_a_port, [100, 0])

        deadline = time.monotonic() + 20
        observed = []
        consecutive_clear = 0
        while consecutive_clear < 5:
            probe = _completion(frontend_port, f"probe-{uuid.uuid4().hex}")
            observed.append(probe)
            if probe != "worker-a:rank-0":
                consecutive_clear += 1
            else:
                # worker-a rank 0's queue report has not reached the
                # frontend yet (the metrics event plane is non-durable and the
                # subscription may land after the re-emit); keep waiting.
                consecutive_clear = 0
            assert time.monotonic() < deadline, (
                f"routing never settled off worker-a:rank-0: {observed!r}"
            )
            time.sleep(0.1)

        targets = [
            _completion(frontend_port, f"seeded-{index}-{uuid.uuid4().hex}")
            for index in range(12)
        ]
        assert "worker-a:rank-0" not in targets, targets
        assert len(set(targets)) >= 2, targets


def test_worker_metrics_heartbeat_reaches_late_frontend(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8000)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-{uuid.uuid4().hex[:12]}"
    heartbeat_env = {"DYN_WORKER_METRICS_HEARTBEAT_SECS": "1"}

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request,
                label="worker-a",
                port=engine_a_port,
                loads="0.90,0.20",
                queues="100,0",
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request,
                label="worker-b",
                port=engine_b_port,
                loads="0.80,0.80",
                queues="100,100",
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
                extra_env=heartbeat_env,
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
                extra_env=heartbeat_env,
            )
        )
        # Let one or more real scrapes publish, then make /metrics fail. Reports
        # drain unheard and only the publisher heartbeat can carry the cached
        # queue/capacity snapshot to the late frontend.
        time.sleep(0.5)
        _set_metrics_enabled(engine_a_port, False)
        _set_metrics_enabled(engine_b_port, False)
        time.sleep(0.5)
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        _wait_for_stable_target(frontend_port, "worker-a:rank-1", "heartbeat")
        followups = {
            _completion(frontend_port, f"heartbeat-{index}-{uuid.uuid4().hex}")
            for index in range(6)
        }
        assert followups == {"worker-a:rank-1"}, followups


def test_text_kv_rebalances_existing_sessions_when_worker_scales_up(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8100)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-scale-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.20,0.30"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.10,0.10"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=1,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        # Give the model-scoped runtime-config watch time to establish worker A
        # as its startup baseline before creating affinity entries.
        time.sleep(0.5)
        session_ids = [f"scale-session-{uuid.uuid4().hex}" for _ in range(100)]
        initial_targets = {
            session_id: _completion(frontend_port, session_id)
            for session_id in session_ids
        }
        assert all(
            selected.startswith("worker-a:rank-")
            for selected in initial_targets.values()
        )

        stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
            )
        )
        _set_queue(engine_a_port, [100, 100])
        _set_queue(engine_b_port, [0, 100])
        _wait_for_stable_target(frontend_port, "worker-b:rank-0", "worker-ready")
        time.sleep(0.5)

        after_scale = {
            session_id: _completion(frontend_port, session_id)
            for session_id in initial_targets
        }
        migrated = {
            session_id: selected
            for session_id, selected in after_scale.items()
            if selected.startswith("worker-b:rank-")
        }

        # Both workers advertise the same KV capacity, so the deterministic
        # cohort should be approximately half of the 100 existing sessions.
        assert 35 <= len(migrated) <= 65
        for session_id, selected in after_scale.items():
            if session_id not in migrated:
                assert selected == initial_targets[session_id]

        # The scale event is consumed once per key. Both migrated and retained
        # sessions now keep their exact worker/rank binding.
        for session_id, selected in after_scale.items():
            assert _completion(frontend_port, session_id) == selected


def test_text_kv_routes_plain_workers_without_dp_rank(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8200)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-plain-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.50"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.50"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)
        _set_queue(engine_a_port, [100])
        _set_queue(engine_b_port, [0])
        session_id = _wait_for_stable_target(
            frontend_port, "worker-b:rank-none", "plain"
        )

        _set_queue(engine_a_port, [0])
        _set_queue(engine_b_port, [100])
        _wait_for_stable_target(frontend_port, "worker-a:rank-none", "plain-new")
        assert _completion(frontend_port, session_id) == "worker-b:rank-none"


def test_text_kv_rebinds_only_sessions_on_a_removed_worker(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8300)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-down-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
            )
        )
        worker_b_stack = contextlib.ExitStack()
        stack.callback(worker_b_stack.close)
        worker_b_stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        _set_queue(engine_a_port, [100, 100])
        _set_queue(engine_b_port, [0, 100])
        removed_session = _wait_for_stable_target(
            frontend_port, "worker-b:rank-0", "removed"
        )

        _set_queue(engine_a_port, [0, 100])
        _set_queue(engine_b_port, [100, 100])
        surviving_session = _wait_for_stable_target(
            frontend_port, "worker-a:rank-0", "surviving"
        )

        # Make rank 1 the least-queued surviving target before removing B.
        _set_queue(engine_a_port, [100, 0])
        worker_b_stack.close()

        # A request that races discovery may fail once. Retrying the same
        # request/session must invalidate B, select A/rank 1, and remain sticky.
        _wait_for_session_target(frontend_port, removed_session, "worker-a:rank-1")
        assert _completion(frontend_port, removed_session) == "worker-a:rank-1"
        assert _completion(frontend_port, removed_session) == "worker-a:rank-1"

        # Affinity already pointing at surviving A remains untouched.
        assert _completion(frontend_port, surviving_session) == "worker-a:rank-0"


def _open_priority_stream(frontend_port: int, priority: int):
    """Open a held streaming request; returns (response, line_iterator).

    The iterator must stay alive for the stream's lifetime: dropping a
    partially-consumed requests/urllib3 body iterator closes the connection
    (urllib3's _error_catcher treats the GeneratorExit as an unclean exit),
    which the frontend would see as a client disconnect.
    """
    response = requests.post(
        f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
        headers={"x-dynamo-request-priority": str(priority)},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hold this slot"}],
            "max_tokens": 4,
            "stream": True,
        },
        stream=True,
        timeout=(5, 60),
    )
    assert response.status_code == 200, (
        f"stream at priority {priority} not admitted: "
        f"{response.status_code}: {response.text}"
    )
    lines = response.iter_lines(decode_unicode=True)
    # Read the first chunk so the request is fully in flight.
    next(lines)
    return response, lines


def _wait_for_admission_rejection(frontend_port: int, priority: int) -> dict:
    """Probe at `priority` until admission rejects (the worker's queue report
    has reached the frontend), returning the rejection body."""
    deadline = time.monotonic() + 15
    last = "no response"
    while time.monotonic() < deadline:
        response = requests.post(
            f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
            headers={"x-dynamo-request-priority": str(priority)},
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "probe"}],
                "max_tokens": 1,
                "stream": True,
            },
            stream=True,
            timeout=(5, 10),
        )
        if response.status_code == 529:
            return response.json()
        last = f"{response.status_code}"
        response.close()
        time.sleep(0.2)
    raise AssertionError(f"admission never rejected at priority {priority}: {last}")


@pytest.mark.timeout(90)
def test_admission_queue_margin_priority_control(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
) -> None:
    """Priority-aware queue-bounded admission through the real reporting chain:
    fake engine /metrics -> openai_backend load reports -> frontend admission.

    With DYN_ADMISSION_QUEUE_MARGIN=1: requests admit while the reported
    engine queue is below the margin; at the margin, same-priority requests
    are rejected with a sanitized structured overload response, and a
    higher-priority request is admitted by evicting the newest lower-priority
    in-flight stream, which receives an in-band structured overload frame
    followed by [DONE]. Response bodies never carry scheduling internals.
    """
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    from tests.router.common import _get_admission_metric

    ports = allocate_ports(3, 8000)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_port, system_port = ports
    namespace = f"admission-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_port, loads="0.10"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_port,
                system_port=system_port,
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="round-robin",
                request_plane="nats",
                extra_env={
                    "DYN_ADMISSION_QUEUE_MARGIN": "1",
                    "DYN_ADMISSION_RETRY_AFTER_MS": "250",
                },
            )
        )
        _wait_for_model(frontend_port)

        # Keep streams open after their first chunk so in-flight requests are
        # long-lived eviction candidates.
        requests.post(
            f"http://127.0.0.1:{engine_port}/admin/hold",
            json={"seconds": 60},
            timeout=5,
        ).raise_for_status()

        # Reported queue is 0 < margin: low-priority streams admit freely.
        low_old, low_old_lines = _open_priority_stream(frontend_port, -100)
        low_new, low_new_lines = _open_priority_stream(frontend_port, -100)
        admission_lines = "\n".join(
            line
            for line in requests.get(
                f"http://127.0.0.1:{frontend_port}/metrics", timeout=5
            ).text.splitlines()
            if "admission" in line and not line.startswith("#")
        )
        assert (
            _get_admission_metric(frontend_port, "worker_admission_inflight") == 2
        ), f"held streams must be tracked in the admission registry:\n{admission_lines}"

        # The engine now reports a queue at the margin.
        requests.post(
            f"http://127.0.0.1:{engine_port}/admin/queue",
            json={"queue": [1]},
            timeout=5,
        ).raise_for_status()

        # Same-priority requests reject once the report lands: sanitized
        # message, retry hint, no scheduling internals.
        rejections_before = _get_admission_metric(
            frontend_port, "worker_admission_rejections_total"
        )
        body = _wait_for_admission_rejection(frontend_port, -100)
        error = body.get("error", body)
        assert error["message"] == "service over capacity, please retry later", body
        details = error.get("details") or {}
        assert details.get("reason") == "admission_capacity", body
        assert details.get("retry_after_ms") == 250, body
        serialized = json.dumps(body)
        assert "-100" not in serialized, f"leaked scheduling internals: {serialized}"
        assert (
            _get_admission_metric(frontend_port, "worker_admission_rejections_total")
            > rejections_before
        )

        # A higher-priority request is admitted by evicting the newest
        # low-priority stream.
        evictions_before = _get_admission_metric(
            frontend_port, "worker_admission_evictions_total"
        )
        high, high_lines = _open_priority_stream(frontend_port, 0)

        error_frames = []
        saw_done = False
        for raw_line in low_new_lines:
            if not raw_line.startswith("data: "):
                continue
            data = raw_line.removeprefix("data: ")
            if data == "[DONE]":
                saw_done = True
                break
            frame = json.loads(data)
            if "error" in frame:
                error_frames.append(frame["error"])
        assert error_frames, "evicted stream saw no in-band error frame"
        victim_error = error_frames[-1]
        assert (
            victim_error["message"] == "service over capacity, please retry later"
        ), victim_error
        assert victim_error["code"] in (429, 529), victim_error
        assert victim_error["retry_after_ms"] == 250, victim_error
        assert "priority" not in json.dumps(victim_error), victim_error
        assert saw_done, "evicted stream must terminate with [DONE]"

        assert (
            _get_admission_metric(frontend_port, "worker_admission_evictions_total")
            > evictions_before
        )

        high.close()
        low_old.close()
