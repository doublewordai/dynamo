# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small-model worker smoke: measured speed, capacity rejection and idle recovery.

Run separately in each patched engine environment. Profile with
 tests/utils/profile_pytest.py before assigning a measured VRAM marker or adding
this test to a shared-GPU CI selection.
"""

import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests

from tests.router.router_process import FrontendRouterProcess
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess

MODEL = "Qwen/Qwen3-0.6B"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.gpu_1,
    pytest.mark.post_merge,
    pytest.mark.model(MODEL),
    pytest.mark.timeout(600),
]


def _worker_snapshot(path: Path):
    try:
        snapshot = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    workers = snapshot["workers"]
    return (
        workers[0] if snapshot["pool_routing_enabled"] and len(workers) == 1 else None
    )


def _wait_for_idle(path: Path):
    deadline = time.monotonic() + 60
    worker = None
    while time.monotonic() < deadline:
        worker = _worker_snapshot(path)
        if (
            worker
            and worker["healthy"]
            and worker["idle"]
            and not worker["probe_pending"]
            and worker["local_requests"] == 0
        ):
            return worker
        time.sleep(0.1)
    pytest.fail(f"worker did not recover to confirmed idle: {worker}")


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "vllm",
            marks=[
                pytest.mark.vllm,
                pytest.mark.requested_vllm_kv_cache_bytes(268435456),
            ],
        ),
        pytest.param(
            "sglang",
            marks=[pytest.mark.sglang, pytest.mark.requested_sglang_kv_tokens(4096)],
        ),
    ],
)
@pytest.mark.parametrize("discovery_backend", ["etcd"], indirect=True)
@pytest.mark.parametrize("request_plane", ["nats"], indirect=True)
@pytest.mark.parametrize("event_plane", ["zmq"], indirect=True)
def test_measured_decode_speed_rejection_and_idle_recovery(
    request,
    backend,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    tmp_path,
):
    namespace = f"decode-speed-{uuid.uuid4().hex[:10]}"
    ports = dynamo_dynamic_ports
    status = tmp_path / "status.json"
    config = tmp_path / "pools.json"
    # Deliberately unreachable targets make admission depend on the bounded probe.
    config.write_text(
        json.dumps(
            {
                "endpoint": f"{namespace}.worker.generate",
                "default_pool": "throughput",
                "pools": {
                    name: {"min_decode_tps_per_user": 1e9, "minimum": 1}
                    for name in ("interactive", "throughput")
                },
            }
        )
    )
    env = {
        **os.environ,
        "DYN_NAMESPACE": namespace,
        "DYN_REQUEST_PLANE": "nats",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_SYSTEM_PORT": str(ports.system_ports[0]),
        "DYN_FORWARDPASS_METRIC_PORT": str(ports.fpm_port),
    }
    command = [
        sys.executable,
        "-m",
        f"dynamo.{backend}",
        "--enable-decode-metrics",
        "--endpoint",
        f"dyn://{namespace}.worker.generate",
    ]
    if backend == "vllm":
        command += ["--model", MODEL, "--enforce-eager", "--max-model-len", "1024"]
        command += build_gpu_mem_args("build_vllm_gpu_mem_args", env) or [
            "--kv-cache-memory-bytes",
            "268435456",
        ]
    else:
        command += [
            "--model-path",
            MODEL,
            "--context-length",
            "1024",
            "--page-size",
            "16",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
        command += build_gpu_mem_args("build_sglang_gpu_mem_args", env) or [
            "--max-total-tokens",
            "4096",
        ]
    kv_events = {
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": f"tcp://*:{ports.kv_event_port}",
    }
    if backend == "vllm":
        kv_events["enable_kv_cache_events"] = True
    command += ["--kv-events-config", json.dumps(kv_events)]
    frontend_env = {
        "DYN_FRONTEND_INTERACTIVITY_CONFIG": str(config),
        "DYN_FRONTEND_INTERACTIVITY_STATUS_PATH": str(status),
        "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "true",
        "DYN_ROUTER_TRACK_OUTPUT_BLOCKS": "true",
    }
    with (
        ManagedProcess(
            command=command,
            env=env,
            timeout=300,
            display_output=True,
            health_check_urls=[f"http://localhost:{ports.system_ports[0]}/health"],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
        ),
        FrontendRouterProcess(
            request,
            block_size=16,
            frontend_port=ports.frontend_port,
            namespace=namespace,
            event_plane="zmq",
            min_initial_workers=1,
            extra_env=frontend_env,
        ),
    ):
        _wait_for_idle(status)
        url = f"http://localhost:{ports.frontend_port}/v1/completions"
        payload = {
            "model": MODEL,
            "prompt": "Count from one to one hundred:",
            "max_tokens": 512,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first_token = last_token = None
        samples = {}
        completion_tokens = None
        rejected = False
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            assert response.status_code == 200, response.text
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"]["completion_tokens"]
                if any(choice.get("text") for choice in chunk.get("choices", [])):
                    last_token = time.monotonic()
                    first_token = first_token or last_token
                worker = _worker_snapshot(status)
                if worker and worker["slowest_decode_tps_per_user"] is not None:
                    rank = next(iter(worker["ranks"].values()))
                    if first_token and time.monotonic() - first_token > 1:
                        samples[rank["revision"]] = rank["decode_tps_per_user"]
                    if (
                        not rejected
                        and worker["local_requests"] == 1
                        and not worker["idle"]
                    ):
                        with requests.post(
                            url,
                            json={**payload, "stream": False, "max_tokens": 1},
                            timeout=10,
                        ) as denied:
                            assert denied.status_code == 503, denied.text
                            assert "interactivity_capacity" in denied.text
                        rejected = True
        assert rejected, "no busy observation arrived while the request was running"
        assert completion_tokens == 512
        assert first_token is not None and last_token > first_token
        assert len(samples) >= 2, f"insufficient steady-state observations: {samples}"
        observed_tps = (completion_tokens - 1) / (last_token - first_token)
        reported_tps = statistics.median(samples.values())
        # A rolling scheduler window and the client stream span differ; catch unit,
        # token-count and first-token-latency errors without a GPU performance SLA.
        assert 0.33 < reported_tps / observed_tps < 3, (reported_tps, observed_tps)
        idle = _wait_for_idle(status)
        assert idle["slowest_decode_tps_per_user"] is None
        with requests.post(
            url, json={**payload, "stream": False, "max_tokens": 8}, timeout=60
        ) as recovered:
            assert recovered.status_code == 200, recovered.text
        _wait_for_idle(status)
