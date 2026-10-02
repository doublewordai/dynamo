# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-container prefill/decode unit for dynamo + vLLM.

One process registers with dynamo as one ordinary aggregated worker. Inside it run
N prefill engines and one decode engine, each on its own GPUs. A request is
prefilled on the prefill engine paired with the decode DP rank the router chose
(spilling to the least-loaded prefill engine when that one falls behind), then
decoded on the decode engine; the KV moves between them through the engines' KV
connector (e.g. NIXL over NVLink inside a MultiConnector that also shares a
Mooncake pool). The prefill and decode handlers are dynamo's own
(``PrefillWorkerHandler`` / ``DecodeWorkerHandler``), chained the way the frontend
chains them in disaggregated serving.

    python -m dynamo.vllm.unit \\
        --unit-prefill-gpus 0,1,2,3 --unit-decode-gpus 4,5,6,7 \\
        --unit-prefill-args "--max-num-batched-tokens 32768" \\
        --unit-decode-args "--data-parallel-size 4 --enable-expert-parallel" \\
        <dynamo.vllm args shared by every engine: --model, --served-model-name,
         --kv-transfer-config, ...>

``--unit-prefill-gpus`` takes one entry per prefill engine; an entry may name
several GPUs joined by ``+`` (``0+1``). Each engine gets its own NIXL side-channel
port and Mooncake store lookup port. Only the decode engine publishes KV events,
load metrics and the model registration; the prefill engines are internal.

``--unit-mooncake-pool 800GB`` starts a Mooncake master and a pool of that size
as child processes before the engines, and points the engines' Mooncake store
connectors at it; the engines attach to the pool owner as dummy clients and move
KV through shared memory. The unit exits if either child exits.
"""

import argparse
import asyncio
import collections
import copy
import json
import logging
import multiprocessing.process as _mp_process
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import uvloop

from dynamo import prometheus_names
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, WorkerType
from dynamo.runtime.logging import configure_dynamo_logging

from .args import parse_args
from .cache_info import configure_kv_event_block_size
from .capacity import per_rank_kv_blocks
from .constants import DisaggregationMode
from .dp_topology import get_dp_range_for_worker
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .health_check import VllmHealthCheckPayload
from .main import (
    register_vllm_model,
    setup_fpm_relay,
    setup_kv_event_publisher,
    setup_kv_state_attachment_owner,
    setup_metrics_collection,
    setup_vllm_engine,
)
from .publisher import StatLoggerFactory
from .state_agent import StateAgentLifecycle
from .worker_factory import WorkerFactory

configure_dynamo_logging()
logger = logging.getLogger(__name__)

MOONCAKE_STORE_CONNECTOR = "MooncakeStoreConnector"

# Engine-core processes inherit the environment at spawn time. Engines are built
# concurrently in threads, so each thread's GPU/port environment is applied only
# around its own process starts, under a lock.
_spawn_env = threading.local()
_spawn_lock = threading.Lock()
_orig_process_start = _mp_process.BaseProcess.start


def _process_start_with_thread_env(self):
    env = getattr(_spawn_env, "env", None)
    if not env:
        return _orig_process_start(self)
    with _spawn_lock:
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return _orig_process_start(self)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


_mp_process.BaseProcess.start = _process_start_with_thread_env


def parse_unit_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--unit-prefill-gpus", required=True)
    ap.add_argument("--unit-decode-gpus", required=True)
    ap.add_argument("--unit-prefill-args", default="")
    ap.add_argument("--unit-decode-args", default="")
    ap.add_argument("--unit-nixl-port-base", type=int, default=5600)
    ap.add_argument("--unit-lookup-port-base", type=int, default=7700)
    ap.add_argument("--unit-prefill-spill-tokens", type=int, default=32768)
    ap.add_argument("--unit-mooncake-pool", default="")
    ap.add_argument("--unit-mooncake-port", type=int, default=50051)
    return ap.parse_known_args(argv)


def _wait_port(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{proc.args[0]} exited with {proc.returncode}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{proc.args[0]} did not listen on {port} within {timeout}s")


class MooncakePool:
    """Mooncake master plus the client that owns the pool memory, as children."""

    def __init__(self, size: str, port: int):
        self.size, self.port = size, port
        self.procs: list[subprocess.Popen] = []
        self.watch_task: asyncio.Task | None = None

    def start(self) -> str:
        env = dict(os.environ, GOMAXPROCS=os.environ.get("GOMAXPROCS", "8"))
        env.pop("PYTHONPATH", None)
        master = subprocess.Popen(
            [
                "mooncake_master",
                f"--port={self.port}",
                "--eviction_high_watermark_ratio=0.95",
                "--enable_metric_reporting=false",
                "--eviction_ratio=0.10",
            ],
            env=env,
        )
        self.procs.append(master)
        _wait_port(self.port, master, 60)
        client = subprocess.Popen(
            [
                "mooncake_client",
                f"--global_segment_size={self.size}",
                f"--master_server_address=127.0.0.1:{self.port}",
                "--metadata_server=P2PHANDSHAKE",
                "--host=127.0.0.1",
                f"--port={self.port + 1}",
                "--protocol=tcp",
                "--threads=8",
            ],
            env=env,
        )
        self.procs.append(client)
        _wait_port(self.port + 1, client, 600)
        config = {
            "mode": "standalone-store",
            "metadata_server": "P2PHANDSHAKE",
            "master_server_address": f"127.0.0.1:{self.port}",
            "global_segment_size": 0,
            "local_buffer_size": "4GB",
            "protocol": "tcp",
            "device_name": "",
            "enable_offload": False,
            # Engines attach to the pool owner as dummy clients and move KV
            # through shared memory with it.
            "real_client_address": f"127.0.0.1:{self.port + 1}",
        }
        fd, path = tempfile.mkstemp(prefix="dynamo-unit-mooncake-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f)
        logger.info(
            "unit: Mooncake pool %s at 127.0.0.1:%d (%s)", self.size, self.port, path
        )
        return path

    async def watch(self) -> None:
        while all(p.poll() is None for p in self.procs):
            await asyncio.sleep(5)
        dead = next(p for p in self.procs if p.poll() is not None)
        logger.error(
            "unit: %s exited with %s; shutting down", dead.args[0], dead.returncode
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def stop(self) -> None:
        for p in reversed(self.procs):
            if p.poll() is None:
                p.terminate()
        for p in reversed(self.procs):
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def _set_store_lookup_port(argv: list[str], port: int) -> list[str]:
    """Give every Mooncake store connector in --kv-transfer-config this lookup
    port: engines sharing a container must not share the lookup socket."""
    out = list(argv)
    for i, arg in enumerate(out):
        if arg == "--kv-transfer-config" and i + 1 < len(out):
            cfg = json.loads(out[i + 1])
            pending = [cfg]
            while pending:
                node = pending.pop()
                if isinstance(node, dict):
                    if node.get("kv_connector") == MOONCAKE_STORE_CONNECTOR:
                        node.setdefault("kv_connector_extra_config", {})[
                            "lookup_rpc_port"
                        ] = port
                    pending.extend(node.values())
                elif isinstance(node, list):
                    pending.extend(node)
            out[i + 1] = json.dumps(cfg)
    return out


class UnitHandler:
    """Prefill on one of the prefill engines, then decode.

    A request goes to the prefill engine paired with the decode DP rank the
    router chose (dp_rank modulo the number of prefill engines), so a session's
    turns keep reusing that engine's GPU prefix cache; it spills to the engine
    with the fewest prompt tokens in flight when the paired engine is more than
    ``spill_tokens`` behind it.
    """

    def __init__(
        self,
        prefill_handlers: list[PrefillWorkerHandler],
        decode_handler,
        spill_tokens: int,
        prefill_dp_sizes: list[int] | None = None,
    ):
        self.prefill_handlers = prefill_handlers
        self.prefill_dp_sizes = prefill_dp_sizes or [1] * len(prefill_handlers)
        self.decode_handler = decode_handler
        self.spill_tokens = spill_tokens
        self.inflight_tokens = [0] * len(prefill_handlers)
        # Seconds from request arrival to the prefill result, and from then to
        # the decode engine's first token; summarised in the log periodically.
        self.prefill_s: collections.deque = collections.deque(maxlen=4096)
        self.handoff_s: collections.deque = collections.deque(maxlen=4096)

    async def log_latency(self, interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval)
            if not self.prefill_s:
                continue
            p, h = sorted(self.prefill_s), sorted(self.handoff_s)
            self.prefill_s.clear()
            self.handoff_s.clear()
            q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
            logger.info(
                "unit: %d prefills, prefill p50 %.3fs p90 %.3fs, prefill->first "
                "token p50 %.3fs p90 %.3fs",
                len(p),
                q(p, 0.5),
                q(p, 0.9),
                q(h, 0.5) if h else float("nan"),
                q(h, 0.9) if h else float("nan"),
            )

    def _pick_prefill(self, request) -> int:
        n = len(self.inflight_tokens)
        least = min(range(n), key=self.inflight_tokens.__getitem__)
        dp_rank = (request.get("routing") or {}).get("dp_rank")
        if dp_rank is None:
            return least
        k = int(dp_rank) % n
        if self.inflight_tokens[k] - self.inflight_tokens[least] > self.spill_tokens:
            return least
        return k

    async def generate(self, request, context):
        tokens = request.get("token_ids") or []
        rank = (request.get("routing") or {}).get("dp_rank")
        k = self._pick_prefill(request)
        t0 = time.monotonic()
        prefill_request = copy.deepcopy(request)
        prefill_routing = prefill_request.get("routing") or {}
        if self.prefill_dp_sizes[k] > 1 and rank is not None:
            # A multi-rank prefill engine serves the session on the rank paired
            # with its decode rank, whose prefix cache already holds it.
            prefill_routing["dp_rank"] = int(rank) % self.prefill_dp_sizes[k]
        else:
            prefill_routing.pop("dp_rank", None)
        prefill_result = None
        self.inflight_tokens[k] += len(tokens)
        try:
            async for out in self.prefill_handlers[k].generate(
                prefill_request, context
            ):
                prefill_result = out
        finally:
            self.inflight_tokens[k] -= len(tokens)
        params = (prefill_result or {}).get("disaggregated_params")
        if not params or prefill_result.get("status") == "error":
            message = (prefill_result or {}).get(
                "message", "prefill returned no disaggregated_params"
            )
            yield {"finish_reason": f"error: {message}", "index": 0, "token_ids": []}
            return
        # Hand off the way the frontend does between prefill and decode
        # workers; the router's dp_rank picks the decode engine's DP rank.
        usage = prefill_result.get("completion_usage") or {}
        request["prefill_result"] = {
            "disaggregated_params": params,
            "prompt_tokens_details": usage.get("prompt_tokens_details"),
        }
        t1 = time.monotonic()
        self.prefill_s.append(t1 - t0)
        first = True
        async for out in self.decode_handler.generate(request, context):
            if first and out.get("token_ids"):
                self.handoff_s.append(time.monotonic() - t1)
                first = False
            yield out

    async def clear_kv_blocks(self, request=None):
        for h in self.prefill_handlers:
            async for _ in h.clear_kv_blocks(request):
                pass
        async for out in self.decode_handler.clear_kv_blocks(request):
            yield out

    async def get_perf_metrics(self, request=None):
        async for out in self.decode_handler.get_perf_metrics(request):
            yield out


def _build_engine(config, gpus: str, nixl_port: int, stat_logger, fpm_worker_id, pool):
    _spawn_env.env = {
        "CUDA_VISIBLE_DEVICES": gpus,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port),
    }
    try:
        # Only the engine that reports to dynamo (the decode engine, which gets
        # the stat logger) registers dynamo's component gauges.
        return setup_vllm_engine(
            config,
            stat_logger,
            fpm_worker_id=fpm_worker_id,
            component_metrics=stat_logger is not None,
        )
    except Exception:
        # The other engines are still starting in threads that cannot be
        # cancelled, so exit now rather than wait for them.
        logger.exception("unit: engine on GPUs %s failed to start", gpus)
        if pool:
            pool.stop()
        os._exit(1)
    finally:
        _spawn_env.env = None


async def unit_worker(argv: list[str], pool: MooncakePool | None = None) -> None:
    unit, common = parse_unit_args(argv)
    prefill_gpus = [g.replace("+", ",") for g in unit.unit_prefill_gpus.split(",") if g]
    decode_gpus = unit.unit_decode_gpus

    decode_config = parse_args(
        _set_store_lookup_port(
            common + shlex.split(unit.unit_decode_args), unit.unit_lookup_port_base
        )
    )
    if decode_config.disaggregation_mode != DisaggregationMode.AGGREGATED:
        raise ValueError(
            "the unit registers as one aggregated worker; do not pass --disaggregation-mode"
        )
    if not decode_config.served_model_name:
        decode_config.served_model_name = (
            decode_config.engine_args.served_model_name
        ) = decode_config.model
    prefill_configs = []
    for k in range(len(prefill_gpus)):
        cfg = parse_args(
            _set_store_lookup_port(
                common
                + shlex.split(unit.unit_prefill_args)
                + ["--disaggregation-mode", "prefill"],
                unit.unit_lookup_port_base + 1 + k,
            )
        )
        cfg.served_model_name = (
            cfg.engine_args.served_model_name
        ) = decode_config.served_model_name
        prefill_configs.append(cfg)

    if pool:
        pool.watch_task = asyncio.create_task(pool.watch())

    shutdown_event = asyncio.Event()
    shutdown_endpoints: list = []
    runtime, loop = create_runtime(
        discovery_backend=decode_config.discovery_backend,
        request_plane=decode_config.request_plane,
        event_plane=decode_config.event_plane,
        response_plane=decode_config.response_plane,
    )
    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)
    factory = WorkerFactory(
        setup_vllm_engine_fn=setup_vllm_engine,
        setup_kv_event_publisher_fn=setup_kv_event_publisher,
        setup_kv_state_attachment_owner_fn=setup_kv_state_attachment_owner,
        register_vllm_model_fn=register_vllm_model,
        setup_fpm_relay_fn=setup_fpm_relay,
        setup_metrics_collection_fn=setup_metrics_collection,
        state_agent_lifecycle=StateAgentLifecycle(),
    )

    ns, comp, ep = (
        decode_config.namespace,
        decode_config.component,
        decode_config.endpoint,
    )
    generate_endpoint = runtime.endpoint(f"{ns}.{comp}.{ep}")
    clear_endpoint = runtime.endpoint(f"{ns}.{comp}.clear_kv_blocks")
    perf_endpoint = runtime.endpoint(f"{ns}.{comp}.get_perf_metrics")
    shutdown_endpoints[:] = [generate_endpoint, clear_endpoint, perf_endpoint]

    # Prefill engines start in threads while the decode engine starts on the
    # event loop thread, where its dynamo stat logger binds to the running loop.
    # Each engine gets its own GPUs and NIXL port.
    prefill_tasks = [
        loop.run_in_executor(
            None,
            _build_engine,
            cfg,
            gpus,
            unit.unit_nixl_port_base + 32 + k,
            None,
            None,
            pool,
        )
        for k, (cfg, gpus) in enumerate(zip(prefill_configs, prefill_gpus))
    ]
    fpm_worker_id = str(generate_endpoint.connection_id())
    stat_logger = StatLoggerFactory(endpoint=generate_endpoint)
    built = [
        _build_engine(
            decode_config,
            decode_gpus,
            unit.unit_nixl_port_base,
            stat_logger,
            fpm_worker_id,
            pool,
        )
    ]
    built += await asyncio.gather(*prefill_tasks)
    logger.info(
        "unit: decode engine on GPUs %s, prefill engines on %s",
        decode_gpus,
        prefill_gpus,
    )

    engine_client, vllm_config, default_sampling_params, prometheus_temp_dir, _ = built[
        0
    ]
    await configure_kv_event_block_size(engine_client, vllm_config)
    _, dp_size = get_dp_range_for_worker(vllm_config)
    stat_logger.set_num_gpu_blocks_all(
        per_rank_kv_blocks(vllm_config.cache_config.num_gpu_blocks, dp_size) or 0
    )
    stat_logger.init_publish()

    max_len = getattr(getattr(vllm_config, "model_config", None), "max_model_len", None)
    decode_handler = DecodeWorkerHandler(
        runtime,
        decode_config,
        engine_client,
        default_sampling_params,
        max_len,
        model_config=getattr(vllm_config, "model_config", None),
        enable_multimodal=decode_config.enable_multimodal,
        generate_endpoint=generate_endpoint,
        use_vllm_tokenizer=decode_config.use_vllm_tokenizer,
        shutdown_event=shutdown_event,
        enable_frontend_decoding=decode_config.frontend_decoding,
    )
    decode_handler.add_temp_dir(prometheus_temp_dir)
    prefill_handlers = []
    for cfg, (p_client, p_vllm_config, p_defaults, p_tmp, _) in zip(
        prefill_configs, built[1:]
    ):
        h = PrefillWorkerHandler(
            runtime,
            cfg,
            p_client,
            p_defaults,
            getattr(
                getattr(p_vllm_config, "model_config", None), "max_model_len", None
            ),
            model_config=getattr(p_vllm_config, "model_config", None),
            enable_multimodal=cfg.enable_multimodal,
            use_vllm_tokenizer=cfg.use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=cfg.frontend_decoding,
        )
        h.add_temp_dir(p_tmp)
        prefill_handlers.append(h)
    handler = UnitHandler(
        prefill_handlers,
        decode_handler,
        unit.unit_prefill_spill_tokens,
        [b[1].parallel_config.data_parallel_size for b in built[1:]],
    )
    handler.latency_task = asyncio.create_task(handler.log_latency())

    # KV events follow the decode engine; a consolidator, when configured,
    # republishes them (port allocated in setup_vllm_engine).
    consolidator_eps = vllm_config.additional_config.get("consolidator_endpoints")
    kv_publishers = await factory._setup_kv_routing(
        decode_config,
        generate_endpoint,
        vllm_config,
        consolidator_enabled=bool(consolidator_eps),
        consolidator_port=(
            int(consolidator_eps[2].split(":")[-1]) if consolidator_eps else None
        ),
    )
    if kv_publishers:
        decode_handler.kv_publishers = kv_publishers
    fpm_relays = setup_fpm_relay(decode_config, generate_endpoint, vllm_config)
    if fpm_relays:
        decode_handler.fpm_relays = fpm_relays
    setup_metrics_collection(decode_config, generate_endpoint, logger)

    decode_handler._first_token_source = await generate_endpoint.first_token_source(
        WorkerType.Aggregated
    )
    await register_vllm_model(
        ModelInput.Text if decode_config.use_vllm_tokenizer else ModelInput.Tokens,
        parse_endpoint_types(decode_config.endpoint_types),
        generate_endpoint,
        decode_config,
        engine_client,
        vllm_config,
        worker_type=WorkerType.Aggregated,
        needs=[],
    )
    health_check_payload = VllmHealthCheckPayload(
        engine_client, use_text_input=decode_config.use_vllm_tokenizer
    ).to_dict()
    name = decode_config.served_model_name or decode_config.model
    labels = [
        (prometheus_names.labels.MODEL, name),
        (prometheus_names.labels.MODEL_NAME, name),
    ]
    logger.info("unit: serving %s as one aggregated worker", name)
    await asyncio.gather(
        generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=labels,
            health_check_payload=health_check_payload,
        ),
        clear_endpoint.serve_endpoint(handler.clear_kv_blocks, metrics_labels=labels),
        perf_endpoint.serve_endpoint(handler.get_perf_metrics, metrics_labels=labels),
    )


def main() -> None:
    unit, _ = parse_unit_args(sys.argv[1:])
    pool = (
        MooncakePool(unit.unit_mooncake_pool, unit.unit_mooncake_port)
        if unit.unit_mooncake_pool
        else None
    )
    try:
        if pool:
            os.environ["MOONCAKE_CONFIG_PATH"] = pool.start()
        uvloop.run(unit_worker(sys.argv[1:], pool))
    finally:
        if pool:
            pool.stop()


if __name__ == "__main__":
    main()
