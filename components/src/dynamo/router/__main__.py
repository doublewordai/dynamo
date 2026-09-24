# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone KV Router Service

Usage: python -m dynamo.router --endpoint <namespace.component.endpoint> [args]

This service provides a standalone KV-aware router for any set of workers
in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing
to prefill workers) or any other scenario requiring intelligent KV cache-aware
routing decisions.
"""

import asyncio
import logging
import os
import signal
from typing import Optional

import uvloop

from dynamo.llm import AicPerfConfig, KvRouter, KvRouterConfig
from dynamo.router.args import (
    DynamoRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
)
from dynamo.router.args import parse_args as parse_router_args
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class RouterDrain:
    """Withdraw discovery before draining streams; keep callbacks alive meanwhile."""

    def __init__(self):
        self.requested = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()
        self.active = 0

    def track(self, handler):
        async def tracked(request):
            self.active += 1
            self.idle.clear()
            try:
                async for response in handler(request):
                    yield response
            finally:
                self.active -= 1
                if self.active == 0:
                    self.idle.set()

        return tracked

    async def finish(self, runtime, endpoints, serving, propagation_seconds):
        runtime.set_health_status(False)
        # Removing discovery registration does not close the NATS listener or
        # TCP callbacks. Requests already selected by a stale client can finish.
        pending = list(endpoints)
        while pending:
            results = await asyncio.gather(
                *(ep.unregister_endpoint_instance() for ep in pending),
                return_exceptions=True,
            )
            pending = [
                ep
                for ep, result in zip(pending, results)
                if isinstance(result, BaseException)
            ]
            if pending:
                logger.warning(
                    "Retrying discovery withdrawal for %s endpoints", len(pending)
                )
                await asyncio.sleep(1)
        logger.info(
            "Router withdrawn from discovery; allowing client watches to converge"
        )
        await asyncio.sleep(propagation_seconds)
        logger.info("Draining router streams: active=%s", self.active)
        await self.idle.wait()
        # Runtime endpoint shutdown handles the final ingress/egress race and
        # waits for graceful endpoint teardown before disconnecting transports.
        runtime.shutdown()
        await serving
        logger.info("Router drain complete")


class StandaloneRouterHandler:
    """Handles routing requests to workers using KV-aware routing."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        worker_endpoint_path: str,
        block_size: int,
        kv_router_config: KvRouterConfig,
        aic_perf_config: Optional[AicPerfConfig],
    ):
        self.runtime = runtime
        self.worker_endpoint_path = worker_endpoint_path
        self.block_size = block_size
        self.kv_router_config = kv_router_config
        self.aic_perf_config = aic_perf_config
        self.kv_router: Optional[KvRouter] = None
        self.worker_client: Optional[Client] = None

    async def initialize(self):
        """Initialize the KV router for workers."""
        try:
            # Parse endpoint path (format: namespace.component.endpoint)
            parts = self.worker_endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid endpoint path format: {self.worker_endpoint_path}. "
                    "Expected format: namespace.component.endpoint"
                )
            namespace, component, endpoint = parts

            # Get worker endpoint
            worker_endpoint = self.runtime.endpoint(
                f"{namespace}.{component}.{endpoint}"
            )
            self.worker_client = await worker_endpoint.client()

            self.kv_router = KvRouter(
                endpoint=worker_endpoint,
                block_size=self.block_size,
                kv_router_config=self.kv_router_config,
                aic_perf_config=self.aic_perf_config,
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvRouter: {e}")
            raise

    async def generate(self, request):
        """
        Generate tokens using the KV-aware router.

        This endpoint routes the request to the best worker and streams back results.
        Wraps the request into PreprocessedRequest format and wraps worker responses
        into LLMEngineOutput format.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot process request")
            raise RuntimeError("Router not initialized")

        # Wrap incoming request into PreprocessedRequest format for KvRouter
        # The request should already have most fields, but we ensure it has the structure
        # Build routing hints from request (supports both nested routing object and legacy dp_rank)
        routing = request.get("routing")
        dp_rank = request.get("dp_rank")
        if routing is None and dp_rank is not None:
            routing = {"dp_rank": dp_rank}

        preprocessed_request = {
            "model": request.get("model", "unknown"),
            "token_ids": request["token_ids"],
            # Preserve engine inputs and MM cache-routing metadata across this hop.
            "prompt_embeds": request.get("prompt_embeds"),
            "multi_modal_data": request.get("multi_modal_data"),
            "multi_modal_uuids": request.get("multi_modal_uuids"),
            "mm_routing_info": request.get("mm_routing_info"),
            "stop_conditions": request.get("stop_conditions", {}),
            "sampling_options": request.get("sampling_options", {}),
            "output_options": request.get("output_options", {}),
            "eos_token_ids": request.get("eos_token_ids", []),
            "annotations": request.get("annotations", []),
            "routing": routing,
            "router_config_override": request.get("router_config_override"),
            "prefill_result": request.get("prefill_result"),
            "encoder_result": request.get("encoder_result"),
            "bootstrap_info": request.get("bootstrap_info"),
            "extra_args": request.get("extra_args"),
            "mm_processor_kwargs": request.get("mm_processor_kwargs"),
            "media_io_kwargs": request.get("media_io_kwargs"),
            # SGLang needs this before applying guided-output constraints.
            "require_reasoning": request.get("require_reasoning", False),
        }

        async for worker_output in await self.kv_router.generate_from_request(
            preprocessed_request  # type: ignore[arg-type]
        ):
            # Wrap worker output into LLMEngineOutput format
            # Worker should return dict with at minimum kv_transfer_params in extra_args
            llm_engine_output = {
                "token_ids": worker_output.get("token_ids", []),  # type: ignore[attr-defined]
                "tokens": worker_output.get("tokens"),  # type: ignore[attr-defined]
                "text": worker_output.get("text"),  # type: ignore[attr-defined]
                "cum_log_probs": worker_output.get("cum_log_probs"),  # type: ignore[attr-defined]
                "log_probs": worker_output.get("log_probs"),  # type: ignore[attr-defined]
                "top_logprobs": worker_output.get("top_logprobs"),  # type: ignore[attr-defined]
                "finish_reason": worker_output.get("finish_reason"),  # type: ignore[attr-defined]
                "stop_reason": worker_output.get("stop_reason"),  # type: ignore[attr-defined]
                "index": worker_output.get("index"),  # type: ignore[attr-defined]
                "disaggregated_params": worker_output.get("disaggregated_params"),  # type: ignore[attr-defined]
                "extra_args": worker_output.get("extra_args"),  # type: ignore[attr-defined]
                "completion_usage": worker_output.get("completion_usage"),  # type: ignore[attr-defined]
                # engine_data carries routed_experts/prompt_logprobs; routing_data carries
                # worker_id/token_ids/timing. Forward both so they survive this router.
                "engine_data": worker_output.get("engine_data"),  # type: ignore[attr-defined]
                "routing_data": worker_output.get("routing_data"),  # type: ignore[attr-defined]
            }
            yield llm_engine_output

    async def best_worker_id(
        self, token_ids, router_config_override=None, cache_namespace=None
    ):
        """
        Get the best worker ID for a given set of tokens without actually routing.

        This method returns the worker ID that would be selected based on KV cache
        overlap, but does NOT actually route the request or update router states.
        It's useful for debugging, monitoring, or implementing custom routing logic.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot get best worker")
            raise RuntimeError("Router not initialized")

        (worker_id, _dp_rank, _overlap_blocks) = await self.kv_router.best_worker(
            token_ids,
            router_config_override,
            cache_namespace=cache_namespace,
        )

        yield worker_id

    async def get_overlap_scores(self, request):
        """
        Get per-worker KV overlap by storage tier without routing the request.

        This endpoint returns matched blocks for each worker_id/dp_rank pair.
        Shared-cache hits are request-global and are also reported per row as
        blocks beyond that rank's device-local prefix.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot get overlap scores")
            raise RuntimeError("Router not initialized")

        scores = await self.kv_router.get_overlap_scores(
            request["token_ids"],
            request.get("router_config_override"),
            request.get("block_mm_infos"),
            request.get("lora_name"),
            request.get("include_shared", True),
            request.get("cache_namespace"),
        )

        yield scores


def parse_args(argv=None) -> DynamoRouterConfig:
    """Parse router CLI arguments (compatibility shim delegating to args.parse_args)."""
    return parse_router_args(argv)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the standalone router service."""

    config = parse_args()

    logger.info("Starting Standalone Router Service")
    logger.debug(
        "Configuration: endpoint=%s, router_block_size=%s, "
        "overlap_score_credit=%s, overlap_score_credit_decay=%s, "
        "prefill_load_scale=%s, decode_active_request_weight=%s, "
        "router_temperature=%s, use_kv_events=%s, router_replica_sync=%s, "
        "router_track_active_blocks=%s, router_track_output_blocks=%s, "
        "router_assume_kv_reuse=%s, router_track_prefill_tokens=%s, "
        "router_ttl_secs=%s, router_approximate_cache_policy=%s",
        config.endpoint,
        config.router_block_size,
        config.overlap_score_credit,
        config.overlap_score_credit_decay,
        config.prefill_load_scale,
        config.decode_active_request_weight,
        config.router_temperature,
        config.use_kv_events,
        config.router_replica_sync,
        config.router_track_active_blocks,
        config.router_track_output_blocks,
        config.router_assume_kv_reuse,
        config.router_track_prefill_tokens,
        config.router_ttl_secs,
        config.router_approximate_cache_policy,
    )

    kv_router_config = build_kv_router_config(config)
    aic_perf_config = build_aic_perf_config(config)

    # Create handler
    handler = StandaloneRouterHandler(
        runtime,
        config.endpoint,
        config.router_block_size,
        kv_router_config,
        aic_perf_config,
    )
    await handler.initialize()

    # Create endpoints
    generate_endpoint = runtime.endpoint(f"{config.namespace}.router.generate")
    best_worker_endpoint = runtime.endpoint(f"{config.namespace}.router.best_worker_id")
    overlap_scores_endpoint = runtime.endpoint(
        f"{config.namespace}.router.get_overlap_scores"
    )

    logger.debug("Starting to serve endpoints...")

    drain = RouterDrain()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, drain.requested.set)
    propagation_seconds = float(
        os.environ.get("DYN_ROUTER_DRAIN_PROPAGATION_SECONDS", "5")
    )
    timeout_seconds = float(os.environ.get("DYN_ROUTER_DRAIN_TIMEOUT_SECONDS", "300"))
    if propagation_seconds < 0 or timeout_seconds <= propagation_seconds:
        raise ValueError(
            "Router drain timeout must exceed nonnegative propagation time"
        )
    endpoints = [generate_endpoint, best_worker_endpoint, overlap_scores_endpoint]
    handlers = [handler.generate, handler.best_worker_id, handler.get_overlap_scores]
    serving = asyncio.gather(
        *(
            ep.serve_endpoint(
                drain.track(fn),
                graceful_shutdown=True,
                metrics_labels=[("service", "router")],
            )
            for ep, fn in zip(endpoints, handlers)
        )
    )
    stopping = asyncio.create_task(drain.requested.wait())
    try:
        done, _ = await asyncio.wait(
            [serving, stopping], return_when=asyncio.FIRST_COMPLETED
        )
        if serving in done:
            await serving  # Surface endpoint failures instead of waiting for a signal.
        else:
            logger.info("Router shutdown requested")
            await asyncio.wait_for(
                drain.finish(runtime, endpoints, serving, propagation_seconds),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        logger.error(
            "Router drain timed out after %ss; active=%s", timeout_seconds, drain.active
        )
        raise
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        stopping.cancel()
        runtime.shutdown()
        if not serving.done():
            serving.cancel()
        await asyncio.gather(stopping, serving, return_exceptions=True)
        logger.info("Standalone Router Service shutting down")


def main():
    """Entry point for the standalone router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
