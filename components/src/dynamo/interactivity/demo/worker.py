# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary CPU token worker: reports load, knows nothing about pool policy."""

import asyncio
import logging
import os
import uuid

from aiohttp import web

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerMetricsPublisher,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)


@dynamo_worker()
async def main(runtime: DistributedRuntime):
    stable_id = os.environ["WORKER_ID"]
    endpoint = runtime.endpoint(
        os.environ.get("WORKER_ENDPOINT", "pooldemo.worker.generate")
    )
    ranks = 2
    active = [0] * ranks
    incarnation = str(uuid.uuid4())
    publisher = WorkerMetricsPublisher()
    await publisher.create_endpoint(endpoint)

    async def status(_request):
        return web.json_response(
            dict(
                worker_id=endpoint.connection_id(),
                stable_id=stable_id,
                incarnation=incarnation,
                active_by_rank=active,
                actual_occupied=sum(active),
            )
        )

    app = web.Application()
    app.router.add_get("/status", status)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", 8081).start()

    async def metrics():
        while True:
            for rank, count in enumerate(active):
                publisher.publish(
                    dp_rank=rank,
                    num_active_reqs=count,
                    num_waiting_reqs=0,
                    kv_used_blocks=count * 16,
                )
            await asyncio.sleep(0.1)

    async def generate(request, context):
        routing = request.get("routing") or {}
        rank = routing.get("dp_rank") or 0
        count = min(
            1200, max(1, (request.get("stop_conditions") or {}).get("max_tokens") or 1)
        )
        active[rank] += 1
        logger.info(
            "START node=%s rank=%s actual_occupied=%s tokens=%s",
            stable_id,
            rank,
            sum(active),
            count,
        )
        try:
            for index in range(count):
                await asyncio.sleep(0.1)
                yield {
                    "token_ids": [3],
                    "finish_reason": "stop" if index == count - 1 else None,
                    "meta_info": {
                        "finish_reason": "stop" if index == count - 1 else None
                    },
                }
        finally:
            active[rank] -= 1
            logger.info(
                "FINISH node=%s rank=%s actual_occupied=%s",
                stable_id,
                rank,
                sum(active),
            )

    config = ModelRuntimeConfig()
    config.stable_routing_id = stable_id
    config.max_num_seqs = 8
    config.total_kv_blocks = 4096
    config.context_length = 4096
    config.data_parallel_size = ranks
    config.kv_event_publishing_enabled = False
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        "/opt/model",
        model_name=os.environ.get("MODEL_NAME", "pool-demo"),
        worker_type=WorkerType.Aggregated,
        kv_cache_block_size=16,
        runtime_config=config,
        ignore_weights=True,
    )
    logger.info(
        "READY node=%s worker_id=%s DP ranks=%s CPU dummy; KV blocks synthetic",
        stable_id,
        endpoint.connection_id(),
        ranks,
    )
    reporting = asyncio.create_task(metrics())
    try:
        await endpoint.serve_endpoint(generate)
    finally:
        reporting.cancel()
        await asyncio.gather(reporting, return_exceptions=True)
        await runner.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
