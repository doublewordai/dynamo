# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep a standalone router serving across worker namespace generations."""

import asyncio
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from dynamo.llm import KvRouter

logger = logging.getLogger(__name__)


def read_endpoints(path: Path, base_endpoint: str, block_size: int) -> tuple[str, ...]:
    """Read one atomic, authoritative snapshot from the rollout controller.

    The controller must only publish generations with the same model/tokenizer
    contract. Block sizes must match so advisory scheduling costs are comparable.
    """
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise TypeError("generation snapshot must be an object")
    if document.get("block_size") != block_size:
        raise ValueError("generation block_size must match --router-block-size")
    endpoints = document.get("endpoints")
    if not isinstance(endpoints, list) or not all(
        isinstance(e, str) for e in endpoints
    ):
        raise ValueError("endpoints must be a list of endpoint paths")
    base, component, endpoint = base_endpoint.split(".")
    for value in endpoints:
        parts = value.split(".")
        if (
            len(parts) != 3
            or not parts[0].startswith(base + "-")
            or parts[0] == base + "-"
            or parts[1:] != [component, endpoint]
        ):
            raise ValueError(
                f"endpoint is outside the configured generation scope: {value}"
            )
    return tuple(sorted(set(endpoints)))


@dataclass
class Generation:
    endpoint: str
    client: object
    router: object


class GenerationRouter:
    """One KV index and scheduler per generation, one stable ingress endpoint.

    A stream holds its generation alive after removal from the admission table.
    Preview never books work; dispatch performs the normal reservation and cleanup.
    """

    def __init__(self, runtime, endpoint, block_size, config, aic_config, path):
        self.runtime = runtime
        self.endpoint = endpoint
        self.block_size = block_size
        self.config = config
        self.aic_config = aic_config
        self.path = Path(path)
        self.generations = {}
        self._refresh_task = None
        self._last_error = None

    async def start(self):
        await self.refresh()
        self._refresh_task = asyncio.create_task(self._watch())

    async def close(self):
        if self._refresh_task:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)
        self.generations = {}

    async def refresh(self):
        endpoints = await asyncio.to_thread(
            read_endpoints, self.path, self.endpoint, self.block_size
        )
        updated = {}
        for name in endpoints:
            generation = self.generations.get(name)
            if generation is None:
                endpoint = self.runtime.endpoint(name)
                client = await endpoint.client()
                router = await asyncio.to_thread(
                    KvRouter,
                    endpoint=endpoint,
                    block_size=self.block_size,
                    kv_router_config=self.config,
                    aic_perf_config=self.aic_config,
                )
                generation = Generation(name, client, router)
            updated[name] = generation
        # No await between reading this table and replacing it: requests see one
        # complete snapshot, and unchanged generations keep their warm KV state.
        self.generations = updated

    async def _watch(self):
        while True:
            await asyncio.sleep(1)
            try:
                await self.refresh()
                self._last_error = None
            except Exception as error:  # noqa: BLE001 - PyO3 returns generic Exception.
                # An invalid or unreadable update must not withdraw healthy workers.
                message = str(error)
                if message != self._last_error:
                    logger.error(
                        "Keeping serving generations after update failure: %s", error
                    )
                    self._last_error = message

    async def _choose(self, request):
        routing = request.get("routing") or {}
        pinned = routing.get("backend_instance_id")
        allowed = routing.get("allowed_worker_ids")
        candidates = []
        for generation in self.generations.values():
            workers = set(generation.client.instance_ids())
            if pinned is not None:
                workers.intersection_update([pinned])
            if allowed is not None:
                workers.intersection_update(allowed)
            if workers:
                candidates.append((generation, len(workers)))
        if not candidates:
            raise RuntimeError("No eligible worker generation")
        previews = await asyncio.gather(
            *(g.router.preview_request(request) for g, _ in candidates),
            return_exceptions=True,
        )
        scored = [
            (generation, count, preview[2])
            for (generation, count), preview in zip(candidates, previews)
            if preview is not None
            and not isinstance(preview, BaseException)
            and math.isfinite(preview[2])
        ]
        if scored:
            best = min(cost for _, _, cost in scored)
            tied = [(g, n) for g, n, cost in scored if cost == best]
        else:
            # No immediate admission (e.g. overload): let an eligible generation's
            # normal bounded scheduler wait. Never bypass its admission policy.
            tied = [
                candidate
                for candidate, preview in zip(candidates, previews)
                if preview is None
            ]
            if not tied:
                errors = [p for p in previews if isinstance(p, Exception)]
                if errors:
                    raise errors[0]
                raise RuntimeError("No finite generation placement cost")
        # Equal-cost generations should not get equal shares when one has only
        # the first replacement worker and the other still has a large fleet.
        return random.choices([g for g, _ in tied], weights=[n for _, n in tied], k=1)[
            0
        ]

    async def generate_from_request(self, request):
        # Advisory selection may race other requests, just as cross-WorkerSet
        # placement does. The chosen scheduler revalidates and books the request.
        # Do not serialize queued requests here: one overloaded set must not block
        # new requests whose prefixes/capacity fit another generation.
        generation = await self._choose(request)
        stream = await generation.router.generate_from_request(request)

        async def responses():
            # Capturing the generation retains its KV/accounting state during drain.
            owner = generation
            try:
                async for response in stream:
                    yield response
            finally:
                del owner

        return responses()

    async def best_worker(
        self, token_ids, router_config_override=None, *, cache_namespace=None
    ):
        request = {
            "model": "",
            "token_ids": token_ids,
            "router_config_override": router_config_override,
            "routing": {"cache_salt": cache_namespace},
        }
        generation = await self._choose(request)
        return await generation.router.best_worker(
            token_ids, router_config_override, cache_namespace=cache_namespace
        )

    async def get_overlap_scores(self, *args):
        # Shared-cache matches belong to a generation, not to the stable pool.
        generations = tuple(self.generations.values())
        rows = await asyncio.gather(
            *(g.router.get_overlap_scores(*args) for g in generations)
        )
        return {"generations": {g.endpoint: row for g, row in zip(generations, rows)}}
