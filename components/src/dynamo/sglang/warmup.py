# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup warmup for aggregated SGLang workers.

The direct Engine API does not run SGLang's HTTP server warmup, so the first
request a worker serves is the first request through its sampler. This sends
that request per DP rank before the worker registers, so first-use kernel
loads and lazy allocations happen at startup rather than under traffic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_WARMUP_TIMEOUT_S = 600
_WARMUP_INPUT_IDS = [10, 11, 12]
_WARMUP_SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 8}


async def warmup_engine(engine: Any, server_args: Any) -> None:
    """Drive every DP rank through one greedy request.

    Raises on timeout or failure so the caller can abort startup instead of
    registering a worker whose sampler is unwarmed.
    """
    if getattr(server_args, "skip_server_warmup", False):
        return
    if not engine.tokenizer_manager.is_generation:
        logger.info("Skipping SGLang warmup for non-generation model")
        return

    async def _warmup_dp_rank(dp_rank: int) -> None:
        results = await engine.async_generate(
            input_ids=_WARMUP_INPUT_IDS,
            sampling_params=_WARMUP_SAMPLING_PARAMS,
            stream=True,
            routed_dp_rank=dp_rank,
        )
        async for _ in results:
            pass

    logger.info("SGLang warmup starting")
    await asyncio.wait_for(
        asyncio.gather(*(_warmup_dp_rank(i) for i in range(server_args.dp_size))),
        timeout=_WARMUP_TIMEOUT_S,
    )
    logger.info("SGLang warmup complete")
