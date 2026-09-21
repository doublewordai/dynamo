# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from .worker_extension import (
    DYNAMO_WORKER_EXTENSION_CLS,
    KV_CACHE_GROUP_METADATA_METHOD,
    DynamoWorkerExtension,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
    "sink_full_attention",
}


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    return additional_config.get(
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        vllm_config.cache_config.block_size,
    )


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    if not group_metadata:
        return fallback_block_size

    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size", fallback_block_size)

    return fallback_block_size


def _is_dynamo_worker_extension(qualname: str | None) -> bool:
    """Whether the configured worker extension is Dynamo's or a subclass of it."""
    if not qualname:
        return False
    if qualname == DYNAMO_WORKER_EXTENSION_CLS:
        return True
    module_name, _, class_name = qualname.rpartition(".")
    try:
        extension_cls = getattr(importlib.import_module(module_name), class_name)
        return issubclass(extension_cls, DynamoWorkerExtension)
    except Exception:
        return False


async def _fetch_group_metadata_from_workers(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> list[dict[str, Any]]:
    """Read cache-group metadata through Dynamo's vLLM worker extension."""
    rank_results = await engine.collective_rpc(KV_CACHE_GROUP_METADATA_METHOD)
    group_metadata = [dict(group) for group in (rank_results[0] or [])]

    # KV events carry the block size of vLLM's per-group cache manager, which
    # spans all decode-context-parallel ranks.
    dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
    if dcp_size and dcp_size > 1:
        for group in group_metadata:
            if group.get("block_size") is not None:
                group["block_size"] *= dcp_size
    return group_metadata


async def fetch_kv_cache_group_metadata(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> list[dict[str, Any]]:
    """Fetch KV cache group metadata (kind and block size per group).

    The worker extension works on every supported vLLM release. The engine-core
    utility only exists in some releases and is used when the worker extension
    slot is taken by a user-provided class.
    """
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if _is_dynamo_worker_extension(
        getattr(parallel_config, "worker_extension_cls", None)
    ):
        return await _fetch_group_metadata_from_workers(engine, vllm_config)

    return await engine.engine_core.call_utility_async("get_kv_cache_group_metadata")


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    fallback_block_size = vllm_config.cache_config.block_size
    try:
        group_metadata = await fetch_kv_cache_group_metadata(engine, vllm_config)
    except Exception as e:
        logger.warning(
            "Failed to fetch KV cache group metadata; falling back to "
            "vLLM cache_config.block_size=%s. KV events of models whose main "
            "attention group uses a different block size will be mis-sized: %s",
            fallback_block_size,
            e,
        )
        kv_event_block_size = fallback_block_size
    else:
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
        )
        logger.info(
            "KV event block size %s (cache_config.block_size=%s, groups=%s)",
            kv_event_block_size,
            fallback_block_size,
            group_metadata,
        )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
