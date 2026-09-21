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


class KvCacheGroupMetadataError(ValueError):
    """The worker extension returned KV cache group metadata of an unexpected shape."""


def validate_worker_group_metadata(rank_results: Any) -> list[dict[str, Any]]:
    """Validate the per-rank worker extension results and return one copy.

    Every rank must report the same groups, and each group must carry its own
    position as ``group_idx``, a non-empty string ``kind`` and a positive
    integer ``block_size``. Anything else means the extension contract does not
    hold on this vLLM release, and the caller must not trust the result.
    """
    if not isinstance(rank_results, (list, tuple)) or not rank_results:
        raise KvCacheGroupMetadataError(
            f"expected one result per worker rank, got {rank_results!r}"
        )

    first = rank_results[0]
    for rank, result in enumerate(rank_results):
        if not isinstance(result, (list, tuple)):
            raise KvCacheGroupMetadataError(
                f"rank {rank} returned {type(result).__name__}, expected a list of groups"
            )
        if list(result) != list(first):
            raise KvCacheGroupMetadataError(
                f"rank {rank} reports different KV cache groups than rank 0: "
                f"{result!r} != {first!r}"
            )

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    group_metadata: list[dict[str, Any]] = []
    for position, group in enumerate(first):
        if not isinstance(group, dict):
            raise KvCacheGroupMetadataError(
                f"group {position} is {type(group).__name__}, expected a dict"
            )
        group_idx, kind, block_size = (
            group.get("group_idx"),
            group.get("kind"),
            group.get("block_size"),
        )
        if not is_int(group_idx) or group_idx != position:
            raise KvCacheGroupMetadataError(
                f"group {position} has group_idx={group_idx!r}"
            )
        if not isinstance(kind, str) or not kind:
            raise KvCacheGroupMetadataError(f"group {position} has kind={kind!r}")
        if not is_int(block_size) or block_size <= 0:
            raise KvCacheGroupMetadataError(
                f"group {position} has block_size={block_size!r}"
            )
        group_metadata.append(dict(group))
    return group_metadata


async def _fetch_group_metadata_from_workers(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> list[dict[str, Any]]:
    """Read cache-group metadata through Dynamo's vLLM worker extension."""
    rank_results = await engine.collective_rpc(KV_CACHE_GROUP_METADATA_METHOD)
    group_metadata = validate_worker_group_metadata(rank_results)

    # KV events carry the block size of vLLM's per-group cache manager, which
    # spans all decode-context-parallel ranks.
    dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
    if dcp_size and dcp_size > 1:
        for group in group_metadata:
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
        # One greppable line: the fallback is only correct when the main
        # attention group uses cache_config.block_size.
        logger.error(
            "KV_EVENT_BLOCK_SIZE_UNVERIFIED: could not read KV cache group "
            "metadata from vLLM (%s: %s); using cache_config.block_size=%s, "
            "which may be WRONG for hybrid/MLA models and would mis-size their "
            "KV events and router block size",
            type(e).__name__,
            e,
            fallback_block_size,
            exc_info=e,
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
