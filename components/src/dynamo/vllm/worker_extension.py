# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker extension used by Dynamo.

vLLM mixes the class named by ``parallel_config.worker_extension_cls`` into its
worker class, which makes the methods below reachable through
``collective_rpc`` by name. Results must be plain msgpack-serializable values:
the engine-core utility channel rejects custom types unless insecure
serialization is enabled.

This module is imported inside vLLM worker processes, so keep its import-time
dependencies minimal.
"""

from typing import Any

DYNAMO_WORKER_EXTENSION_CLS = "dynamo.vllm.worker_extension.DynamoWorkerExtension"
# Benchmark-mode extension; a subclass of DynamoWorkerExtension that lives in
# gc_policy so that importing it starts the GC policy in the worker process.
FPM_GC_WORKER_EXTENSION_CLS = "dynamo.vllm.gc_policy.FpmGcWorkerExtension"
KV_CACHE_GROUP_METADATA_METHOD = "dynamo_get_kv_cache_group_metadata"


class DynamoWorkerExtension:
    """Methods Dynamo adds to the vLLM worker."""

    def dynamo_get_kv_cache_group_metadata(self) -> list[dict[str, Any]]:
        """Describe the worker's KV cache groups.

        Every worker holds the same groups in the same order as the scheduler
        (only layer names differ), so group indices match the ``group_idx``
        carried by KV events. ``block_size`` is the spec block size; callers
        apply any context-parallel scaling themselves.
        """
        from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

        model_runner = getattr(self, "model_runner", None)
        kv_cache_config = getattr(model_runner, "kv_cache_config", None)
        if kv_cache_config is None:
            return []

        metadata: list[dict[str, Any]] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            kind = get_kv_cache_spec_kind(spec)
            # The scheduler represents a merged uniform-type group by its
            # first layer spec; report the same block size it would use.
            layer_specs = getattr(spec, "kv_cache_specs", None)
            if layer_specs:
                spec = next(iter(layer_specs.values()))
            metadata.append(
                {
                    "group_idx": group_idx,
                    "kind": str(getattr(kind, "value", kind)),
                    "block_size": int(spec.block_size),
                }
            )
        return metadata
