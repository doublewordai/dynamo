#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Service for Hierarchical Routing

Usage: python -m dynamo.global_router --config <config.json> --model-name <model>

This service routes requests to local routers in different namespaces based on
a grid-based pool selection strategy. It supports two modes:

- "disagg" mode: Registers as BOTH prefill AND decode worker. Routes prefill
  requests based on (ISL, TTFT) and decode requests based on (context_length, ITL)
  to separate pool types.

- "agg" mode: Registers as a single generate worker. Routes all requests by
  (TTFT, ITL), optionally extended with ISL, to unified pools that handle both
  prefill and decode.

Both modes support priority-based pool overrides from agent hints.
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

import uvloop
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    RouterConfig,
    RouterMode,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from huggingface_hub import snapshot_download

from .backend_args import DynamoGlobalRouterArgGroup, DynamoGlobalRouterConfig
from .handler import GlobalRouterHandler

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def parse_args() -> DynamoGlobalRouterConfig:
    """Parse command-line arguments for the Global Router service."""
    parser = argparse.ArgumentParser(
        description="Dynamo Global Router Service: Hierarchical routing to worker pools",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    DynamoGlobalRouterArgGroup().add_arguments(parser)
    args = parser.parse_args()
    config = DynamoGlobalRouterConfig.from_cli_args(args)
    config.validate()
    return config


async def _registration_kwargs(config: DynamoGlobalRouterConfig) -> dict[str, Any]:
    """Resolve metadata once and share registration settings across endpoints."""
    model_path = config.model_path or config.model_name
    if model_path is None:
        raise ValueError("A model name or model path is required for registration")
    self_host_metadata = None
    if config.revision is not None:
        if Path(model_path).exists():
            raise ValueError(
                "--revision is only supported for Hugging Face model paths"
            )
        # The Rust registration API accepts a path, not a revision. Resolve only
        # metadata at the requested revision and serve that snapshot to frontends
        # so they cannot silently resolve the repository's default revision.
        model_path = await asyncio.to_thread(
            snapshot_download,
            repo_id=model_path,
            revision=config.revision,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.tiktoken",
                "*.jinja",
                "*.jinja2",
                "*.py",
                "merges.txt",
                "vocab.txt",
            ],
        )
        self_host_metadata = True

    runtime_config = ModelRuntimeConfig()
    runtime_config.kv_event_publishing_enabled = False
    if config.context_length is not None:
        runtime_config.context_length = config.context_length
    if config.reasoning_parser is not None:
        runtime_config.reasoning_parser = config.reasoning_parser
    if config.tool_call_parser is not None:
        runtime_config.tool_call_parser = config.tool_call_parser
    return {
        "model_path": model_path,
        "model_name": config.model_name,
        "kv_cache_block_size": config.kv_cache_block_size,
        "runtime_config": runtime_config,
        # GlobalRouter replicas forward into the same pools; the LocalRouters
        # own worker selection and KV accounting, not the frontend at this hop.
        "router_config": RouterConfig(RouterMode.RoundRobin),
        "self_host_metadata": self_host_metadata,
        "ignore_weights": True,
    }


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the Global Router service."""

    config = parse_args()
    # validate() ensures these are non-None; assert to narrow types for mypy
    assert config.config_path is not None
    assert config.model_name is not None
    logger.info("Starting Global Router Service")
    logger.info("Config: %s", config.config_path)
    logger.info("Served model name: %s", config.model_name)
    logger.info("Model path: %s", config.model_path or config.model_name)
    logger.info("Namespace: %s", config.namespace)

    # Create handler
    handler = GlobalRouterHandler(
        runtime=runtime,
        config_path=config.config_path,
        model_name=config.model_name,
        default_ttft_target_ms=config.default_ttft_target_ms,
        default_itl_target_ms=config.default_itl_target_ms,
    )

    # Initialize connections to local routers
    await handler.initialize()

    logger.info(f"Mode: {handler.config.mode}")
    logger.info(f"Pool info: {handler.get_pool_info()}")

    if handler.config.mode == "disagg":
        await _serve_disagg(runtime, config, handler)
    elif handler.config.mode == "agg":
        await _serve_agg(runtime, config, handler)
    else:
        raise ValueError(f"Unknown mode: {handler.config.mode}")


async def _serve_disagg(
    runtime: DistributedRuntime,
    config: DynamoGlobalRouterConfig,
    handler: GlobalRouterHandler,
) -> None:
    """Register and serve disagg-mode endpoints (prefill + decode)."""
    registration = await _registration_kwargs(config)
    prefill_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.prefill_generate"
    )
    decode_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.decode_generate"
    )

    # The GlobalRouter only forwards tokenized requests. It needs model metadata
    # for its deployment cards, but never loads model weights for inference.
    logger.info("Registering as prefill worker...")
    await register_model(
        model_input=ModelInput.Tokens,
        # Prefill workers have no OpenAI surface; the role is declared via
        # `worker_type=Prefill`. We register the legacy `ModelType.Prefill`
        # marker bit (not a surface) so an OLD frontend, which detects prefill
        # via that bit, still routes disaggregated traffic during the
        # cross-version rollout. A new frontend ignores it and dispatches off `worker_type`.
        model_type=ModelType.Prefill,
        endpoint=prefill_endpoint,
        **registration,
        worker_type=WorkerType.Prefill,
        needs=[[WorkerType.Decode]],
    )
    logger.info(
        f"Registered prefill endpoint: {config.namespace}.{config.component_name}.prefill_generate"
    )

    logger.info("Registering as decode worker...")
    await register_model(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=decode_endpoint,
        **registration,
        worker_type=WorkerType.Decode,
        needs=[[WorkerType.Prefill]],
    )
    logger.info(
        f"Registered decode endpoint: {config.namespace}.{config.component_name}.decode_generate"
    )

    logger.info("Global Router ready (disagg mode) - serving endpoints...")

    try:
        await asyncio.gather(
            prefill_endpoint.serve_endpoint(
                handler.handle_prefill,
                graceful_shutdown=True,
                metrics_labels=[
                    ("service", "global_router"),
                    ("type", "prefill"),
                ],
            ),
            decode_endpoint.serve_endpoint(
                handler.handle_decode,
                graceful_shutdown=True,
                metrics_labels=[
                    ("service", "global_router"),
                    ("type", "decode"),
                ],
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve disagg endpoints: {e}")
        raise
    finally:
        logger.info("Global Router Service shutting down")


async def _serve_agg(
    runtime: DistributedRuntime,
    config: DynamoGlobalRouterConfig,
    handler: GlobalRouterHandler,
) -> None:
    """Register and serve agg-mode endpoint (single generate)."""
    registration = await _registration_kwargs(config)
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.generate"
    )

    # The GlobalRouter only forwards tokenized requests. It needs model metadata
    # for its deployment card, but never loads model weights for inference.
    logger.info("Registering as agg worker (Chat + Completions)...")
    await register_model(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=generate_endpoint,
        **registration,
        worker_type=WorkerType.Aggregated,
    )
    logger.info(
        f"Registered agg endpoint: {config.namespace}.{config.component_name}.generate"
    )

    logger.info("Global Router ready (agg mode) - serving endpoint...")

    try:
        await generate_endpoint.serve_endpoint(
            handler.handle_generate,
            graceful_shutdown=True,
            metrics_labels=[
                ("service", "global_router"),
                ("type", "agg"),
            ],
        )
    except Exception as e:
        logger.error(f"Failed to serve agg endpoint: {e}")
        raise
    finally:
        logger.info("Global Router Service shutting down")


def main():
    """Entry point for the Global Router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
