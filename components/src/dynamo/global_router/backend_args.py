#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Dynamo Global Router configuration ArgGroup."""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, env_or_default


class DynamoGlobalRouterArgGroup(ArgGroup):
    """Global Router-specific Dynamo configuration (CLI and env)."""

    def add_arguments(self, parser) -> None:
        """Add Dynamo Global Router arguments to parser."""
        g = parser.add_argument_group("Dynamo Global Router Options")

        add_argument(
            g,
            flag_name="--config",
            env_var="DYN_GLOBAL_ROUTER_CONFIG",
            default=None,
            help="Path to the JSON configuration file defining pool namespaces and selection strategy. Must be set via CLI or env.",
            dest="config_path",
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_GLOBAL_ROUTER_MODEL_NAME",
            default=None,
            help="Public served model name. Also the metadata source when --model-path is omitted. Must be set via CLI or env.",
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_GLOBAL_ROUTER_MODEL_PATH",
            default=None,
            help="Hugging Face repository or local model metadata directory. Defaults to --model-name.",
        )
        add_argument(
            g,
            flag_name="--revision",
            env_var="DYN_GLOBAL_ROUTER_REVISION",
            default=None,
            help="Hugging Face metadata revision. Downloads metadata only and self-hosts it for frontends; not valid with a local model path.",
        )
        add_argument(
            g,
            flag_name="--kv-cache-block-size",
            env_var="DYN_GLOBAL_ROUTER_KV_CACHE_BLOCK_SIZE",
            default=None,
            arg_type=int,
            help="KV cache block size in tokens. Must match the routed workers.",
        )
        add_argument(
            g,
            flag_name="--context-length",
            env_var="DYN_GLOBAL_ROUTER_CONTEXT_LENGTH",
            default=None,
            arg_type=int,
            help="Advertised maximum context length. Must not exceed the routed workers' limit.",
        )
        add_argument(
            g,
            flag_name="--reasoning-parser",
            env_var="DYN_GLOBAL_ROUTER_REASONING_PARSER",
            default=None,
            help="Dynamo reasoning parser advertised to frontends (for example deepseek_v4).",
        )
        add_argument(
            g,
            flag_name="--tool-call-parser",
            env_var="DYN_GLOBAL_ROUTER_TOOL_CALL_PARSER",
            default=None,
            help="Dynamo tool-call parser advertised to frontends (for example deepseek_v4).",
        )
        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            help="Dynamo namespace for the global router.",
        )
        add_argument(
            g,
            flag_name="--component-name",
            env_var="DYN_GLOBAL_ROUTER_COMPONENT_NAME",
            default="global_router",
            help="Component name for the global router.",
        )
        add_argument(
            g,
            flag_name="--default-ttft-target-ms",
            obsolete_flag="--default-ttft-target",
            env_var="DYN_GLOBAL_ROUTER_DEFAULT_TTFT_TARGET_MS",
            default=env_or_default(
                "DYN_GLOBAL_ROUTER_DEFAULT_TTFT_TARGET", None, value_type=float
            ),
            help="Default TTFT target (ms) for prefill pool selection when SLA not present in request.",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--default-itl-target-ms",
            obsolete_flag="--default-itl-target",
            env_var="DYN_GLOBAL_ROUTER_DEFAULT_ITL_TARGET_MS",
            default=env_or_default(
                "DYN_GLOBAL_ROUTER_DEFAULT_ITL_TARGET", None, value_type=float
            ),
            help="Default ITL target (ms) for decode pool selection when SLA not present in request.",
            arg_type=float,
        )


class DynamoGlobalRouterConfig(ConfigBase):
    """Configuration for Dynamo Global Router (CLI/env-backed)."""

    config_path: Optional[str] = None
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    revision: Optional[str] = None
    kv_cache_block_size: Optional[int] = None
    context_length: Optional[int] = None
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    namespace: str
    component_name: str
    default_ttft_target_ms: Optional[float] = None
    default_itl_target_ms: Optional[float] = None

    def validate(self) -> None:
        """Require config_path and model_name to be set via CLI or env."""
        if not self.config_path or not self.config_path.strip():
            raise ValueError(
                "config_path must be set via --config or DYN_GLOBAL_ROUTER_CONFIG"
            )
        if not self.model_name or not self.model_name.strip():
            raise ValueError(
                "model_name must be set via --model-name or DYN_GLOBAL_ROUTER_MODEL_NAME"
            )
        for field, value in (
            ("model_path", self.model_path),
            ("revision", self.revision),
            ("reasoning_parser", self.reasoning_parser),
            ("tool_call_parser", self.tool_call_parser),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field} must not be empty")
        for field, value in (
            ("kv_cache_block_size", self.kv_cache_block_size),
            ("context_length", self.context_length),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be positive")
