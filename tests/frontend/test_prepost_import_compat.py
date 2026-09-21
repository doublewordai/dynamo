# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dynamo.frontend.prepost`` must import on both vLLM protocol layouts.

vLLM 0.29 moved the streaming delta models from
``vllm.entrypoints.openai.engine.protocol`` to
``vllm.entrypoints.generate.base.protocol``. Each test installs a stand-in
``vllm`` module tree shaped like one layout and imports prepost against it.
"""

import importlib
import sys
from types import ModuleType

import pytest

# No framework marker: the stand-in modules make this run without vLLM.
pytestmark = [
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

_OLD_PROTOCOL = "vllm.entrypoints.openai.engine.protocol"
_NEW_PROTOCOL = "vllm.entrypoints.generate.base.protocol"
_DELTA_NAMES = ("DeltaFunctionCall", "DeltaMessage", "DeltaToolCall")

# Every other vLLM name prepost imports; identical in both layouts.
_COMMON_MODULES = {
    "vllm.entrypoints.chat_utils": ("make_tool_call_id",),
    "vllm.entrypoints.openai.chat_completion.protocol": (
        "ChatCompletionNamedToolChoiceParam",
        "ChatCompletionRequest",
    ),
    "vllm.reasoning": ("ReasoningParser",),
    "vllm.renderers": ("ChatParams", "merge_kwargs"),
    "vllm.sampling_params": ("SamplingParams",),
    "vllm.tokenizers": ("TokenizerLike",),
    "vllm.tool_parsers": ("ToolParser",),
    "vllm.tool_parsers.utils": ("get_json_schema_from_tools",),
    "vllm.utils.async_utils": ("make_async",),
}


def _import_prepost(monkeypatch, protocol_module: str) -> tuple[ModuleType, dict]:
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "dynamo.frontend.prepost", raising=False)

    modules = {name: list(attrs) for name, attrs in _COMMON_MODULES.items()}
    modules[protocol_module] = list(_DELTA_NAMES)

    delta_types = {}
    for name, attrs in modules.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            qualified = ".".join(parts[:i])
            if qualified not in sys.modules:
                monkeypatch.setitem(sys.modules, qualified, ModuleType(qualified))
        for attr in attrs:
            value = type(attr, (), {})
            setattr(sys.modules[name], attr, value)
            if name == protocol_module:
                delta_types[attr] = value

    prepost = importlib.import_module("dynamo.frontend.prepost")
    # Do not leave a prepost bound to stand-in modules behind for other tests.
    monkeypatch.delitem(sys.modules, "dynamo.frontend.prepost", raising=False)
    return prepost, delta_types


@pytest.mark.parametrize("protocol_module", [_OLD_PROTOCOL, _NEW_PROTOCOL])
def test_prepost_imports_delta_models_from_either_layout(monkeypatch, protocol_module):
    prepost, delta_types = _import_prepost(monkeypatch, protocol_module)

    for name in _DELTA_NAMES:
        assert getattr(prepost, name) is delta_types[name]
