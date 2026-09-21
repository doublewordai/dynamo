# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise ``dynamo.sglang._compat`` against both supported SGLang layouts.

Each test installs a stand-in ``sglang`` module tree shaped like one release
and loads a private copy of the compat module against it, so both branches of
every shim run regardless of which SGLang (if any) is installed.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_COMPAT_PATH = Path(__file__).resolve().parents[1] / "_compat.py"


def _install_fake_sglang(monkeypatch, modules: dict[str, dict]) -> None:
    """Replace every ``sglang`` module with the given stand-in tree."""
    for name in [m for m in sys.modules if m == "sglang" or m.startswith("sglang.")]:
        monkeypatch.delitem(sys.modules, name)

    tree: dict[str, dict] = {"sglang": {"Engine": object, "ServerArgs": object}}
    for name, attrs in modules.items():
        parts = name.split(".")
        for i in range(1, len(parts)):
            tree.setdefault(".".join(parts[:i]), {})
        tree.setdefault(name, {}).update(attrs)

    for name, attrs in tree.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    for name in tree:
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, sys.modules[name])


def _load_compat(monkeypatch, modules: dict[str, dict]) -> ModuleType:
    _install_fake_sglang(monkeypatch, modules)
    spec = importlib.util.spec_from_file_location(
        "_dynamo_sglang_compat_under_test", _COMPAT_PATH
    )
    assert spec is not None and spec.loader is not None
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    return compat


def _old_layout() -> dict[str, dict]:
    """SGLang 0.5.18: accessors are ServerArgs methods."""
    return {
        "sglang.srt.server_args_config_parser": {"ConfigArgumentMerger": object},
    }


def _new_layout(**model_override_base) -> dict[str, dict]:
    """SGLang 0.5.19+: accessors are module-level functions."""
    return {
        "sglang.srt.utils.server_args_config_parser": {"ConfigArgumentMerger": object},
        "sglang.srt.arg_groups.model_override_base": {
            "model_config_of": lambda server_args: None,
            "use_mla_backend": lambda server_args: False,
            **model_override_base,
        },
    }


@pytest.mark.parametrize("uses_mla", [True, False])
def test_mla_backend_old_layout_uses_server_args_method(monkeypatch, uses_mla):
    compat = _load_compat(monkeypatch, _old_layout())
    server_args = SimpleNamespace(use_mla_backend=lambda: uses_mla)

    assert compat.sglang_uses_mla_backend(server_args) is uses_mla


@pytest.mark.parametrize("uses_mla", [True, False])
def test_mla_backend_new_layout_uses_module_function(monkeypatch, uses_mla):
    seen = []

    def use_mla_backend(server_args):
        seen.append(server_args)
        return uses_mla

    compat = _load_compat(monkeypatch, _new_layout(use_mla_backend=use_mla_backend))
    server_args = SimpleNamespace()

    assert compat.sglang_uses_mla_backend(server_args) is uses_mla
    assert seen == [server_args]


def test_mla_backend_without_any_accessor_raises(monkeypatch):
    compat = _load_compat(monkeypatch, _old_layout())

    with pytest.raises(AttributeError, match="MLA backend accessor"):
        compat.sglang_uses_mla_backend(SimpleNamespace())
