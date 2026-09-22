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


# ---------------------------------------------------------------------------
# Multimodal encoder: module move and encode API change (SGLang 0.5.19+)
# ---------------------------------------------------------------------------

_OLD_ENCODER_MODULE = "sglang.srt.disaggregation.encode_server"
_NEW_ENCODER_MODULE = "sglang.srt.disaggregation.encoder.server"
_NEW_PREPROCESSOR_MODULE = "sglang.srt.disaggregation.encoder.preprocessor"


def test_mm_encoder_class_old_layout(monkeypatch):
    old_encoder = type("MMEncoder", (), {})
    compat = _load_compat(
        monkeypatch, {**_old_layout(), _OLD_ENCODER_MODULE: {"MMEncoder": old_encoder}}
    )

    assert compat.get_mm_encoder_class() is old_encoder


def test_mm_encoder_class_new_layout(monkeypatch):
    new_encoder = type("MMEncoder", (), {})
    compat = _load_compat(
        monkeypatch, {**_new_layout(), _NEW_ENCODER_MODULE: {"MMEncoder": new_encoder}}
    )

    assert compat.get_mm_encoder_class() is new_encoder


def test_mm_encoder_class_missing_raises_import_error(monkeypatch):
    compat = _load_compat(monkeypatch, _old_layout())

    with pytest.raises(ImportError):
        compat.get_mm_encoder_class()


def test_encoder_preprocessor_modules_old_layout(monkeypatch):
    compat = _load_compat(
        monkeypatch, {**_old_layout(), _OLD_ENCODER_MODULE: {"load_video": object()}}
    )

    modules = compat.get_encoder_preprocessor_modules()

    assert [m.__name__ for m in modules] == [_OLD_ENCODER_MODULE]


def test_encoder_preprocessor_modules_new_layout(monkeypatch):
    compat = _load_compat(
        monkeypatch,
        {**_new_layout(), _NEW_PREPROCESSOR_MODULE: {"load_video": object()}},
    )

    modules = compat.get_encoder_preprocessor_modules()

    assert [m.__name__ for m in modules] == [_NEW_PREPROCESSOR_MODULE]


async def test_mm_encode_old_api_calls_encode(monkeypatch):
    compat = _load_compat(monkeypatch, _old_layout())
    expected = ([[1, 2, 2]], object(), {"aux": 1})
    calls = []

    class Encoder:
        async def _encode(self, mm_items, modality):
            calls.append((mm_items, modality))
            return expected

    assert await compat.mm_encode(Encoder(), ["img"], "IMAGE") == expected
    assert calls == [(["img"], "IMAGE")]


async def test_mm_encode_new_api_prepares_then_computes(monkeypatch):
    compat = _load_compat(monkeypatch, _new_layout())
    embeddings = object()
    context = SimpleNamespace(
        preprocess_result=SimpleNamespace(grid_thw=[[1, 2, 2]]),
        aux_data={"aux": 1},
    )
    prepared = []

    class Encoder:
        # Mirrors the SGLang 0.5.20 signatures, keyword-only flags included.
        async def _prepare_encode_context(
            self, requests, modality, *, use_global_cache, is_health_check=False
        ):
            prepared.append((requests, modality, use_global_cache))
            return context

        async def _compute_embedding(self, ctx, *, keep_on_gpu):
            assert ctx is context
            assert keep_on_gpu is False
            return embeddings

    grid, result, aux = await compat.mm_encode(Encoder(), ["img"], "IMAGE")

    assert (grid, result, aux) == ([[1, 2, 2]], embeddings, {"aux": 1})
    [(requests, modality, use_global_cache)] = prepared
    assert modality == "IMAGE"
    assert use_global_cache is False
    [request] = requests
    assert request["mm_items"] == ["img"]
    assert request["req_id"].startswith("dynamo-direct-")


async def test_mm_encode_new_api_without_embeddings_raises(monkeypatch):
    compat = _load_compat(monkeypatch, _new_layout())

    class Encoder:
        async def _prepare_encode_context(self, requests, modality, **kwargs):
            return SimpleNamespace()

        async def _compute_embedding(self, ctx, **kwargs):
            return None

    with pytest.raises(RuntimeError, match="no embeddings"):
        await compat.mm_encode(Encoder(), ["img"], "IMAGE")


async def test_mm_encode_without_any_api_raises(monkeypatch):
    compat = _load_compat(monkeypatch, _new_layout())

    with pytest.raises(RuntimeError, match="encode API"):
        await compat.mm_encode(object(), ["img"], "IMAGE")


def test_mm_encoder_vision_config_old_layout_reads_encoder(monkeypatch):
    compat = _load_compat(monkeypatch, _old_layout())
    encoder = SimpleNamespace(vision_config={"video": {"fps": 2.0}})

    assert compat.mm_encoder_vision_config(encoder) == {"video": {"fps": 2.0}}


def test_mm_encoder_vision_config_new_layout_reads_preprocessor(monkeypatch):
    compat = _load_compat(monkeypatch, _new_layout())
    encoder = SimpleNamespace(
        preprocessor=SimpleNamespace(vision_config={"video": {"fps": 4.0}})
    )

    assert compat.mm_encoder_vision_config(encoder) == {"video": {"fps": 4.0}}
    assert compat.mm_encoder_vision_config(SimpleNamespace()) is None


def test_publish_server_args_uses_runtime_context(monkeypatch):
    calls = []
    layout = _new_layout()
    layout["sglang.srt.runtime_context"] = {
        "publish": lambda server_args, *, role: calls.append((server_args, role))
    }
    compat = _load_compat(monkeypatch, layout)
    server_args = SimpleNamespace()

    compat.publish_server_args(server_args, role="encoder")

    assert calls == [(server_args, "encoder")]


def test_publish_server_args_without_runtime_context_is_noop(monkeypatch):
    compat = _load_compat(monkeypatch, _old_layout())

    compat.publish_server_args(SimpleNamespace(), role="encoder")
