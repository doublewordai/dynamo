# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from dynamo.frontend import hf_metadata

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.asyncio
async def test_resolve_hf_metadata_uses_hub_download(monkeypatch, tmp_path: Path):
    downloaded = tmp_path / "tokenizer.json"
    downloaded.write_text("{}")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(hf_metadata, "hf_hub_download", fake_download)

    result = await hf_metadata.resolve_hf_metadata(
        "Qwen/Qwen3.5-397B-A17B-FP8@revision-1",
        "tokenizer.json",
    )

    assert result == downloaded
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3.5-397B-A17B-FP8",
            "filename": "tokenizer.json",
            "revision": "revision-1",
        }
    ]


def test_repo_without_revision_uses_default_branch():
    assert hf_metadata._repo_and_revision("zai-org/GLM-5.2-FP8") == (
        "zai-org/GLM-5.2-FP8",
        None,
    )
