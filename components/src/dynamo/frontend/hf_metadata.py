# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face metadata downloads for the Python frontend."""

import asyncio
from pathlib import Path

from huggingface_hub import hf_hub_download


def _repo_and_revision(repo: str) -> tuple[str, str | None]:
    """Split Dynamo's ``repo[@revision]`` representation."""
    repo_id, separator, revision = repo.rpartition("@")
    if separator and repo_id and revision:
        return repo_id, revision
    return repo, None


async def resolve_hf_metadata(repo: str, filename: str) -> Path:
    """Download one model metadata file through huggingface_hub/hf_xet."""
    repo_id, revision = _repo_and_revision(repo)
    path = await asyncio.to_thread(
        hf_hub_download,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    return Path(path)
