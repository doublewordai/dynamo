# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang", reason="sglang not installed in this container")

from dynamo.sglang.gateway import reject_unreported_admission_margin  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_margin_is_refused_on_workers_that_do_not_report_their_queue(monkeypatch):
    llm = SimpleNamespace(multimodal_worker=False, embedding_worker=False)
    multimodal = SimpleNamespace(multimodal_worker=True)

    monkeypatch.delenv("DYN_ADMISSION_QUEUE_MARGIN", raising=False)
    reject_unreported_admission_margin(multimodal)

    monkeypatch.setenv("DYN_ADMISSION_QUEUE_MARGIN", "16")
    reject_unreported_admission_margin(llm)
    with pytest.raises(ValueError, match="multimodal-worker"):
        reject_unreported_admission_margin(multimodal)
