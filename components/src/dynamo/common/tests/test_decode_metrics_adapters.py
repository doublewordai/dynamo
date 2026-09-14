# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import dynamo.common.decode_metrics as metrics
from dynamo.vllm.decode_metrics import observe_decode_output

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def outputs(*deltas):
    return {
        0: SimpleNamespace(
            outputs=[
                SimpleNamespace(request_id=key, new_token_ids=list(range(n)))
                for key, n in deltas
            ]
        )
    }


def test_vllm_committed_outputs_include_speculative_tokens_and_preemption(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(metrics.time, "monotonic", lambda: now[0])
    tracker = metrics.DecodeMetricsTracker()
    observe_decode_output(tracker, outputs(("a", 1)), {"a": object()}, 1)
    now[0] = 0.25
    report = observe_decode_output(
        tracker, outputs(("a", 3), ("a", 2)), {"a": object()}, 1
    )
    assert report["tokens_per_user_second"] == 20
    now[0] = 2
    report = observe_decode_output(tracker, {}, {"a": object()}, 0)
    assert report["tokens_per_user_second"] == 0
    assert report["num_waiting_reqs"] == 1
    now[0] = 2.25
    report = observe_decode_output(tracker, outputs(("a", 4)), {}, 0)
    assert report["tokens_per_user_second"] is None
    assert report["num_running_reqs"] == report["num_waiting_reqs"] == 0
