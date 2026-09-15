# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import fpm_dp_rank_bounds

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def ranks(**kwargs):
    args = dict(
        dp_size=16,
        tp_size=16,
        pp_size=1,
        nnodes=4,
        node_rank=0,
        enable_dp_attention=True,
    )
    args.update(kwargs)
    return fpm_dp_rank_bounds(SimpleNamespace(**args))


def test_wide_ep_decode_covers_each_dp_rank_once():
    assert [ranks(node_rank=n) for n in range(4)] == [(0, 4), (4, 8), (8, 12), (12, 16)]


def test_pipeline_prefill_metrics_come_from_all_last_stage_ranks():
    assert [ranks(node_rank=n, dp_size=4, tp_size=4, pp_size=4) for n in range(4)] == [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 4),
    ]


def test_pipeline_and_tensor_parallel_topology():
    assert [ranks(node_rank=n, dp_size=4, tp_size=8, pp_size=2) for n in range(4)] == [
        (0, 0),
        (0, 0),
        (0, 2),
        (2, 4),
    ]


def test_attention_tp_spanning_nodes_publishes_only_its_leader():
    assert [ranks(node_rank=n, dp_size=1, tp_size=16) for n in range(4)] == [
        (0, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]


def test_multiple_pipeline_stages_per_node():
    assert [
        ranks(node_rank=n, nnodes=2, pp_size=4, dp_size=4, tp_size=4) for n in range(2)
    ] == [(0, 0), (0, 4)]
