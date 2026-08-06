# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.openai_backend.load_reporter import (
    EngineCapacity,
    capacity_from_server_info,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


SERVER_INFO = {
    "max_total_num_tokens": 1024,
    "max_prefill_tokens": 512,
    "page_size": 16,
    "dp_size": 2,
}


def test_prefers_effective_max_running_requests_per_dp():
    info = {
        **SERVER_INFO,
        "max_running_requests": 999,
        "internal_states": [
            {"effective_max_running_requests_per_dp": 256},
            {"effective_max_running_requests_per_dp": 256},
        ],
    }
    capacity = capacity_from_server_info(info)
    assert capacity is not None
    assert capacity.max_num_seqs == 256


def test_falls_back_to_max_running_requests_server_arg():
    info = {**SERVER_INFO, "max_running_requests": 128, "internal_states": [{}]}
    capacity = capacity_from_server_info(info)
    assert capacity is not None
    assert capacity.max_num_seqs == 128


def test_missing_running_limit_yields_none():
    capacity = capacity_from_server_info(dict(SERVER_INFO))
    assert capacity is not None
    assert capacity.max_num_seqs is None


def test_unset_server_arg_is_ignored():
    info = {**SERVER_INFO, "max_running_requests": None}
    capacity = capacity_from_server_info(info)
    assert capacity is not None
    assert capacity.max_num_seqs is None


def test_kv_fields_parse_alongside():
    capacity = capacity_from_server_info(dict(SERVER_INFO))
    assert capacity == EngineCapacity(
        total_kv_blocks=64,
        max_num_batched_tokens=512,
        data_parallel_size=2,
        max_num_seqs=None,
    )


def test_non_dict_payload_is_rejected():
    assert capacity_from_server_info(["not", "a", "dict"]) is None
