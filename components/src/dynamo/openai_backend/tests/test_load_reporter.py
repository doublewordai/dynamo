# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from dynamo.openai_backend.load_reporter import (
    MAX_NUM_BATCHED_TOKENS_ENV_VAR,
    MAX_NUM_SEQS_ENV_VAR,
    EngineCapacity,
    capacity_from_server_info,
    capacity_from_vllm_metrics,
    parse_load_samples,
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


def _vllm_cache_config_info(*num_gpu_blocks: int) -> str:
    lines = [
        "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig",
        "# TYPE vllm:cache_config_info gauge",
    ]
    for engine, blocks in enumerate(num_gpu_blocks):
        lines.append(
            f'vllm:cache_config_info{{block_size="16",engine="{engine}",'
            f'num_gpu_blocks="{blocks}"}} 1.0'
        )
    return "\n".join(lines) + "\n"


def test_vllm_single_rank_capacity():
    capacity = capacity_from_vllm_metrics(_vllm_cache_config_info(164_000))
    assert capacity == EngineCapacity(
        total_kv_blocks=164_000,
        max_num_batched_tokens=None,
        data_parallel_size=1,
        max_num_seqs=None,
    )


def test_vllm_equal_ranks_capacity():
    capacity = capacity_from_vllm_metrics(_vllm_cache_config_info(164_000, 164_000))
    assert capacity is not None
    assert capacity.total_kv_blocks == 164_000
    assert capacity.data_parallel_size == 2


def test_vllm_unequal_ranks_use_minimum_and_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="dynamo.openai_backend.load_reporter"):
        capacity = capacity_from_vllm_metrics(_vllm_cache_config_info(164_007, 164_000))
    assert capacity is not None
    assert capacity.total_kv_blocks == 164_000
    assert capacity.data_parallel_size == 2
    assert "unequal num_gpu_blocks" in caplog.text


def test_vllm_missing_cache_config_info_yields_none():
    assert (
        capacity_from_vllm_metrics('vllm:num_requests_running{engine="0"} 0.0\n')
        is None
    )


def test_vllm_limits_come_from_env(monkeypatch):
    monkeypatch.setenv(MAX_NUM_SEQS_ENV_VAR, "256")
    monkeypatch.setenv(MAX_NUM_BATCHED_TOKENS_ENV_VAR, "8192")
    capacity = capacity_from_vllm_metrics(_vllm_cache_config_info(164_000))
    assert capacity is not None
    assert capacity.max_num_seqs == 256
    assert capacity.max_num_batched_tokens == 8192


def test_engine_label_maps_to_dp_rank():
    samples = parse_load_samples(
        'vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
        'vllm:kv_cache_usage_perc{engine="1"} 0.5\n'
        'vllm:num_requests_waiting{engine="0"} 3.0\n'
        'vllm:num_requests_waiting{engine="1"} 4.0\n'
    )
    assert samples is not None
    assert (samples[0].usage_frac, samples[0].waiting) == (0.25, 3.0)
    assert (samples[1].usage_frac, samples[1].waiting) == (0.5, 4.0)
