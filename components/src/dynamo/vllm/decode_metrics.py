# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate committed vLLM output deltas into a scheduler observation."""

from dynamo.common.decode_metrics import DecodeMetricsTracker


def observe_decode_output(
    tracker: DecodeMetricsTracker, engine_outputs, requests, num_running_reqs: int
) -> dict | None:
    output_tokens: dict[str, int] = {}
    for engine_output in (engine_outputs or {}).values():
        for output in engine_output.outputs:
            key = output.request_id
            output_tokens[key] = output_tokens.get(key, 0) + len(output.new_token_ids)
    return tracker.observe(
        output_tokens,
        requests.keys(),
        num_running_reqs,
        len(requests) - num_running_reqs,
    )
