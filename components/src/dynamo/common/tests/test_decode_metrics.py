# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.decode_metrics import DecodeMetricsTracker
from dynamo.common.forward_pass_metrics import ForwardPassMetrics, decode, encode

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def observe(tracker, now, outputs, live, running=None, waiting=0):
    return tracker.observe(
        outputs,
        live,
        len(live) if running is None else running,
        waiting,
        now=now,
        unix_ms=1_000_000 + int(now * 1000),
    )


def test_prefill_first_output_and_speculative_acceptance():
    tracker = DecodeMetricsTracker()
    assert observe(tracker, 0, {}, ["a"])["tokens_per_user_second"] is None
    assert observe(tracker, 10, {"a": 4}, ["a"])["tokens_per_user_second"] is None
    assert observe(tracker, 10.25, {"a": 5}, ["a"])["tokens_per_user_second"] == 20
    assert observe(tracker, 10.5, {"a": 10}, ["a"])["tokens_per_user_second"] == 30


def test_sequence_seconds_with_new_requests_and_prefill_interference():
    tracker = DecodeMetricsTracker()
    observe(tracker, 0, {"a": 1}, ["a"])
    assert (
        observe(tracker, 0.25, {"a": 5, "b": 1}, ["a", "b"])["tokens_per_user_second"]
        == 20
    )
    assert (
        observe(tracker, 0.5, {"a": 5, "b": 5}, ["a", "b"])["tokens_per_user_second"]
        == 20
    )
    # A prefill-only interval stalls both already-decoding sequences.
    assert observe(tracker, 1, {}, ["a", "b", "c"])[
        "tokens_per_user_second"
    ] == pytest.approx(15 / 1.75)


def test_publication_rate_limit_does_not_discard_tokens():
    tracker = DecodeMetricsTracker()
    first = observe(tracker, 0, {"a": 1}, ["a"])
    assert observe(tracker, 0.125, {"a": 4}, ["a"]) is None
    second = observe(tracker, 0.25, {"a": 4}, ["a"])
    assert second["tokens_per_user_second"] == 32
    assert second["observation_revision"] == first["observation_revision"] + 1
    assert second["observed_at_unix_ms"] == 1_000_250


def test_stalled_or_preempted_decode_is_zero_and_never_idle():
    tracker = DecodeMetricsTracker()
    observe(tracker, 0, {"a": 1}, ["a"])
    report = observe(tracker, 2, {}, ["a"], running=0, waiting=1)
    assert report["tokens_per_user_second"] == 0
    assert report["num_waiting_reqs"] == 1


def test_completion_cancellation_and_new_probe_clear_old_window():
    tracker = DecodeMetricsTracker()
    observe(tracker, 0, {"a": 1, "b": 1}, ["a", "b"])
    # a completes, b was cancelled. Count the last output before removing a.
    report = observe(tracker, 0.25, {"a": 10}, [])
    assert report["tokens_per_user_second"] is None
    assert report["num_running_reqs"] == 0
    observe(tracker, 10, {"c": 1}, ["c"])
    assert observe(tracker, 10.25, {"c": 2}, ["c"])["tokens_per_user_second"] == 8
    assert not tracker._decoding - {"c"}


def test_wire_roundtrip_and_legacy_heartbeat_does_not_have_decode_observation():
    tracker = DecodeMetricsTracker()
    report = observe(tracker, 0, {}, [])
    message = ForwardPassMetrics(worker_id="42", dp_rank=7, decode_metrics=report)
    assert decode(encode(message)).decode_metrics == report
    assert decode(encode(ForwardPassMetrics())).decode_metrics is None


def test_monotonic_time_and_rolling_window():
    tracker = DecodeMetricsTracker()
    observe(tracker, 0, {"a": 1}, ["a"])
    observe(tracker, 0.25, {"a": 25}, ["a"])
    observe(tracker, 0.5, {"a": 5}, ["a"])
    assert observe(tracker, 1.5, {"a": 20}, ["a"])["tokens_per_user_second"] == 20
    with pytest.raises(ValueError):
        observe(tracker, 1, {}, ["a"])


def test_rate_limited_idle_observation_clears_previous_busy_period():
    tracker = DecodeMetricsTracker()
    observe(tracker, 0, {"a": 1}, ["a"])
    observe(tracker, 0.25, {"a": 100}, ["a"])
    assert observe(tracker, 0.375, {}, []) is None
    assert observe(tracker, 0.5, {"b": 1}, ["b"])["tokens_per_user_second"] is None
    assert observe(tracker, 0.75, {"b": 2}, ["b"])["tokens_per_user_second"] == 8
