# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import WorkerMetricsPublisher

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """These synchronous API tests do not create a runtime or an event endpoint."""
    yield


def test_decode_snapshot_api_is_independent_of_kv_publish():
    publisher = WorkerMetricsPublisher()
    report = dict(
        num_running_reqs=1,
        num_waiting_reqs=0,
        observation_revision=1,
        observed_at_unix_ms=1000,
    )
    publisher.publish_decode_metrics(tokens_per_user_second=50.5, **report)
    publisher.publish(dp_rank=0, kv_used_blocks=10, num_waiting_reqs=1)
    publisher.publish_decode_metrics(dp_rank=4, tokens_per_user_second=12, **report)
    publisher.publish_decode_metrics(
        dp_rank=4,
        num_running_reqs=0,
        num_waiting_reqs=0,
        observation_revision=2,
        observed_at_unix_ms=1250,
    )


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), -1.0])
def test_decode_snapshot_api_rejects_invalid_speed(speed):
    with pytest.raises(Exception, match="invalid decode metrics observation"):
        WorkerMetricsPublisher().publish_decode_metrics(
            tokens_per_user_second=speed,
            num_running_reqs=1,
            num_waiting_reqs=0,
            observation_revision=1,
            observed_at_unix_ms=1000,
        )
