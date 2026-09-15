# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.planner.config.gpu_budget import (
    GPU_BUDGET_ANNOTATION,
    GpuBudget,
    gpu_budget_from_deployment,
)
from dynamo.planner.errors import DeploymentValidationError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"min_gpus": 32},
        {"min_gpus": -1, "max_gpus": 32},
        {"min_gpus": 48, "max_gpus": 32},
        {"min_gpus": True, "max_gpus": 32},
        {"min_gpus": 0, "max_gpus": 32.0},
        {"min_gpus": 0, "max_gpus": 32, "extra": 1},
    ],
)
def test_malformed_budget_is_rejected(value):
    with pytest.raises(DeploymentValidationError):
        gpu_budget_from_deployment(
            {"metadata": {"annotations": {GPU_BUDGET_ANNOTATION: json.dumps(value)}}}
        )


@pytest.mark.parametrize("low,high", [(0, 0), (32, 48), (32, 32)])
def test_budget_bounds(low, high):
    assert gpu_budget_from_deployment(
        {
            "metadata": {
                "annotations": {
                    GPU_BUDGET_ANNOTATION: json.dumps(
                        {"min_gpus": low, "max_gpus": high}
                    )
                }
            }
        }
    ) == GpuBudget(low, high)


def test_absent_budget_is_optional():
    assert gpu_budget_from_deployment({}) is None
