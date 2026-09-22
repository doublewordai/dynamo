# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A durable GPU allocation supplied by the fleet allocator."""

import json
from dataclasses import dataclass
from typing import Optional

from dynamo.planner.errors import DeploymentValidationError

GPU_BUDGET_ANNOTATION = "doubleword.ai/planner-gpu-budget"


@dataclass(frozen=True)
class GpuBudget:
    min_gpus: int
    max_gpus: int


def gpu_budget_from_deployment(deployment: dict) -> Optional[GpuBudget]:
    annotations = deployment.get("metadata", {}).get("annotations", {}) or {}
    raw = annotations.get(GPU_BUDGET_ANNOTATION)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"min_gpus", "max_gpus"}:
            raise ValueError("expected min_gpus and max_gpus")
        low, high = value["min_gpus"], value["max_gpus"]
        if type(low) is not int or type(high) is not int or not 0 <= low <= high:
            raise ValueError("expected integer bounds with 0 <= min_gpus <= max_gpus")
        return GpuBudget(low, high)
    except (TypeError, ValueError) as exc:
        raise DeploymentValidationError(
            [f"Invalid {GPU_BUDGET_ANNOTATION}: {exc}"]
        ) from exc
