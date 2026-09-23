# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU allocations published on the DGD by an external fleet allocator.

The allocation is the ``doubleword.ai/planner-gpu-budget`` annotation,
``{"min_gpus": N, "max_gpus": M}``. The planner applies it as its runtime GPU
budget band. Because the allocation can change while a deployment is idle or
has no workers at all, the planner also moves held replicas under the new
ceiling and, once replica counts are stable, up to the endpoint floors, without
waiting for traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from dynamo.planner.core.budget import (
    proportional_clamp_pair,
    proportional_clamp_single,
)

GPU_BUDGET_ANNOTATION = "doubleword.ai/planner-gpu-budget"


@dataclass(frozen=True)
class GpuBudget:
    min_gpus: int
    max_gpus: int


def gpu_budget_from_deployment(deployment: dict) -> Optional[GpuBudget]:
    """Return the DGD's allocation, None when absent; ValueError when malformed."""
    annotations = deployment.get("metadata", {}).get("annotations") or {}
    raw = annotations.get(GPU_BUDGET_ANNOTATION)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{GPU_BUDGET_ANNOTATION} is not JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"min_gpus", "max_gpus"}:
        raise ValueError(f"{GPU_BUDGET_ANNOTATION} needs min_gpus and max_gpus")
    low, high = value["min_gpus"], value["max_gpus"]
    if type(low) is not int or type(high) is not int or not 0 <= low <= high:
        raise ValueError(
            f"{GPU_BUDGET_ANNOTATION} needs integers 0 <= min_gpus <= max_gpus"
        )
    return GpuBudget(low, high)


def fleet_reconcile_targets(
    held: tuple[Optional[int], Optional[int]],
    serving: tuple[Optional[int], Optional[int]],
    gpu_cost: tuple[Optional[int], Optional[int]],
    floors: tuple[Optional[int], Optional[int]],
    max_gpus: int,
    stable: bool,
) -> Optional[tuple[Optional[int], Optional[int]]]:
    """Return ``(prefill, decode)`` targets, or None when nothing changes.

    ``held`` counts serving plus verified pending replicas; ``None`` entries
    are roles this planner does not manage, and a role whose count stays is
    returned as ``None``. When ``stable``, held counts are raised to the
    endpoint floors; they are then fitted under the ceiling with the clamps the
    planner applies to its own decisions (a ceiling below the minimum footprint
    yields zero). While counts are moving, only reductions are returned, capped
    at the serving count so pending replicas are cancelled first, which is the
    only change the Kubernetes connector admits during startup.
    """
    (held_p, held_d), (p_gpu, d_gpu), (p_floor, d_floor) = held, gpu_cost, floors

    def wanted(count: int, floor: Optional[int]) -> int:
        return max(count, floor or 0) if stable else count

    if held_p is not None and held_d is not None:
        if p_gpu is None or d_gpu is None:
            return None
        fitted: tuple[Optional[int], Optional[int]] = proportional_clamp_pair(
            wanted(held_p, p_floor),
            wanted(held_d, d_floor),
            p_gpu,
            d_gpu,
            -1,
            max_gpus,
            p_floor or 0,
            d_floor or 0,
        )
    elif held_p is not None or held_d is not None:
        count, gpu, floor = (
            (held_p, p_gpu, p_floor) if held_p is not None else (held_d, d_gpu, d_floor)
        )
        assert count is not None
        if gpu is None:
            return None
        single = proportional_clamp_single(
            wanted(count, floor), gpu, -1, max_gpus, floor or 0
        )
        fitted = (single, None) if held_p is not None else (None, single)
    else:
        return None

    def role(
        target: Optional[int], count: Optional[int], ready: Optional[int]
    ) -> Optional[int]:
        if target is None or count is None or target == count:
            return None
        if stable:
            return target
        if target > count:
            return None
        return min(target, ready or 0)

    result = (
        role(fitted[0], held_p, serving[0]),
        role(fitted[1], held_d, serving[1]),
    )
    return None if result == (None, None) else result
