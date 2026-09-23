# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.adapters import DisaggPlanner
from dynamo.planner.core.fleet_gpu_budget import (
    GPU_BUDGET_ANNOTATION,
    GpuBudget,
    fleet_reconcile_targets,
    gpu_budget_from_deployment,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _deployment(value):
    annotations = {} if value is None else {GPU_BUDGET_ANNOTATION: value}
    return {"metadata": {"annotations": annotations}}


def test_allocation_parses_from_the_deployment_annotation():
    assert gpu_budget_from_deployment(_deployment(None)) is None
    assert gpu_budget_from_deployment(
        _deployment(json.dumps({"min_gpus": 16, "max_gpus": 48}))
    ) == GpuBudget(16, 48)
    assert gpu_budget_from_deployment(
        _deployment('{"min_gpus":0,"max_gpus":0}')
    ) == GpuBudget(0, 0)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"max_gpus": 8}',
        '{"min_gpus": 8, "max_gpus": 4}',
        '{"min_gpus": -1, "max_gpus": 4}',
        '{"min_gpus": 1.5, "max_gpus": 4}',
        '{"min_gpus": 0, "max_gpus": 4, "extra": 1}',
    ],
)
def test_malformed_allocation_is_rejected(raw):
    with pytest.raises(ValueError, match=GPU_BUDGET_ANNOTATION):
        gpu_budget_from_deployment(_deployment(raw))


@pytest.mark.parametrize(
    "held, serving, max_gpus, stable, expected",
    [
        # A zero allocation, or one below the minimum pair, holds nothing.
        ((2, 2), (2, 2), 0, True, (0, 0)),
        ((2, 2), (2, 2), 4, True, (0, 0)),
        # Asleep and stable: wake to the endpoint floors.
        ((0, 0), (0, 0), 16, True, (1, 1)),
        # Over the ceiling: shrink.
        ((4, 4), (4, 4), 16, True, (2, 2)),
        # Within the band: nothing to do.
        ((2, 2), (2, 2), 16, True, None),
        # Starting up: never wake, and reductions stop at the serving count
        # so pending replicas are cancelled first.
        ((0, 1), (0, 0), 16, False, None),
        ((4, 1), (1, 1), 8, False, (1, None)),
        ((2, 2), (1, 1), 0, False, (0, 0)),
    ],
)
def test_disagg_reconcile_targets(held, serving, max_gpus, stable, expected):
    assert (
        fleet_reconcile_targets(held, serving, (4, 4), (1, 1), max_gpus, stable)
        == expected
    )


def test_single_role_reconcile_uses_the_resolved_replica_cost():
    # Two-node replicas cost 16 GPUs each.
    def target(held, max_gpus, stable=True, serving=None):
        return fleet_reconcile_targets(
            (None, held),
            (None, held if serving is None else serving),
            (None, 16),
            (None, 1),
            max_gpus,
            stable,
        )

    assert target(3, 32) == (None, 2)
    assert target(0, 32) == (None, 1)
    assert target(0, 8) is None
    # One serving plus three pending under a two-replica ceiling.
    assert target(4, 32, stable=False, serving=1) == (None, 1)


def test_reconcile_waits_for_resolved_gpu_costs():
    assert fleet_reconcile_targets((2, 2), (2, 2), (None, 4), (1, 1), 0, True) is None


def _planner(budgets, *, active=(0, 0), pending=(0, 0), scaling=False, advisory=False):
    state = DeploymentState()
    for component, name, count, starting in (
        (state.prefill, "prefill-worker", active[0], pending[0]),
        (state.decode, "decode-worker", active[1], pending[1]),
    ):
        component.info = WorkerInfo(k8s_name=name)
        component.num_gpus = 4
        component.gpus_per_replica = 4
        component.replicas.active = count
        component.replicas.expected = None if scaling else count
        component.replicas.scaling = scaling
        component.replicas.pending_startup = starting

    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.metrics_state.return_value = Metrics()
    environment.fleet_gpu_budget.side_effect = budgets
    environment.apply_scaling = AsyncMock()

    config = PlannerConfig(
        mode="disagg",
        advisory=advisory,
        namespace="test-namespace",
        min_gpu_budget=-1,
        max_gpu_budget=64,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics",
        return_value=MagicMock(),
    ):
        planner = DisaggPlanner(None, config, environment)
    return planner, environment


def _applied(environment):
    return [
        {t.sub_component_type: t.desired_replicas for t in c.args[0]}
        for c in environment.apply_scaling.await_args_list
    ]


@pytest.mark.asyncio
async def test_allocation_replaces_the_band_until_it_changes_or_is_removed():
    planner, _ = _planner(
        [GpuBudget(8, 32), GpuBudget(8, 32), GpuBudget(8, 32), GpuBudget(8, 48), None]
    )
    generation = planner._config_generation

    await planner._apply_fleet_gpu_budget()
    assert (planner.config.min_gpu_budget, planner.config.max_gpu_budget) == (8, 32)
    assert planner._config_generation == generation + 1

    # A control API update stands while the allocation is unchanged.
    await planner.patch_min_endpoints({"max_gpu_budget": 24})
    await planner._apply_fleet_gpu_budget()
    await planner._apply_fleet_gpu_budget()
    assert planner.config.max_gpu_budget == 24

    await planner._apply_fleet_gpu_budget()
    assert (planner.config.min_gpu_budget, planner.config.max_gpu_budget) == (8, 48)

    await planner._apply_fleet_gpu_budget()
    assert (planner.config.min_gpu_budget, planner.config.max_gpu_budget) == (-1, 64)


@pytest.mark.asyncio
async def test_zero_allocation_drains_and_a_later_allocation_wakes_without_traffic():
    planner, environment = _planner([GpuBudget(0, 0)], active=(2, 2))
    await planner._apply_fleet_gpu_budget()
    assert _applied(environment) == [
        {SubComponentType.PREFILL: 0, SubComponentType.DECODE: 0}
    ]

    planner, environment = _planner([GpuBudget(8, 16)], active=(0, 0))
    await planner._apply_fleet_gpu_budget()
    assert _applied(environment) == [
        {SubComponentType.PREFILL: 1, SubComponentType.DECODE: 1}
    ]


@pytest.mark.asyncio
async def test_no_wake_while_counts_move_and_no_writes_in_advisory_mode():
    planner, environment = _planner([GpuBudget(8, 16)], scaling=True)
    await planner._apply_fleet_gpu_budget()
    assert _applied(environment) == []

    # Mid-rollout (nothing verified pending): the connector admits no change.
    planner, environment = _planner([GpuBudget(0, 0)], active=(2, 2), scaling=True)
    await planner._apply_fleet_gpu_budget()
    assert _applied(environment) == []

    planner, environment = _planner([GpuBudget(0, 0)], active=(2, 2), advisory=True)
    await planner._apply_fleet_gpu_budget()
    assert planner.config.max_gpu_budget == 0
    assert _applied(environment) == []


@pytest.mark.asyncio
async def test_shrink_during_startup_cancels_pending_replicas_first():
    planner, environment = _planner(
        [GpuBudget(0, 8)], active=(1, 1), pending=(3, 0), scaling=True
    )
    await planner._apply_fleet_gpu_budget()
    assert _applied(environment) == [{SubComponentType.PREFILL: 1}]


@pytest.mark.asyncio
async def test_plugin_pipelines_keep_the_band_but_leave_scaling_to_the_engine():
    planner, environment = _planner([GpuBudget(0, 0)], active=(2, 2))
    planner.config.scheduling.gateway.enabled = True
    await planner._apply_fleet_gpu_budget()
    assert planner.config.max_gpu_budget == 0
    assert _applied(environment) == []


@pytest.mark.asyncio
async def test_malformed_allocation_keeps_the_current_band():
    planner, environment = _planner([ValueError("bad allocation")])
    await planner._apply_fleet_gpu_budget()
    assert (planner.config.min_gpu_budget, planner.config.max_gpu_budget) == (-1, 64)
    assert _applied(environment) == []
