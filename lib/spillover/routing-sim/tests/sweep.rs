// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sweep tests: the grid is applied to the requested settings, and known monotonic directions
//! hold on `overload_ramp`.

use std::collections::BTreeSet;

use dw_routing_sim::Scenario;
use dw_routing_sim::scenarios;
use dw_routing_sim::sweep::{ParamSpec, run};

fn overload_scenario() -> Scenario {
    scenarios::load_builtin("overload_ramp").expect("builtin scenario parses")
}

#[test]
fn sweep_runs_the_requested_grid() {
    let scenario = overload_scenario();
    let specs = vec![
        ParamSpec::parse("failover_penalty_blocks=100,800").unwrap(),
        ParamSpec::parse("occupancy_threshold=0.8,0.9").unwrap(),
    ];
    let result = run(&scenario, &specs, 4).unwrap();
    assert_eq!(result.rows.len(), 4);

    let grid: BTreeSet<(u64, u64)> = result
        .rows
        .iter()
        .map(|row| {
            (
                row.settings["failover_penalty_blocks"] as u64,
                (row.settings["occupancy_threshold"] * 10.0).round() as u64,
            )
        })
        .collect();
    assert_eq!(
        grid,
        BTreeSet::from([(100, 8), (100, 9), (800, 8), (800, 9)])
    );
    // Every grid point runs the same workload. Policy parameters change when requests
    // complete, so a saturated ramp can shift one turn across the duration boundary; allow a
    // small spread rather than exact equality.
    let counts: Vec<usize> = result.rows.iter().map(|row| row.requests).collect();
    let min = *counts.iter().min().unwrap();
    let max = *counts.iter().max().unwrap();
    assert!(
        max - min <= 1,
        "grid points ran different workloads: requests ranged {min}..={max}"
    );
}

#[test]
fn sweep_rows_report_proxy_tiers() {
    let scenario = overload_scenario();
    let specs = vec![ParamSpec::parse("failover_penalty_blocks=500").unwrap()];
    let result = run(&scenario, &specs, 1).unwrap();
    let row = &result.rows[0];
    assert_eq!(row.failures, 0);
    assert!(row.proxy_share > 0.0 && row.proxy_share < 1.0);
    // X carries the spill; Y only fills once X is much more expensive.
    let x = row.tier_counts.get("X").copied().unwrap_or(0);
    let y = row.tier_counts.get("Y").copied().unwrap_or(0);
    assert!(x > y, "X ({x}) should carry more than Y ({y})");
}

#[test]
fn sweep_is_deterministic() {
    let scenario = overload_scenario();
    let specs = vec![ParamSpec::parse("failover_penalty_blocks=100,800").unwrap()];
    let first = run(&scenario, &specs, 2).unwrap();
    let second = run(&scenario, &specs, 2).unwrap();
    assert_eq!(first.markdown(), second.markdown());
    assert_eq!(first.json(), second.json());
}

/// A higher proxy-tier penalty makes proxies less attractive, so the proxy share can only
/// fall (or stay put) as `X.penalty_blocks` grows. This is the sweep's monotonicity check.
///
/// The run-level share is strictly monotone. The phase-level "peak" share can wobble by a
/// fraction of a point because including the concurrency signal adds feedback: routing
/// changes the active-request counts, which shift when spill begins within the peak phase.
#[test]
fn larger_tier_penalty_never_increases_peak_proxy_share() {
    let scenario = overload_scenario();
    let specs = vec![ParamSpec::parse("X.penalty_blocks=0,200,400,800,1600").unwrap()];
    let result = run(&scenario, &specs, 4).unwrap();
    assert_eq!(result.rows.len(), 5);
    for pair in result.rows.windows(2) {
        assert!(
            pair[1].proxy_share <= pair[0].proxy_share + 1e-9,
            "proxy share rose from {} to {} when X.penalty_blocks increased",
            pair[0].proxy_share,
            pair[1].proxy_share
        );
        assert!(
            pair[1].peak_proxy_share <= pair[0].peak_proxy_share + 0.02,
            "peak proxy share rose from {} to {} when X.penalty_blocks increased",
            pair[0].peak_proxy_share,
            pair[1].peak_proxy_share
        );
    }
}

/// The admission margin is a real sweep knob even though it is not a policy field. Raising it
/// can only reduce steering away from primary and can only raise the primary cache-hit rate; at a
/// high enough value no primary worker is excluded at all.
#[test]
fn larger_admission_margin_never_increases_steering() {
    let scenario = scenarios::load_builtin("admission_margin_high").unwrap();
    let specs = vec![ParamSpec::parse("admission_queue_margin=0,1,3,10,1000").unwrap()];
    let result = run(&scenario, &specs, 4).unwrap();
    assert_eq!(result.rows.len(), 5);
    assert_eq!(result.rows[0].settings["admission_queue_margin"], 0.0);
    for row in &result.rows {
        assert_eq!(
            row.admission_529, 0,
            "margin {} caused 529s",
            row.settings["admission_queue_margin"]
        );
    }
    // Raising the margin can only make steering less likely, so the metric must never rise
    // between adjacent grid points, not merely between the two endpoints.
    for pair in result.rows.windows(2) {
        assert!(
            pair[1].steering_exclusions <= pair[0].steering_exclusions,
            "steering rose from {} to {} when admission_queue_margin increased from {} to {}",
            pair[0].steering_exclusions,
            pair[1].steering_exclusions,
            pair[0].settings["admission_queue_margin"],
            pair[1].settings["admission_queue_margin"]
        );
    }
    let low = &result.rows[0];
    let high = result.rows.last().unwrap();
    assert!(
        low.steering_exclusions > high.steering_exclusions,
        "margin 0 steered {} times, margin 1000 steered {}",
        low.steering_exclusions,
        high.steering_exclusions
    );
    assert_eq!(high.steering_exclusions, 0);
    assert!(low.proxy_share > high.proxy_share);
}
