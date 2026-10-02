// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scenario validation: values that would hang or mislabel a run are rejected at parse time.

use dw_routing_sim::Scenario;
use dw_routing_sim::config::{RatePoint, TierDef};
use dw_routing_sim::scenarios;
use dw_routing_sim::sweep::{ParamSpec, apply_setting, run};

fn base() -> Scenario {
    scenarios::load_builtin("low_load").expect("builtin scenario parses")
}

#[test]
fn non_finite_duration_is_rejected() {
    let mut scenario = base();
    scenario.duration_seconds = f64::INFINITY;
    assert!(scenario.validate().is_err());
    scenario.duration_seconds = -1.0;
    assert!(scenario.validate().is_err());
}

#[test]
fn non_finite_arrival_point_is_rejected() {
    let mut scenario = base();
    scenario.arrival_rate.push(RatePoint {
        time: 0.0,
        rate: f64::NAN,
    });
    assert!(scenario.validate().is_err());
}

#[test]
fn unordered_arrival_times_are_rejected() {
    let mut scenario = base();
    scenario.arrival_rate = vec![
        RatePoint {
            time: 0.0,
            rate: 1.0,
        },
        RatePoint {
            time: 5.0,
            rate: 1.0,
        },
        RatePoint {
            time: 1.0,
            rate: 1.0,
        },
    ];
    assert!(scenario.validate().is_err());
}

#[test]
fn zero_block_size_is_rejected() {
    let mut scenario = base();
    scenario.block_size = 0;
    assert!(scenario.validate().is_err());
}

#[test]
fn duplicate_primary_ids_are_rejected() {
    let mut scenario = base();
    let duplicate = scenario.primary[0].clone();
    scenario.primary.push(duplicate);
    assert!(scenario.validate().is_err());
}

#[test]
fn proxy_rank_colliding_with_a_primary_is_rejected() {
    let mut scenario = base();
    scenario.proxies[0].dp_rank_start = scenario.primary[0].id as u32;
    assert!(scenario.validate().is_err());
}

#[test]
fn overlapping_proxy_tiers_are_rejected() {
    let mut scenario = base();
    let mut duplicate = scenario.proxies[0].clone();
    duplicate.tier = "duplicate".to_string();
    scenario.proxies.push(duplicate);
    assert!(scenario.validate().is_err());
}

#[test]
fn invalid_policy_parameters_are_rejected() {
    let mut scenario = base();
    scenario.policy.occupancy_threshold = 0.0;
    let error = scenario
        .validate()
        .expect_err("threshold 0 must be rejected");
    assert!(format!("{error}").contains("occupancy_threshold"));
}

#[test]
fn invalid_tier_parameters_are_rejected() {
    let mut scenario = base();
    scenario.policy.tiers.push(TierDef {
        name: "broken".to_string(),
        // Includes rank 0, which the policy reserves for primary workers.
        dp_ranks: [0, 3],
        penalty_blocks: 0.0,
        weight_blocks: 0.0,
    });
    assert!(scenario.validate().is_err());
}

#[test]
fn duplicate_sweep_params_are_rejected() {
    let scenario = base();
    let specs = vec![
        ParamSpec::parse("failover_penalty_blocks=100,800").unwrap(),
        ParamSpec::parse("failover_penalty_blocks=200,400").unwrap(),
    ];
    let error = run(&scenario, &specs, 1).expect_err("duplicate param must be rejected");
    assert!(format!("{error}").contains("more than once"));
}

#[test]
fn negative_admission_margin_is_rejected() {
    let mut scenario = base();
    let error = apply_setting(&mut scenario, "admission_queue_margin", -1.0)
        .expect_err("negative margin must be rejected");
    assert!(format!("{error}").contains("non-negative"));
}
