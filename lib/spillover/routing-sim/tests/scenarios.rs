// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scenario tests. Every shipped scenario runs with the real spillover policy; the heuristic
//! stand-in is kept only for the `--heuristic` CLI flag and one smoke test.

use dw_routing_sim::{
    DefaultSelector, HeuristicSelector, PolicySelector, Report, Scenario, check_assertions,
    run_scenario, run_scenario_with_default_reference, scenarios,
};
use dynamo_kv_router::KvRouterConfig;

fn policy_selector(scenario: &Scenario) -> PolicySelector {
    PolicySelector::new(
        &KvRouterConfig::default(),
        &scenario.policy.model,
        &scenario.policy.parameters(),
        scenario.seed,
    )
}

fn run_policy(scenario: &Scenario) -> Report {
    let mut selector = policy_selector(scenario);
    run_scenario_with_default_reference(scenario, &mut selector)
}

fn check(name: &str) -> Report {
    let scenario = scenarios::load_builtin(name).expect("builtin scenario parses");
    let report = run_policy(&scenario);
    let failures = check_assertions(&scenario, &report);
    assert!(
        failures.is_empty(),
        "scenario `{name}` failed assertions: {failures:#?}\n\n{}",
        report.markdown()
    );
    report
}

#[test]
fn low_load_stays_on_hosted() {
    check("low_load");
}

#[test]
fn overload_ramp_shares_and_recovers() {
    check("overload_ramp");
}

#[test]
fn stickiness_reuses_worker() {
    check("stickiness");
}

#[test]
fn proxy_rate_limit_reroutes() {
    check("proxy_rate_limited");
}

#[test]
fn no_parameters_falls_back() {
    check("no_parameters");
}

#[test]
fn hosted_outage_uses_proxies() {
    check("hosted_outage");
}

#[test]
fn admission_scenarios_pass_and_steer() {
    let low = check("admission_margin_low");
    let high = check("admission_margin_high");
    // The low margin must actually exclude a hosted worker, otherwise the scenario is inert.
    assert!(low.overall.steering_exclusions > 0);
    assert_eq!(high.overall.steering_exclusions, 0);
    assert_eq!(low.overall.admission_529, 0);
    assert_eq!(high.overall.admission_529, 0);
}

/// The point of the margin: a low value pushes cached conversations to a proxy, a high value
/// keeps them on hosted. These two scenarios are identical except for the margin, so the gap
/// is the gate's effect.
#[test]
fn margin_above_failover_keeps_more_on_hosted() {
    let low = run_policy(&scenarios::load_builtin("admission_margin_low").unwrap());
    let high = run_policy(&scenarios::load_builtin("admission_margin_high").unwrap());
    assert!(
        high.overall.hosted_share > low.overall.hosted_share,
        "hosted share: high {} should exceed low {}",
        high.overall.hosted_share,
        low.overall.hosted_share
    );
    assert!(
        high.overall.hosted_cache_hit_rate > low.overall.hosted_cache_hit_rate,
        "hosted cache hit: high {} should exceed low {}",
        high.overall.hosted_cache_hit_rate,
        low.overall.hosted_cache_hit_rate
    );
    assert!(high.overall.steering_exclusions < low.overall.steering_exclusions);
}

/// With every hosted worker at its margin and no proxy left to take the request, the router
/// must record a 529 instead of silently dropping it.
#[test]
fn saturated_hosted_workers_without_proxies_record_529() {
    let scenario = Scenario::parse(
        r#"
name: admission_saturated
seed: 1
duration_seconds: 30
arrival_rate:
  - { time: 0, rate: 2.0 }
  - { time: 30, rate: 2.0 }
hosted:
  - id: 0
    capacity_blocks: 400
    prefill_tokens_per_second: 100000
    decode_tokens_per_second: 40
    max_concurrent_requests: 2
proxies: []
workload:
  system_prompt_tokens: 64
  user_tokens: { min: 48, max: 96 }
  output_tokens: { min: 64, max: 128 }
  think_time_seconds: { min: 0.1, max: 0.2 }
  turns_per_session: { min: 1, max: 1 }
policy:
  model: test-model@interactive
  occupancy_threshold: 0.8
  hosted_capacity_blocks: 400
admission:
  hosted_queue_margin: 0
"#,
    )
    .unwrap();
    let report = run_policy(&scenario);
    assert!(report.overall.requests > 0);
    assert!(report.overall.steering_exclusions > 0);
    assert!(report.overall.admission_529 > 0);
    assert_eq!(report.overall.admission_529, report.overall.failures);
}

/// Per-worker overrides shadow the shared margin, so an operator can leave one hosted process
/// unenforced (or more patient) while others steer.
#[test]
fn admission_margin_overrides_are_per_worker() {
    let scenario = Scenario::parse(
        r#"
name: admission_overrides
seed: 1
duration_seconds: 5
arrival_rate:
  - { time: 0, rate: 0.2 }
hosted:
  - id: 7
    capacity_blocks: 400
    prefill_tokens_per_second: 100000
    decode_tokens_per_second: 40
    max_concurrent_requests: 2
  - id: 9
    capacity_blocks: 400
    prefill_tokens_per_second: 100000
    decode_tokens_per_second: 40
    max_concurrent_requests: 2
proxies: []
workload:
  system_prompt_tokens: 64
  user_tokens: { min: 48, max: 96 }
  output_tokens: { min: 64, max: 128 }
  think_time_seconds: { min: 0.1, max: 0.2 }
  turns_per_session: { min: 1, max: 1 }
policy:
  model: test-model@interactive
  occupancy_threshold: 0.8
  hosted_capacity_blocks: 800
admission:
  hosted_queue_margin: 100
  hosted_queue_margin_overrides:
    9: 0
"#,
    )
    .unwrap();
    assert_eq!(scenario.admission.as_ref().unwrap().margin_for(7), 100);
    assert_eq!(scenario.admission.as_ref().unwrap().margin_for(9), 0);
    // Worker 9 is always excluded at margin 0; worker 7 never is.
    let report = run_policy(&scenario);
    assert!(report.overall.steering_exclusions > 0);
    assert_eq!(report.overall.failures, 0);
}

#[test]
fn runs_are_deterministic() {
    let scenario = scenarios::load_builtin("overload_ramp").unwrap();
    let first = run_policy(&scenario);
    let second = run_policy(&scenario);
    assert_eq!(first.decision_trace, second.decision_trace);
    assert_eq!(first.markdown(), second.markdown());
    assert_eq!(first.json(), second.json());
}

#[test]
fn no_parameters_matches_default_selector() {
    let scenario = scenarios::load_builtin("no_parameters").unwrap();
    let config = KvRouterConfig::default();
    let mut policy = PolicySelector::new(
        &config,
        &scenario.policy.model,
        &scenario.policy.parameters(),
        scenario.seed,
    );
    let policy_report = run_scenario(&scenario, &mut policy);
    let mut default = DefaultSelector::new(&config, scenario.seed);
    let default_report = run_scenario(&scenario, &mut default);
    assert_eq!(policy_report.decision_trace, default_report.decision_trace);
}

#[test]
fn heuristic_selector_smoke_test() {
    let scenario = scenarios::load_builtin("low_load").unwrap();
    let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
    let report = run_scenario(&scenario, &mut selector);
    assert!(report.overall.requests > 0);
}

#[test]
fn scenario_files_on_disk_match_embedded() {
    for name in scenarios::NAMES {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("scenarios")
            .join(format!("{name}.yaml"));
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(
            &on_disk,
            scenarios::text(name).unwrap(),
            "scenario {name} drifted"
        );
    }
}
