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
