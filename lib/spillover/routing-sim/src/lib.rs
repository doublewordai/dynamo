//! Deterministic, virtual-time simulation of the spillover policy. See docs/PLAN.md
//! "Simulation" for the model, scenarios and CI assertions.
//!
//! The simulation is intentionally independent of the production policy's internals: it builds
//! `SchedulingRequest`s with `dw-spillover-testkit` and asks a [`Selector`]. The real selector
//! uses `dw_spillover_policy::build_policy`; [`HeuristicSelector`] is a stand-in kept for the
//! `--heuristic` CLI flag and a smoke test.

pub mod config;
pub mod engine;
pub mod hash;
pub mod report;
pub mod scenarios;
pub mod selector;
pub mod sweep;
pub mod workload;

pub use config::Scenario;
pub use report::{Report, check_assertions};
pub use selector::{DefaultSelector, HeuristicSelector, PolicySelector, SelectionInput, Selector};

use dynamo_kv_router::KvRouterConfig;

/// Run a scenario with the given selector and build its report.
pub fn run_scenario(scenario: &Scenario, selector: &mut dyn Selector) -> Report {
    let data = engine::Engine::new(scenario, selector).run();
    Report::build(scenario, &data)
}

/// Like [`run_scenario`], but additionally runs upstream's `DefaultWorkerSelector` on the same
/// scenario and seed when an assertion asks for a comparison against it. The reference run is
/// shared by the default-equivalence and worker-stickiness checks.
pub fn run_scenario_with_default_reference(
    scenario: &Scenario,
    selector: &mut dyn Selector,
) -> Report {
    let mut report = run_scenario(scenario, selector);
    let needs_default = scenario.assertions.all_decisions_match_default
        || scenario
            .assertions
            .worker_stickiness_vs_default_min_delta
            .is_some();
    if needs_default {
        let mut default = DefaultSelector::new(&KvRouterConfig::default(), scenario.seed);
        let reference = run_scenario(scenario, &mut default);
        if scenario
            .assertions
            .worker_stickiness_vs_default_min_delta
            .is_some()
        {
            report.default_worker_stickiness = Some(reference.overall.worker_stickiness);
        }
        if scenario.assertions.all_decisions_match_default {
            let mismatches = report
                .decision_trace
                .iter()
                .zip(&reference.decision_trace)
                .filter(|(policy, default)| policy != default)
                .count()
                + report
                    .decision_trace
                    .len()
                    .abs_diff(reference.decision_trace.len());
            report.default_mismatches = Some(mismatches);
        }
    }
    report
}
