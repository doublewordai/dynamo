// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parameter sweeps: run one scenario over a grid of policy settings and tabulate the metrics
//! that matter for tuning. Independent points run in parallel; each point keeps the scenario's
//! seed, so the grid is deterministic regardless of thread count.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_kv_router::KvRouterConfig;
use serde::Serialize;

use crate::config::{AdmissionConfig, Scenario};
use crate::report::{Report, Summary};
use crate::selector::PolicySelector;

/// One swept parameter: a model or tier field name plus the values to try.
#[derive(Debug, Clone, Serialize)]
pub struct ParamSpec {
    pub name: String,
    pub values: Vec<f64>,
}

impl ParamSpec {
    /// Parse `name=v1,v2,...`, e.g. `failover_penalty_blocks=100,200` or `X.penalty_blocks=0,300`.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let (name, values) = text
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--param expects name=v1,v2,... (got {text:?})"))?;
        if name.is_empty() {
            anyhow::bail!("--param name must not be empty");
        }
        let values = values
            .split(',')
            .map(|value| {
                let parsed: f64 = value
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--param {name}: invalid number {value:?}"))?;
                if !parsed.is_finite() {
                    anyhow::bail!("--param {name}: value must be finite");
                }
                Ok(parsed)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if values.is_empty() {
            anyhow::bail!("--param {name} needs at least one value");
        }
        Ok(Self {
            name: name.to_string(),
            values,
        })
    }
}

/// Metrics from one grid point.
#[derive(Debug, Clone, Serialize)]
pub struct SweepRow {
    pub settings: BTreeMap<String, f64>,
    pub requests: usize,
    /// Proxy share over the whole run.
    pub proxy_share: f64,
    /// Proxy share within the phase named `peak`, or the whole run when there is no such phase.
    pub peak_proxy_share: f64,
    pub peak_mean_primary_occupancy: f64,
    pub peak_max_primary_occupancy: f64,
    pub peak_class_stickiness: f64,
    pub worker_stickiness: f64,
    pub cache_hit_rate: f64,
    /// Cache-hit rate over primary requests only.
    pub primary_cache_hit_rate: f64,
    pub failures: usize,
    /// Sum over requests of primary workers excluded by the admission margin.
    pub steering_exclusions: usize,
    /// Requests refused because every primary worker was saturated and no proxy was available.
    pub admission_529: usize,
    /// Requests per proxy tier; primary requests are not counted.
    pub tier_counts: BTreeMap<String, usize>,
}

/// The full sweep: the grid and one row per point.
#[derive(Debug, Clone, Serialize)]
pub struct SweepResult {
    pub scenario: String,
    pub params: Vec<ParamSpec>,
    /// Phase used for the peak columns, when the scenario defines one named `peak`.
    pub peak_phase: Option<String>,
    pub rows: Vec<SweepRow>,
}

impl SweepResult {
    /// A markdown table plus a short header. One column per proxy tier found in the results.
    pub fn markdown(&self) -> String {
        let tiers = self.tier_names();
        let mut out = String::new();
        out.push_str(&format!("# routing-sim sweep: {}\n\n", self.scenario));
        out.push_str(&format!(
            "Peak columns use the `{}` phase.\n\n",
            self.peak_phase.as_deref().unwrap_or("overall")
        ));
        out.push_str("Grid: ");
        for (index, spec) in self.params.iter().enumerate() {
            if index > 0 {
                out.push_str("; ");
            }
            out.push_str(&format!(
                "{}={}",
                spec.name,
                spec.values
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        out.push_str("\n\n");
        out.push_str("| settings | requests | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | worker sticky % | cache hit % | primary cache hit % | steer excl | 529 | failures |");
        for tier in &tiers {
            out.push_str(&format!(" {tier} % |"));
        }
        out.push_str("\n|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        for _ in &tiers {
            out.push_str("---|");
        }
        out.push('\n');
        for row in &self.rows {
            out.push_str(&format!(
                "| {} | {} |",
                settings_label(&row.settings),
                row.requests
            ));
            out.push_str(&format!(
                " {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | {} |",
                row.proxy_share * 100.0,
                row.peak_proxy_share * 100.0,
                row.peak_mean_primary_occupancy * 100.0,
                row.peak_max_primary_occupancy * 100.0,
                row.peak_class_stickiness * 100.0,
                row.worker_stickiness * 100.0,
                row.cache_hit_rate * 100.0,
                row.primary_cache_hit_rate * 100.0,
                row.steering_exclusions,
                row.admission_529,
                row.failures
            ));
            for tier in &tiers {
                let share = if row.requests == 0 {
                    0.0
                } else {
                    row.tier_counts.get(tier).copied().unwrap_or(0) as f64 / row.requests as f64
                };
                out.push_str(&format!(" {:.1} |", share * 100.0));
            }
            out.push('\n');
        }
        out
    }

    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
    }

    fn tier_names(&self) -> Vec<String> {
        let mut tiers: BTreeMap<String, ()> = BTreeMap::new();
        for row in &self.rows {
            for tier in row.tier_counts.keys() {
                tiers.insert(tier.clone(), ());
            }
        }
        tiers.into_keys().collect()
    }
}

fn settings_label(settings: &BTreeMap<String, f64>) -> String {
    settings
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Expand the specs into their cartesian product, preserving each spec's value order.
pub fn grid(specs: &[ParamSpec]) -> Vec<BTreeMap<String, f64>> {
    let mut combos = vec![BTreeMap::new()];
    for spec in specs {
        let mut next = Vec::with_capacity(combos.len() * spec.values.len());
        for combo in &combos {
            for value in &spec.values {
                let mut extended = combo.clone();
                extended.insert(spec.name.clone(), *value);
                next.push(extended);
            }
        }
        combos = next;
    }
    combos
}

/// Apply one named setting to a scenario. Model fields are bare names; tier fields are
/// `<tier>.<field>`, e.g. `X.penalty_blocks`.
pub fn apply_setting(scenario: &mut Scenario, name: &str, value: f64) -> anyhow::Result<()> {
    let policy = &mut scenario.policy;
    match name {
        "occupancy_threshold" => policy.occupancy_threshold = value,
        "primary_capacity_blocks" => policy.primary_capacity_blocks = Some(value),
        "primary_max_requests" => policy.primary_max_requests = Some(value),
        "failover_penalty_blocks" => policy.failover_penalty_blocks = value,
        "pending_weight_blocks" => policy.pending_weight_blocks = value,
        // Not a policy field: the margin is a worker-process environment value, so the sweep
        // creates the admission model when the scenario does not declare one. Values are
        // truncated to whole requests.
        "admission_queue_margin" => {
            let margin = value.max(0.0) as u64;
            match &mut scenario.admission {
                Some(admission) => admission.primary_queue_margin = margin,
                None => {
                    scenario.admission = Some(AdmissionConfig {
                        primary_queue_margin: margin,
                        primary_queue_margin_overrides: BTreeMap::new(),
                    })
                }
            }
        }
        other => {
            let (tier_name, field) = other.split_once('.').ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown sweep parameter {other:?}; expected a model field or `<tier>.<field>`"
                )
            })?;
            let tier = policy
                .tiers
                .iter_mut()
                .find(|tier| tier.name == tier_name)
                .ok_or_else(|| {
                    anyhow::anyhow!("sweep parameter {other:?} names an unknown tier {tier_name:?}")
                })?;
            match field {
                "penalty_blocks" => tier.penalty_blocks = value,
                "weight_blocks" => tier.weight_blocks = value,
                _ => anyhow::bail!("unknown tier field {field:?} in {other:?}"),
            }
        }
    }
    Ok(())
}

/// Run every point of the grid and collect the rows in grid order. `jobs` bounds the number of
/// worker threads; each point still uses the scenario's own seed.
pub fn run(scenario: &Scenario, specs: &[ParamSpec], jobs: usize) -> anyhow::Result<SweepResult> {
    if scenario.policy.no_parameters {
        anyhow::bail!(
            "scenario {:?} sets no_parameters; a sweep needs the real policy",
            scenario.name
        );
    }
    let combos = grid(specs);
    // Validate every point up front so worker threads cannot fail after spawning.
    let scenarios: Vec<Scenario> =
        combos
            .iter()
            .map(|settings| {
                let mut point = scenario.clone();
                for (name, value) in settings {
                    apply_setting(&mut point, name, *value)?;
                }
                point.policy.parameters().validate().map_err(|error| {
                    anyhow::anyhow!("invalid sweep point {settings:?}: {error}")
                })?;
                Ok(point)
            })
            .collect::<anyhow::Result<_>>()?;

    let peak_phase = scenario
        .phases
        .iter()
        .find(|phase| phase.name == "peak")
        .map(|phase| phase.name.clone());
    let jobs = jobs.max(1).min(scenarios.len().max(1));
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<SweepRow>>> =
        (0..scenarios.len()).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= scenarios.len() {
                        break;
                    }
                    let mut row = run_point(&scenarios[index], peak_phase.as_deref());
                    row.settings = combos[index].clone();
                    *slots[index].lock().expect("sweep slot lock") = Some(row);
                }
            });
        }
    });

    let rows = slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("sweep slot lock")
                .expect("row filled")
        })
        .collect();
    Ok(SweepResult {
        scenario: scenario.name.clone(),
        params: specs.to_vec(),
        peak_phase,
        rows,
    })
}

fn run_point(scenario: &Scenario, peak_phase: Option<&str>) -> SweepRow {
    let mut selector = PolicySelector::new(
        &KvRouterConfig::default(),
        &scenario.policy.model,
        &scenario.policy.parameters(),
        scenario.seed,
    );
    let report = crate::run_scenario(scenario, &mut selector);
    let peak = peak_summary(&report, peak_phase);
    SweepRow {
        // Filled from the grid combination by `run`; empty only when called directly.
        settings: BTreeMap::new(),
        requests: report.overall.requests,
        proxy_share: report.overall.proxy_share,
        peak_proxy_share: peak.proxy_share,
        peak_mean_primary_occupancy: peak.mean_primary_occupancy,
        peak_max_primary_occupancy: peak.max_primary_occupancy,
        peak_class_stickiness: peak.class_stickiness,
        worker_stickiness: report.overall.worker_stickiness,
        cache_hit_rate: report.overall.cache_hit_rate,
        primary_cache_hit_rate: report.overall.primary_cache_hit_rate,
        failures: report.overall.failures,
        steering_exclusions: report.overall.steering_exclusions,
        admission_529: report.overall.admission_529,
        tier_counts: report.overall.by_tier.clone(),
    }
}

fn peak_summary<'a>(report: &'a Report, peak_phase: Option<&str>) -> &'a Summary {
    match peak_phase.and_then(|name| report.phases.get(name)) {
        Some(summary) => summary,
        None => &report.overall,
    }
}
