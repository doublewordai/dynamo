//! Report aggregation: per class, per window and per phase, as JSON and a markdown table.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::config::{Assertions, Scenario};

/// One routing outcome, kept for the whole run so reports and assertions can aggregate it
/// however they like.
#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub arrival_time: f64,
    /// Tier name for proxy requests, `"hosted"` for hosted, `"none"` if it never routed.
    pub class: String,
    pub is_proxy: bool,
    pub prompt_tokens: u64,
    pub cache_hit_tokens: u64,
    pub is_followup: bool,
    pub sticky: bool,
    /// Same class (hosted vs proxy) as the previous turn. Only meaningful for follow-ups.
    pub class_sticky: bool,
    pub previous_under_threshold: bool,
    pub hosted_occupancy_at_selection: f64,
    pub failed: bool,
    pub rate_limited_attempts: usize,
    /// Hosted workers this request's decision excluded because their engine queue was at
    /// or above the admission margin.
    pub steering_excluded: usize,
    /// True when the request was refused (529/overload) because every hosted worker was at
    /// its margin and no proxy could take it.
    pub admission_529: bool,
}

/// Raw output of a run, before aggregation.
#[derive(Debug, Clone, Default)]
pub struct RunData {
    pub records: Vec<RequestRecord>,
    /// `(time, hosted decode occupancy)` samples at every decision and completion.
    pub occupancy_samples: Vec<(f64, f64)>,
    /// First time each tier was used, keyed by tier name.
    pub spill_first: BTreeMap<String, f64>,
    pub decision_trace: Vec<String>,
    pub total_rate_limited: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub requests: usize,
    pub hosted: usize,
    pub proxy: usize,
    pub by_tier: BTreeMap<String, usize>,
    pub hosted_share: f64,
    pub proxy_share: f64,
    pub cache_hit_rate: f64,
    pub mean_hosted_occupancy: f64,
    pub max_hosted_occupancy: f64,
    pub worker_stickiness: f64,
    pub class_stickiness: f64,
    pub failures: usize,
    /// Sum over requests of hosted workers excluded by the admission margin.
    pub steering_exclusions: usize,
    /// Requests refused because every hosted worker was saturated and no proxy was available.
    pub admission_529: usize,
    /// Cache-hit rate over hosted requests only, i.e. how many cached conversations stay on
    /// hosted rather than spilling. 0 when no request was hosted.
    pub hosted_cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WindowReport {
    pub start: f64,
    pub end: f64,
    #[serde(flatten)]
    pub summary: Summary,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub scenario: String,
    pub duration_seconds: f64,
    pub overall: Summary,
    pub windows: Vec<WindowReport>,
    pub phases: BTreeMap<String, Summary>,
    pub spill_order: Vec<String>,
    pub rate_limited: usize,
    pub decision_trace: Vec<String>,
    pub default_mismatches: Option<usize>,
    /// Overall worker stickiness of upstream's `DefaultWorkerSelector` on the same scenario and
    /// seed, when a scenario asks for the relative check.
    pub default_worker_stickiness: Option<f64>,
}

const WINDOW_SECONDS: f64 = 10.0;

impl Report {
    pub fn build(scenario: &Scenario, data: &RunData) -> Self {
        let overall = summarize(&data.records, &data.occupancy_samples);
        let mut windows = Vec::new();
        let count = (scenario.duration_seconds / WINDOW_SECONDS).ceil().max(1.0) as usize;
        for index in 0..count {
            let start = index as f64 * WINDOW_SECONDS;
            let end = (start + WINDOW_SECONDS).min(scenario.duration_seconds);
            let records = filter_records(&data.records, start, end);
            let samples = filter_samples(&data.occupancy_samples, start, end);
            windows.push(WindowReport {
                start,
                end,
                summary: summarize(&records, &samples),
            });
        }
        let phases = scenario
            .phases
            .iter()
            .map(|phase| {
                (
                    phase.name.clone(),
                    summarize(
                        &filter_records(&data.records, phase.start, phase.end),
                        &filter_samples(&data.occupancy_samples, phase.start, phase.end),
                    ),
                )
            })
            .collect();
        let spill_order = data
            .spill_first
            .iter()
            .map(|(name, time)| (name.clone(), *time))
            .collect::<Vec<_>>();
        let mut spill_order = spill_order;
        spill_order.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Self {
            scenario: scenario.name.clone(),
            duration_seconds: scenario.duration_seconds,
            overall,
            windows,
            phases,
            spill_order: spill_order.into_iter().map(|(name, _)| name).collect(),
            rate_limited: data.total_rate_limited,
            decision_trace: data.decision_trace.clone(),
            default_mismatches: None,
            default_worker_stickiness: None,
        }
    }

    pub fn tier_share(&self, tier: &str) -> f64 {
        if self.overall.requests == 0 {
            0.0
        } else {
            self.overall.by_tier.get(tier).copied().unwrap_or(0) as f64
                / self.overall.requests as f64
        }
    }

    pub fn markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# routing-sim: {}\n\n", self.scenario));
        out.push_str(&format!(
            "Duration {:.0}s, {} requests, {} failures, {} rate limited, spill order: {}\n\n",
            self.duration_seconds,
            self.overall.requests,
            self.overall.failures,
            self.rate_limited,
            if self.spill_order.is_empty() {
                "none".to_string()
            } else {
                self.spill_order.join(" -> ")
            }
        ));
        let tiers: Vec<String> = self.overall.by_tier.keys().cloned().collect();
        out.push_str("| window | requests | hosted |");
        for tier in &tiers {
            out.push_str(&format!(" {tier} |"));
        }
        out.push_str(
            " hosted share | proxy share | cache hit | hosted cache hit | mean occ | max occ | worker sticky | class sticky | steer excl | 529 | failures |\n",
        );
        out.push_str("|---|---|---|");
        for _ in &tiers {
            out.push_str("---|");
        }
        out.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
        for window in &self.windows {
            push_row(
                &mut out,
                &format!("{:.0}-{:.0}s", window.start, window.end),
                &window.summary,
                &tiers,
            );
        }
        push_row(&mut out, "overall", &self.overall, &tiers);
        out.push('\n');
        for (name, summary) in &self.phases {
            out.push_str(&format!(
                "Phase `{name}`: hosted {:.1}%, cache hit {:.1}%, worker sticky {:.1}%, class sticky {:.1}%.\n",
                summary.hosted_share * 100.0,
                summary.cache_hit_rate * 100.0,
                summary.worker_stickiness * 100.0,
                summary.class_stickiness * 100.0
            ));
        }
        out
    }

    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
    }
}

fn push_row(out: &mut String, label: &str, summary: &Summary, tiers: &[String]) {
    out.push_str(&format!(
        "| {label} | {} | {} |",
        summary.requests, summary.hosted
    ));
    for tier in tiers {
        out.push_str(&format!(
            " {} |",
            summary.by_tier.get(tier).copied().unwrap_or(0)
        ));
    }
    out.push_str(&format!(
        " {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {} | {} | {} |\n",
        summary.hosted_share * 100.0,
        summary.proxy_share * 100.0,
        summary.cache_hit_rate * 100.0,
        summary.hosted_cache_hit_rate * 100.0,
        summary.mean_hosted_occupancy * 100.0,
        summary.max_hosted_occupancy * 100.0,
        summary.worker_stickiness * 100.0,
        summary.class_stickiness * 100.0,
        summary.steering_exclusions,
        summary.admission_529,
        summary.failures
    ));
}

fn summarize(records: &[RequestRecord], samples: &[(f64, f64)]) -> Summary {
    let mut hosted = 0usize;
    let mut proxy = 0usize;
    let mut failures = 0usize;
    let mut prompt_tokens = 0u64;
    let mut cache_hit_tokens = 0u64;
    let mut by_tier: BTreeMap<String, usize> = BTreeMap::new();
    let mut followups = 0usize;
    let mut worker_sticky = 0usize;
    let mut class_sticky = 0usize;
    let mut steering_exclusions = 0usize;
    let mut admission_529 = 0usize;
    let mut hosted_prompt_tokens = 0u64;
    let mut hosted_cache_hit_tokens = 0u64;
    for record in records {
        if record.failed {
            failures += 1;
        } else if record.is_proxy {
            proxy += 1;
            *by_tier.entry(record.class.clone()).or_default() += 1;
        } else {
            hosted += 1;
            hosted_prompt_tokens += record.prompt_tokens;
            hosted_cache_hit_tokens += record.cache_hit_tokens;
        }
        steering_exclusions += record.steering_excluded;
        if record.admission_529 {
            admission_529 += 1;
        }
        prompt_tokens += record.prompt_tokens;
        cache_hit_tokens += record.cache_hit_tokens;
        if record.is_followup && record.previous_under_threshold {
            followups += 1;
            if record.sticky {
                worker_sticky += 1;
            }
            if record.class_sticky {
                class_sticky += 1;
            }
        }
    }
    let requests = records.len();
    let occupancy = if samples.is_empty() {
        (0.0, 0.0)
    } else {
        let mean = samples.iter().map(|(_, value)| value).sum::<f64>() / samples.len() as f64;
        let max = samples.iter().map(|(_, value)| *value).fold(0.0, f64::max);
        (mean, max)
    };
    Summary {
        requests,
        hosted,
        proxy,
        by_tier,
        hosted_share: ratio(hosted as f64, requests as f64),
        proxy_share: ratio(proxy as f64, requests as f64),
        cache_hit_rate: ratio(cache_hit_tokens as f64, prompt_tokens as f64),
        mean_hosted_occupancy: occupancy.0,
        max_hosted_occupancy: occupancy.1,
        worker_stickiness: if followups == 0 {
            1.0
        } else {
            worker_sticky as f64 / followups as f64
        },
        class_stickiness: if followups == 0 {
            1.0
        } else {
            class_sticky as f64 / followups as f64
        },
        failures,
        steering_exclusions,
        admission_529,
        hosted_cache_hit_rate: ratio(hosted_cache_hit_tokens as f64, hosted_prompt_tokens as f64),
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn filter_records(records: &[RequestRecord], start: f64, end: f64) -> Vec<RequestRecord> {
    records
        .iter()
        .filter(|record| record.arrival_time >= start && record.arrival_time < end)
        .cloned()
        .collect()
}

fn filter_samples(samples: &[(f64, f64)], start: f64, end: f64) -> Vec<(f64, f64)> {
    samples
        .iter()
        .filter(|(time, _)| *time >= start && *time < end)
        .copied()
        .collect()
}

/// Check a report against a scenario's assertions. Returns human-readable failures.
pub fn check_assertions(scenario: &Scenario, report: &Report) -> Vec<String> {
    let mut failures = Vec::new();
    let assertions: &Assertions = &scenario.assertions;
    let overall = &report.overall;

    if let Some(min) = assertions.hosted_share_min
        && overall.hosted_share < min
    {
        failures.push(format!(
            "hosted_share {:.3} < {min:.3}",
            overall.hosted_share
        ));
    }
    if let Some(max) = assertions.proxy_share_max
        && overall.proxy_share > max
    {
        failures.push(format!("proxy_share {:.3} > {max:.3}", overall.proxy_share));
    }
    if let Some(min) = assertions.peak_hosted_occupancy_min
        && overall.max_hosted_occupancy < min
    {
        failures.push(format!(
            "peak_hosted_occupancy {:.3} < {min:.3}",
            overall.max_hosted_occupancy
        ));
    }
    if let Some(max) = assertions.proxy_share_max_when_hosted_under_threshold {
        let threshold = scenario.policy.occupancy_threshold;
        for window in &report.windows {
            if window.summary.requests == 0 {
                continue;
            }
            if window.summary.max_hosted_occupancy < threshold && window.summary.proxy_share > max {
                failures.push(format!(
                    "window {:.0}-{:.0}s: proxy_share {:.3} > {max:.3} while occupancy {:.3} < {threshold:.3}",
                    window.start,
                    window.end,
                    window.summary.proxy_share,
                    window.summary.max_hosted_occupancy
                ));
            }
        }
    }
    if let Some(order) = &assertions.tier_order {
        for pair in order.windows(2) {
            let (first, second) = (report.tier_share(&pair[0]), report.tier_share(&pair[1]));
            if first <= second {
                failures.push(format!(
                    "tier order: {} share {first:.3} is not greater than {} share {second:.3}",
                    pair[0], pair[1]
                ));
            }
        }
    }
    for (tier, min) in &assertions.tier_share_min {
        let share = report.tier_share(tier);
        if share < *min {
            failures.push(format!("tier {tier} share {share:.3} < {min:.3}"));
        }
    }
    if let Some(min) = assertions.class_stickiness_min
        && overall.class_stickiness < min
    {
        failures.push(format!(
            "class_stickiness {:.3} < {min:.3}",
            overall.class_stickiness
        ));
    }
    if let Some(min_delta) = assertions.worker_stickiness_vs_default_min_delta {
        match report.default_worker_stickiness {
            Some(default_stickiness) => {
                let delta = overall.worker_stickiness - default_stickiness;
                if delta < min_delta {
                    failures.push(format!(
                        "worker_stickiness delta vs default {delta:+.3} < {min_delta:+.3} \
                         (policy {:.3}, default {default_stickiness:.3})",
                        overall.worker_stickiness
                    ));
                }
            }
            None => failures.push(
                "worker_stickiness_vs_default_min_delta set but no default run was recorded"
                    .to_string(),
            ),
        }
    }
    if let Some(max) = assertions.failures_max
        && overall.failures > max
    {
        failures.push(format!("failures {} > {max}", overall.failures));
    }
    if let Some(max) = assertions.steering_exclusions_max
        && overall.steering_exclusions > max
    {
        failures.push(format!(
            "steering_exclusions {} > {max}",
            overall.steering_exclusions
        ));
    }
    if let Some(max) = assertions.admission_529_max
        && overall.admission_529 > max
    {
        failures.push(format!("admission_529 {} > {max}", overall.admission_529));
    }
    if assertions.all_decisions_match_default
        && let Some(mismatches) = report.default_mismatches
        && mismatches > 0
    {
        failures.push(format!(
            "{mismatches} decisions differ from DefaultWorkerSelector"
        ));
    }
    for (name, phase_assertion) in &assertions.phases {
        let Some(summary) = report.phases.get(name) else {
            failures.push(format!("phase `{name}` not found"));
            continue;
        };
        if let Some(min) = phase_assertion.hosted_share_min
            && summary.hosted_share < min
        {
            failures.push(format!(
                "phase `{name}` hosted_share {:.3} < {min:.3}",
                summary.hosted_share
            ));
        }
        if let Some(min) = phase_assertion.proxy_share_min
            && summary.proxy_share < min
        {
            failures.push(format!(
                "phase `{name}` proxy_share {:.3} < {min:.3}",
                summary.proxy_share
            ));
        }
        if let Some(max) = phase_assertion.hosted_share_max
            && summary.hosted_share > max
        {
            failures.push(format!(
                "phase `{name}` hosted_share {:.3} > {max:.3}",
                summary.hosted_share
            ));
        }
    }
    failures
}
