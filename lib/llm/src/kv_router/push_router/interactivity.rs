// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-owned pool assignments with a barrier across frontend admissions.

mod coordination;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::*;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_kv_router::protocols::{ActiveLoad, DecodeMetrics, PotentialLoad};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Pool {
    min_decode_tps_per_user: f64,
    minimum: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct PoolConfig {
    endpoint: String,
    pools: BTreeMap<String, Pool>,
    default_pool: String,
    home_priority: i32,
    borrowed_priority: i32,
    threshold: f64,
    sustained_seconds: f64,
    cooldown_seconds: f64,
    drain_seconds: f64,
    sample_seconds: f64,
    telemetry_ttl_seconds: f64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            pools: BTreeMap::new(),
            default_pool: String::new(),
            home_priority: 100,
            borrowed_priority: 0,
            threshold: 0.8,
            sustained_seconds: 30.0,
            cooldown_seconds: 60.0,
            drain_seconds: 300.0,
            sample_seconds: 0.25,
            telemetry_ttl_seconds: 5.0,
        }
    }
}

impl PoolConfig {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.endpoint.split('.').count() == 3
                && self.endpoint.split('.').all(|part| !part.is_empty()),
            "pool endpoint must be namespace.component.generate"
        );
        anyhow::ensure!(
            self.pools.len() == 2 && self.pools.contains_key(&self.default_pool),
            "two pools and a valid default_pool are required"
        );
        anyhow::ensure!(
            self.home_priority > self.borrowed_priority,
            "home_priority must exceed borrowed_priority"
        );
        for (name, pool) in &self.pools {
            anyhow::ensure!(
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
                "invalid pool name"
            );
            anyhow::ensure!(
                pool.min_decode_tps_per_user.is_finite()
                    && pool.min_decode_tps_per_user > 0.0
                    && pool.minimum > 0,
                "pool decode speed targets must be finite and positive, and minima positive"
            );
        }
        anyhow::ensure!(
            self.threshold.is_finite() && self.threshold > 0.0 && self.threshold < 1.0,
            "invalid threshold"
        );
        anyhow::ensure!(
            [
                self.sustained_seconds,
                self.cooldown_seconds,
                self.drain_seconds,
                self.sample_seconds,
                self.telemetry_ttl_seconds
            ]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0),
            "timings must be finite and positive"
        );
        Ok(())
    }
}

/// A single endpoint config remains valid; an array enables independent fleets.
fn parse_configs(bytes: &[u8]) -> anyhow::Result<Vec<PoolConfig>> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let configs: Vec<PoolConfig> = if value.is_array() {
        serde_json::from_value(value)?
    } else {
        vec![serde_json::from_value(value)?]
    };
    anyhow::ensure!(
        !configs.is_empty(),
        "at least one pool endpoint is required"
    );
    let mut endpoints = HashSet::new();
    for config in &configs {
        config.validate()?;
        anyhow::ensure!(
            endpoints.insert(&config.endpoint),
            "duplicate pool endpoint"
        );
    }
    Ok(configs)
}

fn worker_identity(id: u64, config: &ModelRuntimeConfig) -> String {
    config
        .stable_routing_id
        .clone()
        .unwrap_or_else(|| id.to_string())
}

#[derive(Clone)]
struct RankReport {
    decode: DecodeMetrics,
    waiting: u64,
    kv_used: Option<u64>,
    revision: u64,
    received: Instant,
}

#[derive(Clone)]
struct LocalRequest {
    pool: String,
    reclaim: bool,
}

#[derive(Clone)]
struct Probe {
    request: u64,
    rank: u32,
    revision: u64,
    completed: bool,
}

#[derive(Clone)]
struct Member {
    is_present: bool,
    stable_id: String,
    pool: String,
    target: Option<String>,
    changed: Instant,
    rank_start: u32,
    rank_count: u32,
    decode_blocks: HashMap<u32, u64>,
    kv_total: Option<u64>,
    reports: HashMap<u32, RankReport>,
    local: HashMap<u64, LocalRequest>,
    has_uncertain_requests: bool,
    probe: Option<Probe>,
}

#[derive(Clone, Debug, Serialize)]
struct RankView {
    decode_tps_per_user: Option<f64>,
    running: u64,
    waiting: u64,
    age_seconds: f64,
    revision: u64,
    observed_at_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
struct View {
    worker_id: u64,
    stable_id: String,
    pool: String,
    target: Option<String>,
    healthy: bool,
    occupied: u64,
    local_requests: usize,
    has_uncertain_requests: bool,
    effective_target: f64,
    slowest_decode_tps_per_user: Option<f64>,
    idle: bool,
    waiting: u64,
    kv_used_blocks: u64,
    kv_total_blocks: Option<u64>,
    ranks: BTreeMap<u32, RankView>,
    available_by_pool: BTreeMap<String, bool>,
    exclusion_by_pool: BTreeMap<String, Option<&'static str>>,
    probe_pending: bool,
    reclaim_pending: bool,
    reclaim_available: bool,
}

impl View {
    fn eligible(&self, pool: &str) -> bool {
        self.available_by_pool.get(pool).copied().unwrap_or(false)
            || (self.pool == pool && self.reclaim_available)
    }

    fn headroom(&self, threshold: f64) -> bool {
        self.healthy
            && (self.idle
                || self
                    .slowest_decode_tps_per_user
                    .is_some_and(|speed| speed > self.effective_target / threshold))
    }
}

impl Member {
    fn rank_occupied(&self, rank: u32) -> u64 {
        self.decode_blocks.get(&rank).copied().unwrap_or(0)
    }

    fn view(&self, id: u64, config: &PoolConfig, now: Instant) -> View {
        let ranks = self.rank_start..self.rank_start.saturating_add(self.rank_count);
        let healthy = self.is_present
            && ranks.clone().all(|rank| {
                self.reports.get(&rank).is_some_and(|r| {
                    now.saturating_duration_since(r.received).as_secs_f64()
                        <= config.telemetry_ttl_seconds
                })
            });
        let effective_target = self.local.values().fold(
            config.pools[&self.pool].min_decode_tps_per_user,
            |target, r| target.max(config.pools[&r.pool].min_decode_tps_per_user),
        );
        let occupied = ranks.clone().map(|rank| self.rank_occupied(rank)).sum();
        let idle = healthy && self.reports.values().all(|r| r.decode.is_idle());
        let active_rates = self.reports.values().filter(|r| !r.decode.is_idle());
        let measured = active_rates
            .clone()
            .all(|r| r.decode.tokens_per_user_second.is_some());
        let slowest = measured
            .then(|| {
                active_rates
                    .filter_map(|r| r.decode.tokens_per_user_second)
                    .reduce(f64::min)
            })
            .flatten();
        let reclaim_pending = self.local.values().any(|r| r.reclaim);
        let reclaim_available = healthy
            && measured
            && !idle
            && self.target.is_none()
            && !reclaim_pending
            && slowest.is_some_and(|speed| speed <= effective_target)
            && self.local.values().any(|r| r.pool != self.pool);
        let exclusion_by_pool: BTreeMap<_, _> = config
            .pools
            .iter()
            .map(|(name, pool)| {
                let required = effective_target.max(pool.min_decode_tps_per_user);
                let reason = if !healthy {
                    Some("missing_or_stale_telemetry")
                } else if self.target.is_some() {
                    Some("draining")
                } else if name != &self.pool && reclaim_pending {
                    Some("reclaim_pending")
                } else if !measured {
                    Some("no_decode_sample")
                } else if slowest.is_some_and(|speed| speed <= required) {
                    Some("below_speed_target")
                } else if idle && (self.probe.is_some() || self.has_uncertain_requests) {
                    Some("idle_probe_pending")
                } else {
                    None
                };
                (name.clone(), reason)
            })
            .collect();
        View {
            worker_id: id,
            stable_id: self.stable_id.clone(),
            pool: self.pool.clone(),
            target: self.target.clone(),
            healthy,
            occupied,
            local_requests: self.local.len(),
            has_uncertain_requests: self.has_uncertain_requests,
            effective_target,
            slowest_decode_tps_per_user: slowest,
            idle,
            waiting: self.reports.values().map(|r| r.waiting).sum(),
            kv_used_blocks: self.reports.values().filter_map(|r| r.kv_used).sum(),
            kv_total_blocks: self
                .kv_total
                .map(|total| total.saturating_mul(self.rank_count as u64)),
            ranks: self
                .reports
                .iter()
                .map(|(&rank, r)| {
                    (
                        rank,
                        RankView {
                            decode_tps_per_user: r.decode.tokens_per_user_second,
                            running: r.decode.num_running_reqs,
                            waiting: r.waiting,
                            age_seconds: now.saturating_duration_since(r.received).as_secs_f64(),
                            revision: r.revision,
                            observed_at_unix_ms: r.decode.observed_at_unix_ms,
                        },
                    )
                })
                .collect(),
            available_by_pool: exclusion_by_pool
                .iter()
                .map(|(pool, reason)| (pool.clone(), reason.is_none()))
                .collect(),
            exclusion_by_pool,
            probe_pending: self.probe.is_some(),
            reclaim_pending,
            reclaim_available,
        }
    }
}

#[derive(Clone)]
struct State {
    config: PoolConfig,
    members: HashMap<u64, Member>,
    pressure_since: HashMap<String, Instant>,
    next_request: u64,
    shortage: bool,
    membership_changed: bool,
    drain_ready: bool,
    is_ready: bool,
}

impl State {
    fn new(config: PoolConfig) -> Self {
        Self {
            config,
            members: HashMap::new(),
            pressure_since: HashMap::new(),
            next_request: 0,
            shortage: false,
            membership_changed: false,
            drain_ready: false,
            is_ready: false,
        }
    }

    fn reconcile(&mut self, configs: &HashMap<u64, ModelRuntimeConfig>, now: Instant) {
        let before: HashSet<_> = self.members.keys().copied().collect();
        self.members.retain(|id, member| {
            configs.get(id).is_some_and(|config| {
                worker_identity(*id, config) == member.stable_id
                    && config.data_parallel_start_rank == member.rank_start
                    && config.data_parallel_size.max(1) == member.rank_count
            })
        });
        let mut discovered: Vec<_> = configs.iter().collect();
        discovered.sort_by_key(|(id, config)| (worker_identity(**id, config), **id));
        let mut added = false;
        for (id, config) in discovered {
            if !self.members.contains_key(id) {
                // On ties use the request default, then balance subsequent joins.
                let pool = self
                    .config
                    .pools
                    .keys()
                    .min_by_key(|pool| {
                        (
                            self.members.values().filter(|m| &m.pool == *pool).count(),
                            *pool != &self.config.default_pool,
                        )
                    })
                    .cloned();
                let Some(pool) = pool else {
                    continue;
                };
                let stable_id = worker_identity(*id, config);
                tracing::info!(endpoint=%self.config.endpoint, worker_id=id, %stable_id, %pool, "Pool worker registered in smaller class");
                self.members.insert(
                    *id,
                    Member {
                        is_present: true,
                        stable_id,
                        pool,
                        target: None,
                        changed: now,
                        rank_start: config.data_parallel_start_rank,
                        rank_count: config.data_parallel_size.max(1),
                        decode_blocks: HashMap::new(),
                        kv_total: config.total_kv_blocks,
                        reports: HashMap::new(),
                        local: HashMap::new(),
                        has_uncertain_requests: false,
                        probe: None,
                    },
                );
                added = true;
            }
            if let Some(member) = self.members.get_mut(id) {
                member.is_present = true;
                member.kv_total = config.total_kv_blocks;
                if member.target.is_some()
                    && now.duration_since(member.changed).as_secs_f64() >= self.config.drain_seconds
                {
                    tracing::info!(endpoint=%self.config.endpoint, worker_id=id, pool=%member.pool, "Pool drain timed out; keeping frontend classification");
                    member.target = None;
                    member.changed = now;
                }
            }
        }
        if added || before != self.members.keys().copied().collect() {
            self.membership_changed = true;
            self.pressure_since.clear();
            // Re-evaluate any drain against the newly discovered fleet.
            for member in self.members.values_mut() {
                member.target = None;
            }
            tracing::info!(endpoint=%self.config.endpoint, workers=self.members.len(), pool_routing_enabled=self.is_ready, "Pool fleet membership changed");
        }
    }

    fn report(&mut self, load: ActiveLoad, now: Instant) {
        let Some(member) = self.members.get_mut(&load.worker_id) else {
            return;
        };
        if !(member.rank_start..member.rank_start.saturating_add(member.rank_count))
            .contains(&load.dp_rank)
        {
            return;
        }
        // KV diagnostics may change without renewing the decode observation.
        if let Some(used) = load.kv_used_blocks
            && let Some(report) = member.reports.get_mut(&load.dp_rank)
        {
            report.kv_used = Some(used);
        }
        let Some(decode) = load.decode_metrics.filter(DecodeMetrics::is_valid) else {
            return;
        };
        let revision = decode.observation_revision;
        if member
            .reports
            .get(&load.dp_rank)
            .is_some_and(|old| old.revision >= revision)
        {
            return;
        }
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        // Bound both transport delay and clock skew. Heartbeats retain the source timestamp.
        if decode.observed_at_unix_ms > unix_ms.saturating_add(1000) {
            return;
        }
        let age = Duration::from_millis(unix_ms.saturating_sub(decode.observed_at_unix_ms));
        if age.as_secs_f64() > self.config.telemetry_ttl_seconds {
            return;
        }
        if member.probe.as_ref().is_some_and(|probe| {
            probe.rank == load.dp_rank
                && revision > probe.revision
                && ((decode.num_running_reqs > 0 && decode.tokens_per_user_second.is_some())
                    || (probe.completed && decode.is_idle()))
        }) {
            member.probe = None;
        }
        member.reports.insert(
            load.dp_rank,
            RankReport {
                waiting: decode.num_waiting_reqs,
                kv_used: load
                    .kv_used_blocks
                    .or_else(|| member.reports.get(&load.dp_rank).and_then(|r| r.kv_used)),
                revision,
                received: now.checked_sub(age).unwrap_or(now),
                decode,
            },
        );
    }

    fn update_blocks(&mut self, loads: Vec<PotentialLoad>) {
        for member in self.members.values_mut() {
            member.decode_blocks.clear();
        }
        for load in loads {
            if let Some(member) = self.members.get_mut(&load.worker_id)
                && (member.rank_start..member.rank_start.saturating_add(member.rank_count))
                    .contains(&load.dp_rank)
            {
                member
                    .decode_blocks
                    .insert(load.dp_rank, load.potential_decode_blocks as u64);
            }
        }
    }

    fn views(&self, now: Instant) -> Vec<View> {
        let mut views: Vec<_> = self
            .members
            .iter()
            .filter(|(_, member)| member.is_present)
            .map(|(id, member)| member.view(*id, &self.config, now))
            .collect();
        let mut seen = HashMap::new();
        for view in &views {
            *seen.entry(view.stable_id.clone()).or_insert(0) += 1;
        }
        for view in &mut views {
            if seen[&view.stable_id] != 1 {
                view.healthy = false;
                view.reclaim_available = false;
                view.available_by_pool.values_mut().for_each(|v| *v = false);
                view.exclusion_by_pool
                    .values_mut()
                    .for_each(|v| *v = Some("duplicate_worker_identity"));
            }
        }
        views.sort_by(|a, b| a.stable_id.cmp(&b.stable_id));
        views
    }

    fn rebalance(&mut self, now: Instant) {
        let views = self.views(now);
        self.shortage = self
            .config
            .pools
            .keys()
            .all(|pool| views.iter().all(|view| !view.eligible(pool)));
        if views.len() <= 1 || views.iter().any(|v| !v.healthy) {
            self.pressure_since.clear();
            return;
        }
        if let Some(draining) = views.iter().find(|v| v.target.is_some()) {
            self.pressure_since.clear();
            if self.drain_ready
                && draining.idle
                && draining.occupied == 0
                && draining.local_requests == 0
            {
                let Some(member) = self.members.get_mut(&draining.worker_id) else {
                    return;
                };
                if let Some(target) = member.target.take() {
                    tracing::info!(endpoint=%self.config.endpoint, worker_id=draining.worker_id, stable_id=%member.stable_id, source=%member.pool, target=%target, "Pool reclassified in frontend");
                    member.pool = target;
                    member.changed = now;
                }
            }
            return;
        }
        if self.membership_changed {
            let names: Vec<_> = self.config.pools.keys().cloned().collect();
            let first = views.iter().filter(|v| v.pool == names[0]).count();
            let second = views.len() - first;
            if first.abs_diff(second) <= 1 {
                self.membership_changed = false;
            } else {
                let (donor, recipient) = if first > second {
                    (&names[0], &names[1])
                } else {
                    (&names[1], &names[0])
                };
                if let Some(candidate) = views
                    .iter()
                    .filter(|v| {
                        &v.pool == donor
                            && v.idle
                            && v.local_requests == 0
                            && !v.has_uncertain_requests
                            && views.iter().filter(|other| &other.pool == donor).count()
                                > self.config.pools[donor].minimum
                            && views
                                .iter()
                                .filter(|other| {
                                    &other.pool == donor && other.worker_id != v.worker_id
                                })
                                .all(|other| other.headroom(self.config.threshold))
                    })
                    .min_by_key(|v| (v.occupied, &v.stable_id))
                    && let Some(member) = self.members.get_mut(&candidate.worker_id)
                {
                    member.target = Some(recipient.clone());
                    member.changed = now;
                    tracing::info!(endpoint=%self.config.endpoint, worker_id=candidate.worker_id, source=%donor, target=%recipient, "Pool drain started to balance changed fleet");
                }
                return;
            }
        }
        let groups: BTreeMap<_, Vec<_>> = self
            .config
            .pools
            .keys()
            .map(|name| (name, views.iter().filter(|v| &v.pool == name).collect()))
            .collect();
        if groups
            .iter()
            .any(|(name, group)| group.len() < self.config.pools[*name].minimum)
        {
            self.pressure_since.clear();
            return;
        }
        let pressured: BTreeMap<_, bool> = groups
            .iter()
            .map(|(name, group)| {
                (
                    *name,
                    group.iter().all(|view| {
                        !view.idle
                            && view.slowest_decode_tps_per_user.is_some_and(|speed| {
                                speed <= view.effective_target / self.config.threshold
                            })
                    }),
                )
            })
            .collect();
        let names: Vec<_> = self.config.pools.keys().cloned().collect();
        for recipient in &names {
            if pressured[recipient] {
                self.pressure_since.entry(recipient.clone()).or_insert(now);
            } else {
                self.pressure_since.remove(recipient);
            }
        }
        if self
            .members
            .values()
            .any(|m| now.duration_since(m.changed).as_secs_f64() < self.config.cooldown_seconds)
        {
            return;
        }
        for i in 0..2 {
            let recipient = &names[i];
            let donor = &names[1 - i];
            if !self.pressure_since.get(recipient).is_some_and(|since| {
                now.duration_since(*since).as_secs_f64() >= self.config.sustained_seconds
            }) || groups[donor].len() <= self.config.pools[donor].minimum
            {
                continue;
            }
            for candidate in &groups[donor] {
                if !candidate.idle
                    || candidate.local_requests != 0
                    || candidate.has_uncertain_requests
                {
                    continue;
                }
                if !groups[donor]
                    .iter()
                    .filter(|v| v.worker_id != candidate.worker_id)
                    .all(|v| v.headroom(self.config.threshold))
                {
                    continue;
                }
                if let Some(member) = self.members.get_mut(&candidate.worker_id) {
                    member.target = Some(recipient.clone());
                    member.changed = now;
                    self.pressure_since.clear();
                    tracing::info!(worker_id=candidate.worker_id, source=%donor, target=%recipient, "Idle pool donor drain started after speed pressure");
                    return;
                }
            }
        }
    }
}

pub(super) struct PoolManager {
    state: Mutex<State>,
}

/// Tracks selection through backend completion, including dispatches awaiting a response.
pub(super) struct PoolLease {
    priority: i32,
    manager: Arc<PoolManager>,
    worker: u64,
    request: u64,
    dispatch_started: bool,
    completed: bool,
}
impl PoolLease {
    pub(super) fn start_dispatch(&mut self) {
        self.dispatch_started = true;
    }

    pub(super) fn complete(&mut self) {
        self.completed = true;
    }

    pub(super) fn apply_priority(&self, request: &mut PreprocessedRequest) {
        // Pool ownership is authoritative over client-supplied engine priority.
        request.routing_mut().priority = Some(self.priority);
    }
}

impl Drop for PoolLease {
    fn drop(&mut self) {
        if let Some(member) = self.manager.state.lock().members.get_mut(&self.worker) {
            member.local.remove(&self.request);
            if let Some(probe) = &mut member.probe
                && probe.request == self.request
            {
                if !self.dispatch_started {
                    member.probe = None;
                } else if self.completed {
                    probe.completed = true;
                    probe.revision = member
                        .reports
                        .get(&probe.rank)
                        .map_or(probe.revision, |r| r.revision);
                }
            }
            // A transport failure or dropped stream is not proof that the backend stopped.
            member.has_uncertain_requests |= self.dispatch_started && !self.completed;
        }
    }
}

impl PoolManager {
    pub(super) fn from_env(chooser: &Arc<KvRouter>) -> anyhow::Result<Option<Arc<Self>>> {
        let Some(path) = std::env::var_os("DYN_FRONTEND_INTERACTIVITY_CONFIG") else {
            return Ok(None);
        };
        let configs = parse_configs(&std::fs::read(path)?)?;
        let multiple = configs.len() > 1;
        let endpoint = chooser.client().endpoint.clone();
        let id = endpoint.id();
        let endpoint_name = format!("{}.{}.{}", id.namespace, id.component, id.name);
        let Some(config) = configs.into_iter().find(|c| c.endpoint == endpoint_name) else {
            return Ok(None);
        };
        anyhow::ensure!(
            chooser.kv_router_config().router_track_active_blocks
                && chooser.kv_router_config().router_track_output_blocks,
            "interactivity pools require active and output block tracking"
        );
        let period = Duration::from_secs_f64(config.sample_seconds);
        let client = endpoint.drt().etcd_client().cloned().ok_or_else(|| {
            anyhow::anyhow!("interactivity pool coordination requires etcd discovery")
        })?;
        let mut coordinator = coordination::Coordinator::new(client, &config.endpoint);
        let manager = Arc::new(Self {
            state: Mutex::new(State::new(config)),
        });
        let task_manager = Arc::clone(&manager);
        let configs = chooser.runtime_configs();
        let weak_chooser = Arc::downgrade(chooser);
        let cancel = endpoint.drt().primary_token();
        let status_path = std::env::var_os("DYN_FRONTEND_INTERACTIVITY_STATUS_PATH").map(|path| {
            if multiple {
                let mut path = path;
                path.push(format!(".{endpoint_name}"));
                path
            } else {
                path
            }
        });
        tokio::spawn(async move {
            let manager = task_manager;
            async {
                let mut last_status = String::new();
                loop {
                    let subscription = tokio::select! {
                        _ = cancel.cancelled() => return,
                        result = EventSubscriber::for_endpoint(&endpoint, crate::kv_router::KV_METRICS_SUBJECT) => result,
                    };
                    let mut events = match subscription {
                        Ok(sub) => sub.typed::<ActiveLoad>(),
                        Err(error) => {
                            tracing::warn!(%error, "Pool telemetry subscription failed");
                            tokio::select! { _ = cancel.cancelled() => return, _ = tokio::time::sleep(Duration::from_secs(1)) => {} }
                            if weak_chooser.strong_count() == 0 {
                                return;
                            }
                            continue;
                        }
                    };
                    let mut tick = tokio::time::interval(period);
                    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        let event = tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = tick.tick() => None,
                            event = events.next() => match event {
                                Some(Ok((_, event))) => Some(event),
                                Some(Err(error)) => {tracing::warn!(%error, "Invalid pool telemetry"); continue;},
                                None => break,
                            }
                        };
                        let event_is_tick = event.is_none();
                        {
                            let now = Instant::now();
                            let mut state = manager.state.lock();
                            let Some(chooser) = weak_chooser.upgrade() else {
                                return;
                            };
                            state.update_blocks(current_blocks(&chooser));
                            if let Some(event) = event {
                                state.report(event, now);
                            }
                        }
                        // Poll a linearizable snapshot at the policy cadence. Telemetry events
                        // only update local observations; they never drive independent moves.
                        if event_is_tick {
                            let discovered = configs.borrow().clone();
                            if let Err(error) = coordinator.step(&manager, &discovered).await {
                                tracing::warn!(%error, endpoint=%endpoint_name, "Pool coordination failed; retaining assignments");
                            }
                        }
                        // Keep fleet diagnostics at the policy cadence, independent
                        // of how many workers publish telemetry concurrently.
                        if !event_is_tick {
                            continue;
                        }
                        let status = {
                            let state = manager.state.lock();
                            serde_json::json!({"workers": state.views(Instant::now()), "capacity_shortage": state.shortage, "authority": "leader", "frontend": coordinator.frontend(), "revision": coordinator.revision(), "endpoint": endpoint_name, "pool_routing_enabled": state.is_ready})
                        }.to_string();
                        if status != last_status {
                            tracing::debug!(endpoint=%endpoint_name, state=%status, "Frontend pool state");
                            last_status = status.clone();
                        }
                        if let Some(path) = &status_path {
                            let path = std::path::PathBuf::from(path);
                            let tmp = path.with_extension("tmp");
                            if let Err(error) = async {
                                tokio::fs::write(&tmp, status).await?;
                                tokio::fs::rename(&tmp, &path).await
                            }
                            .await
                            {
                                tracing::warn!(%error, "Cannot write frontend pool status");
                            }
                        }
                    }
                }
            }.await;
            match tokio::time::timeout(Duration::from_secs(5), coordinator.leave(&manager)).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    tracing::warn!(%error, "Pool frontend departure could not be committed")
                }
                Err(_) => tracing::warn!("Pool frontend departure timed out"),
            }
        });
        Ok(Some(manager))
    }

    fn admit(self: &Arc<Self>, worker: WorkerWithDpRank, pool: &str) -> Option<PoolLease> {
        let mut state = self.state.lock();
        if !state.is_ready {
            return None;
        }
        let view = state
            .views(Instant::now())
            .into_iter()
            .find(|v| v.worker_id == worker.worker_id)?;
        if !view.eligible(pool) {
            return None;
        }
        let borrowed = pool != view.pool;
        let priority = if borrowed {
            state.config.borrowed_priority
        } else {
            state.config.home_priority
        };
        let rank = view.ranks.get(&worker.dp_rank)?;
        let probe = rank.running == 0 && rank.waiting == 0;
        let reclaim = !view.available_by_pool[pool];
        // A rank without a measurement can only use the bounded idle probe.
        if probe && (view.probe_pending || view.has_uncertain_requests || reclaim) {
            return None;
        }
        state.next_request += 1;
        let request = state.next_request;
        let member = state.members.get_mut(&worker.worker_id)?;
        if probe {
            member.probe = Some(Probe {
                request,
                rank: worker.dp_rank,
                revision: rank.revision,
                completed: false,
            });
        }
        member.local.insert(
            request,
            LocalRequest {
                pool: pool.into(),
                reclaim,
            },
        );
        tracing::debug!(endpoint=%state.config.endpoint, worker_id=worker.worker_id, dp_rank=worker.dp_rank, request_pool=pool, node_pool=%view.pool, borrowed, priority, reclaim, probe, measured_tps=?view.slowest_decode_tps_per_user, effective_target=view.effective_target.max(state.config.pools[pool].min_decode_tps_per_user), selected_rank_tps=?rank.decode_tps_per_user, telemetry_age_seconds=rank.age_seconds, "Pool routing admitted by frontend");
        Some(PoolLease {
            priority,
            manager: Arc::clone(self),
            worker: worker.worker_id,
            request,
            dispatch_started: false,
            completed: false,
        })
    }
}

// An empty projection returns the scheduler's existing active decode blocks,
// including output growth and shared-prefix accounting, without adding a request.
fn current_blocks(chooser: &KvRouter) -> Vec<PotentialLoad> {
    chooser
        .scheduler
        .get_potential_loads(Some(Vec::new()), 0, HashMap::new(), false)
}

fn pool_error(code: u16, message: &str) -> Error {
    DynamoError::builder()
        .error_type(if code == 400 {
            ErrorType::InvalidArgument
        } else {
            ErrorType::Unavailable
        })
        .http_status(code)
        .message(message.to_owned())
        .build()
        .into()
}

impl KvPushRouter {
    pub(super) fn pools_enabled(&self) -> bool {
        self.pools.is_some()
    }

    pub(super) async fn select_pool_worker(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        query_only: bool,
        affinity_worker: Option<WorkerWithDpRank>,
        migration_worker_ids: Option<HashSet<u64>>,
    ) -> Result<WorkerSelection, Error> {
        let manager = self.pools.as_ref().ok_or_else(|| {
            pool_error(
                400,
                "Frontend interactivity pools are not configured for this endpoint",
            )
        })?;
        if phase != RequestPhase::Aggregated
            || self.chooser.queueing_enabled()
            || request.sampling_options.n.unwrap_or(1) != 1
            || request.sampling_options.best_of.unwrap_or(1) != 1
            || request.sampling_options.use_beam_search.unwrap_or(false)
            || request
                .multi_modal_data
                .as_ref()
                .is_some_and(|data| !data.is_empty())
        {
            return Err(pool_error(
                400,
                "Pools require aggregated single-sequence text serving with router queueing disabled",
            ));
        }
        let (pool, views) = {
            let mut state = manager.state.lock();
            state.update_blocks(current_blocks(&self.chooser));
            let pool = request
                .routing
                .as_ref()
                .and_then(|r| r.interactivity_pool.clone())
                .unwrap_or_else(|| state.config.default_pool.clone());
            if !state.config.pools.contains_key(&pool) {
                return Err(pool_error(400, "Unknown interactivity pool"));
            }
            (pool, state.views(Instant::now()))
        };
        for home in [true, false] {
            let mut eligible: HashSet<_> = views
                .iter()
                .filter(|v| (v.pool == pool) == home && v.eligible(&pool))
                .map(|v| v.worker_id)
                .collect();
            while !eligible.is_empty() {
                let selection = self
                    .select_worker(
                        request.context().id(),
                        request,
                        RoutingRequestParts::new(request),
                        phase,
                        true,
                        SelectionOptions {
                            affinity_worker,
                            migration_worker_ids: super::selection::intersect_allowed_workers(
                                Some(eligible.clone()),
                                migration_worker_ids.clone(),
                            ),
                            policy_class: request.metadata().get("policy-class").cloned(),
                            session_id: request
                                .agent_context
                                .as_ref()
                                .map(|c| c.session_id.clone()),
                        },
                    )
                    .await;
                let selection = match selection {
                    Ok(selection) => selection,
                    Err(error)
                        if is_exhausted_by_exclusions(&error)
                            || affinity_worker.is_some()
                            || pinned_worker_hint(phase, request.routing.as_ref()).is_some() =>
                    {
                        break;
                    }
                    Err(error) => return Err(error),
                };
                if query_only {
                    return Ok(selection);
                }
                let key = WorkerWithDpRank::new(selection.instance_id, selection.dp_rank);
                manager
                    .state
                    .lock()
                    .update_blocks(current_blocks(&self.chooser));
                if let Some(lease) = manager.admit(key, &pool) {
                    let mut selection = self
                        .select_worker(
                            request.context().id(),
                            request,
                            RoutingRequestParts::new(request),
                            phase,
                            false,
                            SelectionOptions {
                                affinity_worker: Some(key),
                                migration_worker_ids: migration_worker_ids.clone(),
                                policy_class: request.metadata().get("policy-class").cloned(),
                                session_id: request
                                    .agent_context
                                    .as_ref()
                                    .map(|c| c.session_id.clone()),
                            },
                        )
                        .await?;
                    selection.pool_lease = Some(lease);
                    return Ok(selection);
                }
                eligible.remove(&key.worker_id);
            }
        }
        tracing::debug!(endpoint=%manager.state.lock().config.endpoint, request_pool=%pool, "Pool routing rejected by frontend: no eligible capacity");
        Err(PoolCapacityRejection.into())
    }
}

#[cfg(test)]
mod tests;
