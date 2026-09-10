// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned, best-effort interactivity policy. Workers only publish telemetry.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::*;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_kv_router::protocols::{ActiveLoad, PotentialLoad};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pool {
    kv_fraction: f64,
    minimum: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PoolConfig {
    endpoint: String,
    pools: BTreeMap<String, Pool>,
    default_pool: String,
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
        for (name, pool) in &self.pools {
            anyhow::ensure!(
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
                "invalid pool name"
            );
            anyhow::ensure!(
                pool.kv_fraction.is_finite()
                    && pool.kv_fraction > 0.0
                    && pool.kv_fraction <= 1.0
                    && pool.minimum > 0,
                "pool KV fractions must be in (0, 1] and minima positive"
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

struct RankReport {
    waiting: u64,
    kv_used: Option<u64>,
    revision: u64,
    received: Instant,
}

struct Member {
    stable_id: String,
    pool: String,
    target: Option<String>,
    changed: Instant,
    rank_start: u32,
    rank_count: u32,
    decode_blocks: HashMap<u32, u64>,
    kv_total: Option<u64>,
    reports: HashMap<u32, RankReport>,
    local: HashSet<u64>,
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
    effective_cap: u64,
    kv_fraction: f64,
    waiting: u64,
    kv_used_blocks: u64,
    kv_total_blocks: Option<u64>,
    available_by_pool: BTreeMap<String, u64>,
}

impl Member {
    fn rank_occupied(&self, rank: u32) -> u64 {
        self.decode_blocks.get(&rank).copied().unwrap_or(0)
    }

    fn budget(&self, fraction: f64) -> u64 {
        (self.kv_total.unwrap_or(0) as f64 * self.rank_count as f64 * fraction).floor() as u64
    }

    fn view(&self, id: u64, config: &PoolConfig, now: Instant) -> View {
        let ranks = self.rank_start..self.rank_start.saturating_add(self.rank_count);
        let kv_total = self
            .kv_total
            .unwrap_or(0)
            .saturating_mul(self.rank_count as u64);
        let healthy = kv_total > 0
            && self.decode_blocks.len() == self.rank_count as usize
            && ranks.clone().all(|rank| {
                self.reports.get(&rank).is_some_and(|r| {
                    now.duration_since(r.received).as_secs_f64() <= config.telemetry_ttl_seconds
                })
            });
        let effective_cap = self.budget(config.pools[&self.pool].kv_fraction);
        let occupied = ranks.clone().map(|rank| self.rank_occupied(rank)).sum();
        let available_by_pool = config
            .pools
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    if name == &self.pool && healthy && self.target.is_none() {
                        effective_cap.saturating_sub(occupied)
                    } else {
                        0
                    },
                )
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
            effective_cap,
            kv_fraction: occupied as f64 / kv_total.max(1) as f64,
            waiting: self.reports.values().map(|r| r.waiting).sum(),
            kv_used_blocks: self.reports.values().filter_map(|r| r.kv_used).sum(),
            kv_total_blocks: Some(kv_total),
            available_by_pool,
        }
    }
}

struct State {
    config: PoolConfig,
    members: HashMap<u64, Member>,
    pressure_since: HashMap<String, Instant>,
    next_request: u64,
    shortage: bool,
    membership_changed: bool,
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
                        stable_id,
                        pool,
                        target: None,
                        changed: now,
                        rank_start: config.data_parallel_start_rank,
                        rank_count: config.data_parallel_size.max(1),
                        decode_blocks: HashMap::new(),
                        kv_total: config.total_kv_blocks,
                        reports: HashMap::new(),
                        local: HashSet::new(),
                    },
                );
                added = true;
            }
            if let Some(member) = self.members.get_mut(id) {
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
            tracing::info!(endpoint=%self.config.endpoint, workers=self.members.len(), pool_routing_enabled=self.members.len()>1, "Pool fleet membership changed");
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
        let Some(revision) = load.load_report_revision else {
            return;
        };
        if member
            .reports
            .get(&load.dp_rank)
            .is_some_and(|old| old.revision >= revision)
        {
            return;
        }
        // Only worker observations refresh freshness; scheduler events have no revision.
        // A transport heartbeat replay cannot make stale scheduler data fresh.
        member.reports.insert(
            load.dp_rank,
            RankReport {
                waiting: load.num_waiting_reqs.unwrap_or(0),
                kv_used: load.kv_used_blocks,
                revision,
                received: now,
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
            .map(|(id, member)| member.view(*id, &self.config, now))
            .collect();
        let mut seen = HashMap::new();
        for view in &views {
            *seen.entry(view.stable_id.clone()).or_insert(0) += 1;
        }
        for view in &mut views {
            if seen[&view.stable_id] != 1 {
                view.healthy = false;
                view.available_by_pool.values_mut().for_each(|v| *v = 0);
            }
        }
        views.sort_by(|a, b| a.stable_id.cmp(&b.stable_id));
        views
    }

    fn rebalance(&mut self, now: Instant) {
        self.shortage = false;
        let views = self.views(now);
        if views.len() <= 1 || views.iter().any(|v| !v.healthy) {
            self.pressure_since.clear();
            return;
        }
        if let Some(draining) = views.iter().find(|v| v.target.is_some()) {
            self.pressure_since.clear();
            if draining.occupied == 0 && draining.local_requests == 0 {
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
                    .filter(|v| &v.pool == donor)
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
        let occupancy: BTreeMap<_, f64> = groups
            .iter()
            .map(|(name, group)| {
                (
                    *name,
                    group.iter().map(|v| v.occupied).sum::<u64>() as f64
                        / group.iter().map(|v| v.effective_cap).sum::<u64>() as f64,
                )
            })
            .collect();
        self.shortage = occupancy.values().all(|v| *v > self.config.threshold);
        let names: Vec<_> = self.config.pools.keys().cloned().collect();
        for i in 0..2 {
            let recipient = &names[i];
            let donor = &names[1 - i];
            if occupancy[recipient] > self.config.threshold
                && occupancy[donor] < self.config.threshold
            {
                self.pressure_since.entry(recipient.clone()).or_insert(now);
            } else {
                self.pressure_since.remove(recipient);
            }
        }
        if self.shortage
            || self
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
            let mut candidates = groups[donor].clone();
            candidates.sort_by_key(|v| (v.occupied, &v.stable_id));
            let total_cap: u64 = candidates.iter().map(|v| v.effective_cap).sum();
            let total_occupied: u64 = candidates.iter().map(|v| v.occupied).sum();
            for candidate in candidates {
                if (total_occupied as f64)
                    < self.config.threshold * (total_cap - candidate.effective_cap) as f64
                {
                    let Some(member) = self.members.get_mut(&candidate.worker_id) else {
                        continue;
                    };
                    tracing::info!(endpoint=%self.config.endpoint, worker_id=candidate.worker_id, stable_id=%member.stable_id, source=%member.pool, target=%recipient, "Pool drain started in frontend");
                    member.target = Some(recipient.clone());
                    member.changed = now;
                    self.pressure_since.clear();
                    return;
                }
            }
        }
    }
}

pub(super) struct PoolManager {
    state: Mutex<State>,
}

/// Local accounting only. Dropping the frontend stream releases this record;
/// backend telemetry may continue reporting the request until cancellation finishes.
pub(super) struct PoolLease {
    manager: Weak<PoolManager>,
    worker: u64,
    request: u64,
}
impl Drop for PoolLease {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade()
            && let Some(member) = manager.state.lock().members.get_mut(&self.worker)
        {
            member.local.remove(&self.request);
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
        let manager = Arc::new(Self {
            state: Mutex::new(State::new(config)),
        });
        let weak = Arc::downgrade(&manager);
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
                        if weak.strong_count() == 0 {
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
                    let Some(manager) = weak.upgrade() else {
                        return;
                    };
                    let status = {
                        let now = Instant::now();
                        let mut state = manager.state.lock();
                        state.reconcile(&configs.borrow(), now);
                        let Some(chooser) = weak_chooser.upgrade() else {
                            return;
                        };
                        state.update_blocks(current_blocks(&chooser));
                        if let Some(event) = event {
                            state.report(event, now);
                        }
                        state.rebalance(now);
                        serde_json::json!({"workers": state.views(now), "capacity_shortage": state.shortage, "authority": "frontend-local-best-effort", "endpoint": endpoint_name, "pool_routing_enabled": state.members.len()>1})
                    };
                    let status = status.to_string();
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
        });
        Ok(Some(manager))
    }

    fn admit(self: &Arc<Self>, worker: WorkerWithDpRank, pool: &str) -> Option<PoolLease> {
        let mut state = self.state.lock();
        let view = state
            .views(Instant::now())
            .into_iter()
            .find(|v| v.worker_id == worker.worker_id)?;
        if view.available_by_pool.get(pool).copied().unwrap_or(0) == 0 {
            return None;
        }
        state.next_request += 1;
        let request = state.next_request;
        let member = state.members.get_mut(&worker.worker_id)?;
        member.local.insert(request);
        tracing::debug!(endpoint=%state.config.endpoint, worker_id=worker.worker_id, dp_rank=worker.dp_rank, request_pool=pool, node_pool=%view.pool, active_decode_blocks=view.occupied, kv_fraction=view.kv_fraction, budget_blocks=view.effective_cap, "Pool routing admitted by frontend");
        Some(PoolLease {
            manager: Arc::downgrade(self),
            worker: worker.worker_id,
            request,
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
    pub(super) fn pools_bypassed(&self) -> bool {
        self.pools.is_some() && self.chooser.workers_with_configs.borrow().len() == 1
    }

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
            state.reconcile(&self.chooser.workers_with_configs.borrow(), Instant::now());
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
        let mut eligible: HashSet<_> = views
            .iter()
            .filter(|v| v.pool == pool && v.available_by_pool[&pool] > 0)
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
                        session_id: request.agent_context.as_ref().map(|c| c.session_id.clone()),
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
        tracing::debug!(endpoint=%manager.state.lock().config.endpoint, request_pool=%pool, "Pool routing rejected by frontend: no eligible capacity");
        Err(PoolCapacityRejection.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> (State, Instant) {
        let config: PoolConfig = serde_json::from_value(serde_json::json!({
            "endpoint": "pooldemo.worker.generate",
            "pools": {"interactive": {"kv_fraction": 0.1, "minimum": 1}, "throughput": {"kv_fraction": 0.4, "minimum": 1}},
            "default_pool": "throughput", "sustained_seconds": 4, "cooldown_seconds": 5
        })).unwrap();
        config.validate().unwrap();
        let now = Instant::now();
        let mut state = State::new(config);
        for (id, pool) in [(1, "interactive"), (2, "throughput"), (3, "throughput")] {
            state.members.insert(
                id,
                Member {
                    stable_id: id.to_string(),
                    pool: pool.into(),
                    target: None,
                    changed: now - Duration::from_secs(100),
                    rank_start: 0,
                    rank_count: 2,
                    decode_blocks: HashMap::from([(0, 0), (1, 0)]),
                    kv_total: Some(100),
                    reports: (0..2)
                        .map(|rank| {
                            (
                                rank,
                                RankReport {
                                    waiting: 0,
                                    kv_used: Some(0),
                                    revision: 1,
                                    received: now,
                                },
                            )
                        })
                        .collect(),
                    local: HashSet::new(),
                },
            );
        }
        (state, now)
    }

    #[test]
    fn interactivity_registration_and_removal_balance_workers_not_ranks() {
        let (fixture, now) = state();
        let mut state = State::new(fixture.config);
        let mut configs = HashMap::new();
        for id in 1..=3 {
            configs.insert(
                id,
                ModelRuntimeConfig {
                    data_parallel_size: 2,
                    total_kv_blocks: Some(100),
                    ..Default::default()
                },
            );
            state.reconcile(&configs, now);
            let first = state
                .members
                .values()
                .filter(|m| m.pool == "interactive")
                .count();
            assert!(first.abs_diff(state.members.len() - first) <= 1);
        }
        assert_eq!(state.members[&1].pool, "throughput");
        assert_eq!(state.members[&2].pool, "interactive");
        configs.remove(&2);
        state.reconcile(&configs, now);
        for member in state.members.values_mut() {
            member.decode_blocks = HashMap::from([(0, 0), (1, 0)]);
            member.reports = (0..2)
                .map(|rank| {
                    (
                        rank,
                        RankReport {
                            waiting: 0,
                            kv_used: Some(0),
                            revision: 1,
                            received: now,
                        },
                    )
                })
                .collect();
        }
        state.rebalance(now);
        state.rebalance(now);
        assert_eq!(
            state
                .members
                .values()
                .filter(|m| m.pool == "interactive")
                .count(),
            1
        );
    }

    #[test]
    fn interactivity_block_budget_aggregates_dp_and_ignores_request_count_and_kv_used() {
        let (mut state, now) = state();
        let member = state.members.get_mut(&1).unwrap();
        member.decode_blocks = HashMap::from([(0, 11), (1, 9)]);
        member.reports.get_mut(&0).unwrap().kv_used = Some(99);
        let view = member.view(1, &state.config, now);
        assert_eq!(view.occupied, 20);
        assert_eq!(view.kv_total_blocks, Some(200));
        assert_eq!(view.effective_cap, 20);
        assert_eq!(view.available_by_pool["interactive"], 0);
        member.decode_blocks.insert(1, 8);
        assert_eq!(
            member.view(1, &state.config, now).available_by_pool["interactive"],
            1
        );
    }

    #[test]
    fn interactivity_rejects_other_pool_even_with_idle_capacity() {
        let (mut state, now) = state();
        state
            .members
            .get_mut(&1)
            .unwrap()
            .decode_blocks
            .insert(0, 20);
        let views = state.views(now);
        assert_eq!(views[0].available_by_pool["interactive"], 0);
        assert_eq!(views[1].available_by_pool["interactive"], 0);
        assert_eq!(views[1].available_by_pool["throughput"], 80);
        let manager = Arc::new(PoolManager {
            state: Mutex::new(state),
        });
        for rank in 0..2 {
            assert!(
                manager
                    .admit(WorkerWithDpRank::new(1, rank), "interactive")
                    .is_none()
            );
            assert!(
                manager
                    .admit(WorkerWithDpRank::new(2, rank), "interactive")
                    .is_none()
            );
            assert!(
                manager
                    .admit(WorkerWithDpRank::new(1, rank), "throughput")
                    .is_none()
            );
        }
        let lease = manager
            .admit(WorkerWithDpRank::new(2, 0), "throughput")
            .unwrap();
        assert_eq!(manager.state.lock().members[&2].local.len(), 1);
        drop(lease);
        assert!(manager.state.lock().members[&2].local.is_empty());
    }

    #[test]
    fn interactivity_requires_fresh_workers_all_ranks_and_known_capacity() {
        let (mut state, now) = state();
        assert!(state.views(now)[0].healthy);
        let later = now + Duration::from_secs(6);
        state.report(
            ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                kv_used_blocks: Some(0),
                load_report_revision: Some(1),
                ..Default::default()
            },
            later,
        );
        assert!(!state.views(later)[0].healthy);
        state.report(
            ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                kv_used_blocks: Some(0),
                load_report_revision: Some(2),
                ..Default::default()
            },
            later,
        );
        assert!(!state.views(later)[0].healthy); // Rank 1 remains stale.
        let member = state.members.get_mut(&1).unwrap();
        member.kv_total = None;
        assert!(!member.view(1, &state.config, now).healthy);
    }

    #[test]
    fn interactivity_pressure_moves_both_directions_and_preserves_minimum() {
        let (mut state, now) = state();
        state
            .members
            .get_mut(&1)
            .unwrap()
            .decode_blocks
            .insert(0, 20);
        state.rebalance(now);
        let later = now + Duration::from_secs(4);
        state.rebalance(later);
        assert_eq!(state.members[&2].target.as_deref(), Some("interactive"));
        assert_eq!(state.views(later)[1].available_by_pool["throughput"], 0);
        state.rebalance(later);
        assert_eq!(state.members[&2].pool, "interactive");
        // Refresh observations, clear interactive load, and fill throughput.
        let reverse = now + Duration::from_secs(10);
        for member in state.members.values_mut() {
            member.decode_blocks = HashMap::from([(0, 0), (1, 0)]);
            for report in member.reports.values_mut() {
                report.received = reverse;
            }
        }
        state
            .members
            .get_mut(&3)
            .unwrap()
            .decode_blocks
            .insert(0, 80);
        state.rebalance(reverse);
        state.rebalance(reverse + Duration::from_secs(4));
        state.rebalance(reverse + Duration::from_secs(4));
        assert_eq!(
            state
                .members
                .values()
                .filter(|m| m.pool == "throughput")
                .count(),
            2
        );
        for member in state
            .members
            .values_mut()
            .filter(|m| m.pool == "throughput")
        {
            member.decode_blocks.insert(0, 80);
        }
        state.rebalance(reverse + Duration::from_secs(4));
        assert!(state.members.values().all(|m| m.target.is_none()));
    }

    #[test]
    fn interactivity_drain_waits_for_blocks_and_local_streams() {
        let (mut state, now) = state();
        let member = state.members.get_mut(&2).unwrap();
        member.target = Some("interactive".into());
        member.decode_blocks.insert(0, 1);
        state.rebalance(now);
        assert_eq!(state.members[&2].pool, "throughput");
        let member = state.members.get_mut(&2).unwrap();
        member.decode_blocks.insert(0, 0);
        member.local.insert(1);
        state.rebalance(now);
        assert_eq!(state.members[&2].pool, "throughput");
        state.members.get_mut(&2).unwrap().local.clear();
        state.rebalance(now);
        assert_eq!(state.members[&2].pool, "interactive");
    }

    #[test]
    fn interactivity_endpoints_have_independent_blocks_and_membership() {
        let (mut first, now) = state();
        let (second, _) = state();
        first
            .members
            .get_mut(&1)
            .unwrap()
            .decode_blocks
            .insert(0, 20);
        first.rebalance(now);
        first.rebalance(now + Duration::from_secs(4));
        first.rebalance(now + Duration::from_secs(4));
        assert_eq!(first.members[&2].pool, "interactive");
        assert_eq!(second.members[&2].pool, "throughput");
        assert_eq!(second.views(now)[0].occupied, 0);
    }

    #[test]
    fn interactivity_config_rejects_invalid_fractions_and_duplicate_endpoints() {
        let (fixture, _) = state();
        for fraction in [0.0, -0.1, 1.1, f64::NAN] {
            let mut config = fixture.config.clone();
            config.pools.get_mut("interactive").unwrap().kv_fraction = fraction;
            assert!(config.validate().is_err());
        }
        let config = serde_json::json!({"endpoint": "demo.worker.generate", "default_pool": "throughput",
            "pools": {"interactive": {"kv_fraction": 0.1, "minimum": 1}, "throughput": {"kv_fraction": 0.4, "minimum": 1}}});
        assert!(
            parse_configs(&serde_json::to_vec(&serde_json::json!([config, config])).unwrap())
                .is_err()
        );
        let mut second = config.clone();
        second["endpoint"] = "demo.other.generate".into();
        assert_eq!(
            parse_configs(&serde_json::to_vec(&serde_json::json!([config, second])).unwrap())
                .unwrap()
                .len(),
            2
        );
    }
}
