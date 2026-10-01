// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The virtual-time event loop: workers, proxies, router signals and request lifecycle.
//!
//! Time advances only to the next event, so a scenario runs in milliseconds while modelling
//! queues, first-token latencies and cache residency in seconds. All randomness comes from
//! seeded `fastrand` generators, so identical inputs give identical reports.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use dw_spillover_testkit as testkit;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use fastrand::Rng;

use crate::config::Scenario;
use crate::hash::block_hashes;
use crate::report::{RequestRecord, RunData};
use crate::selector::{SelectionInput, Selector};
use crate::workload::Workload;

/// LRU prefix cache over block hashes.
struct BlockCache {
    capacity: usize,
    order: VecDeque<u64>,
    present: HashSet<u64>,
}

impl BlockCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
            present: HashSet::new(),
        }
    }

    /// Number of leading consecutive blocks already cached.
    fn overlap(&self, blocks: &[u64]) -> usize {
        blocks
            .iter()
            .take_while(|hash| self.present.contains(hash))
            .count()
    }

    fn insert_all(&mut self, blocks: &[u64]) {
        for &hash in blocks {
            if self.present.contains(&hash) {
                if let Some(position) = self.order.iter().position(|&value| value == hash) {
                    self.order.remove(position);
                }
            } else {
                self.present.insert(hash);
            }
            self.order.push_back(hash);
        }
        while self.order.len() > self.capacity {
            if let Some(evicted) = self.order.pop_front()
                && !self.order.contains(&evicted)
            {
                self.present.remove(&evicted);
            }
        }
    }
}

/// Proxy virtual cache: block hash to expiry time.
struct ProxyCache {
    ttl: f64,
    expires: HashMap<u64, f64>,
}

impl ProxyCache {
    fn new(ttl: f64) -> Self {
        Self {
            ttl,
            expires: HashMap::new(),
        }
    }

    fn overlap(&self, blocks: &[u64], now: f64) -> usize {
        blocks
            .iter()
            .take_while(|hash| self.expires.get(hash).is_some_and(|expiry| *expiry >= now))
            .count()
    }

    fn insert_all(&mut self, blocks: &[u64], now: f64) {
        self.expires.retain(|_, expiry| *expiry >= now);
        for &hash in blocks {
            self.expires.insert(hash, now + self.ttl);
        }
    }
}

struct PrimaryWorker {
    id: u64,
    online: bool,
    capacity_blocks: usize,
    prefill_rate: f64,
    decode_rate: f64,
    max_concurrent: usize,
    slowdown: f64,
    cache: BlockCache,
    active: Vec<u64>,
    queue: VecDeque<u64>,
}

struct ProxyWorker {
    worker_id: u64,
    tier: String,
    ttft: f64,
    ttft_jitter: f64,
    decode: f64,
    decode_jitter: f64,
    limit: Option<usize>,
    cache: ProxyCache,
    active: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum WorkerRef {
    Primary(usize),
    Proxy(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Queued,
    Prefilling,
    Decoding,
}

struct ActiveRequest {
    worker: WorkerRef,
    worker_key: WorkerWithDpRank,
    phase: Phase,
    remaining_prefill_tokens: usize,
    output_tokens: usize,
    /// Complete prompt block hashes of this request, fixed at admission, the way the
    /// router's `BlockTracker::acquire_prompt` sees them.
    prompt_blocks: Vec<u64>,
    decode_rate: f64,
    session: usize,
    turn: usize,
    prompt: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EventKind {
    TurnStart { session: usize, turn: usize },
    PrefillDone(u64),
    RequestDone(u64),
    PrimaryChange { online: bool },
}

struct Event {
    time: f64,
    seq: u64,
    kind: EventKind,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so `BinaryHeap` pops the earliest event.
        other
            .time
            .total_cmp(&self.time)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Common facts about one decision, shared by the primary and proxy start paths.
struct DecisionContext {
    arrival_time: f64,
    session: usize,
    turn: usize,
    prompt: Vec<u32>,
    prompt_blocks: Vec<u64>,
    output_tokens: usize,
    cache_hit_blocks: usize,
    occupancy: f64,
    is_followup: bool,
    previous_under_threshold: bool,
    attempts: usize,
    /// Primary workers excluded from this decision because their engine queue was at
    /// or above the admission margin.
    steering_excluded: usize,
}

pub struct Engine<'s> {
    scenario: &'s Scenario,
    selector: &'s mut dyn Selector,
    workload: Workload,
    time: f64,
    duration: f64,
    block_size: u32,
    events: BinaryHeap<Event>,
    seq: u64,
    service_rng: Rng,
    primary: Vec<PrimaryWorker>,
    proxies: Vec<ProxyWorker>,
    workers: HashMap<u64, testkit::SimWorker>,
    active: HashMap<u64, ActiveRequest>,
    next_id: u64,
    conversation: Vec<Vec<u32>>,
    last_worker: Vec<Option<WorkerWithDpRank>>,
    last_is_proxy: Vec<Option<bool>>,
    records: Vec<RequestRecord>,
    occupancy_samples: Vec<(f64, f64)>,
    spill_first: std::collections::BTreeMap<String, f64>,
    decision_trace: Vec<String>,
    rate_limited_count: usize,
}

impl<'s> Engine<'s> {
    pub fn new(scenario: &'s Scenario, selector: &'s mut dyn Selector) -> Self {
        let workload = Workload::generate(scenario);
        let mut primary = Vec::new();
        let mut workers = HashMap::new();
        for config in &scenario.primary {
            primary.push(PrimaryWorker {
                id: config.id,
                online: true,
                capacity_blocks: config.capacity_blocks,
                prefill_rate: config.prefill_tokens_per_second,
                decode_rate: config.decode_tokens_per_second,
                max_concurrent: config.max_concurrent_requests,
                slowdown: config.batching_slowdown,
                cache: BlockCache::new(config.capacity_blocks),
                active: Vec::new(),
                queue: VecDeque::new(),
            });
            workers.insert(
                config.id,
                testkit::SimWorker::primary_with_seq_capacity(
                    config.capacity_blocks as u64,
                    config.max_concurrent_requests as u64,
                ),
            );
        }
        let mut proxies = Vec::new();
        for config in &scenario.proxies {
            for offset in 0..config.workers {
                let dp_rank = config.dp_rank_start + offset as u32;
                let worker_id = dp_rank as u64;
                proxies.push(ProxyWorker {
                    worker_id,
                    tier: config.tier.clone(),
                    ttft: config.ttft_seconds,
                    ttft_jitter: config.ttft_jitter,
                    decode: config.decode_tokens_per_second,
                    decode_jitter: config.decode_jitter,
                    limit: config.concurrency_limit,
                    cache: ProxyCache::new(config.cache_ttl_seconds),
                    active: Vec::new(),
                });
                workers.insert(worker_id, testkit::SimWorker::proxy(dp_rank));
            }
        }

        let mut engine = Self {
            scenario,
            selector,
            conversation: vec![workload.system_tokens.clone(); workload.sessions.len()],
            last_worker: vec![None; workload.sessions.len()],
            last_is_proxy: vec![None; workload.sessions.len()],
            workload,
            time: 0.0,
            duration: scenario.duration_seconds,
            block_size: scenario.block_size,
            events: BinaryHeap::new(),
            seq: 0,
            service_rng: Rng::with_seed(scenario.seed ^ 0x5eed_1234),
            primary,
            proxies,
            workers,
            active: HashMap::new(),
            next_id: 0,
            records: Vec::new(),
            occupancy_samples: vec![(0.0, 0.0)],
            spill_first: std::collections::BTreeMap::new(),
            decision_trace: Vec::new(),
            rate_limited_count: 0,
        };
        let session_starts: Vec<(usize, f64)> = engine
            .workload
            .sessions
            .iter()
            .enumerate()
            .map(|(session, spec)| (session, spec.start_time))
            .collect();
        for (session, start_time) in session_starts {
            engine.schedule(start_time, EventKind::TurnStart { session, turn: 0 });
        }
        for change in &scenario.primary_online {
            engine.schedule(
                change.time,
                EventKind::PrimaryChange {
                    online: change.online,
                },
            );
        }
        engine
    }

    pub fn run(mut self) -> RunData {
        while let Some(event) = self.events.pop() {
            if event.time > self.duration {
                break;
            }
            self.time = event.time;
            match event.kind {
                EventKind::TurnStart { session, turn } => self.fire_turn(session, turn),
                EventKind::PrefillDone(id) => self.on_prefill_done(id),
                EventKind::RequestDone(id) => self.on_request_done(id),
                EventKind::PrimaryChange { online } => self.toggle_primary(online),
            }
        }
        RunData {
            records: self.records,
            occupancy_samples: self.occupancy_samples,
            spill_first: self.spill_first,
            decision_trace: self.decision_trace,
            total_rate_limited: self.rate_limited_count,
        }
    }

    fn schedule(&mut self, time: f64, kind: EventKind) {
        self.seq += 1;
        self.events.push(Event {
            time,
            seq: self.seq,
            kind,
        });
    }

    fn classify(&self, worker: WorkerWithDpRank) -> WorkerRef {
        if let Some(index) = self.primary.iter().position(|h| h.id == worker.worker_id) {
            WorkerRef::Primary(index)
        } else {
            let index = self
                .proxies
                .iter()
                .position(|p| p.worker_id == worker.worker_id)
                .expect("selected worker is neither primary nor a proxy");
            WorkerRef::Proxy(index)
        }
    }

    /// Highest per-worker primary occupancy, matching what the policy compares against the
    /// threshold. Occupancy is the larger of a worker's KV block fraction and its projected
    /// concurrency fraction. No request is in flight here, so the projected decode footprint
    /// omits the arriving request's own uncached blocks.
    fn primary_occupancy(&self) -> f64 {
        (0..self.primary.len())
            .filter(|index| self.primary[*index].online)
            .map(|index| self.primary_occupancy_for(index))
            .fold(0.0, f64::max)
    }

    /// Highest per-worker primary occupancy at decision time, from the same load signals the
    /// policy is handed (`build_request`): `decode_cost_blocks` is the worker's tracked active
    /// decode blocks plus the arriving request's own uncached blocks, and the concurrency signal
    /// counts every active request including queued ones, then projects the arriving request.
    fn primary_occupancy_with_blocks(&self, prompt_blocks: &[u64]) -> f64 {
        (0..self.primary.len())
            .filter(|index| self.primary[*index].online)
            .map(|index| self.primary_occupancy_for_blocks(index, prompt_blocks))
            .fold(0.0, f64::max)
    }

    /// Policy-visible occupancy of one primary worker with no arriving request. Used for the
    /// between-decision samples (`on_request_done`, `on_prefill_done`, `toggle_primary`).
    fn primary_occupancy_for(&self, index: usize) -> f64 {
        self.primary_occupancy_for_blocks(index, &[])
    }

    /// The occupancy the policy sees for one primary worker when `prompt_blocks` is the
    /// arriving request: the larger of its projected KV block fraction
    /// (`active_decode_blocks + the request's additional active blocks` / advertised capacity)
    /// and its projected concurrency fraction (`active_requests + 1` / advertised `max_num_seqs`).
    /// Those are exactly the `load_signals` and `additional_active_blocks` `build_request`
    /// hands the selector. An offline worker has no available capacity, so its occupancy is 0.
    fn primary_occupancy_for_blocks(&self, index: usize, prompt_blocks: &[u64]) -> f64 {
        let worker = &self.primary[index];
        if !worker.online {
            return 0.0;
        }
        let signals = self.load_signals(self.primary_key(index));
        let membership = self.active_prompt_union(WorkerRef::Primary(index));
        let active_overlap = prompt_blocks
            .iter()
            .take_while(|hash| membership.contains(hash))
            .count();
        let additional_active_blocks = prompt_blocks.len().saturating_sub(active_overlap);
        let decode_blocks = (signals.active_decode_blocks + additional_active_blocks) as f64;
        let kv =
            (worker.capacity_blocks > 0).then(|| decode_blocks / worker.capacity_blocks as f64);
        let concurrency = (worker.max_concurrent > 0)
            .then(|| (signals.active_requests as f64 + 1.0) / worker.max_concurrent as f64);
        match (kv, concurrency) {
            (Some(kv), Some(concurrency)) => kv.max(concurrency),
            (Some(kv), None) => kv,
            (None, Some(concurrency)) => concurrency,
            (None, None) => 0.0,
        }
    }

    /// The router rank of a primary worker, for building its load signals. Offline workers are
    /// absent from `self.workers`, but their occupancy short-circuits before this is reached.
    fn primary_key(&self, index: usize) -> WorkerWithDpRank {
        let id = self.primary[index].id;
        let dp_rank = self
            .workers
            .get(&id)
            .map(|worker| worker.dp_start_rank)
            .unwrap_or(0);
        WorkerWithDpRank::new(id, dp_rank)
    }

    /// Per-worker union of the complete prompt block hashes of every request the router
    /// has admitted to that worker, mirroring `BlockTracker::active_blocks` with output-block
    /// tracking off: complete blocks only, a prefix shared by concurrent requests counted
    /// once, and prompt blocks acquired at admission (`add_request_with_prefill_tracking`
    /// calls `acquire_prompt`), so prefilling and engine-queued requests contribute too.
    fn active_prompt_union(&self, worker: WorkerRef) -> HashSet<u64> {
        let mut union = HashSet::new();
        for request in self.active.values() {
            if request.worker != worker {
                continue;
            }
            union.extend(request.prompt_blocks.iter().copied());
        }
        union
    }

    /// Engine-waiting requests on a primary worker: admitted but not yet running, matching
    /// the waiting count the real worker reports (`report_engine_waiting`).
    fn primary_queue_depth(&self, index: usize) -> usize {
        self.active
            .values()
            .filter(|request| {
                request.worker == WorkerRef::Primary(index) && request.phase == Phase::Queued
            })
            .count()
    }

    /// Primary workers whose engine queue is at or above their admission margin. Only primary
    /// workers are eligible: proxies never report engine waiting, so their estimate stays
    /// unenforced, exactly as in the fork.
    fn steering_exclusions(&self) -> HashSet<u64> {
        let Some(admission) = &self.scenario.admission else {
            return HashSet::new();
        };
        self.primary
            .iter()
            .enumerate()
            .filter(|(index, worker)| {
                worker.online
                    && self.primary_queue_depth(*index) as u64 >= admission.margin_for(worker.id)
            })
            .map(|(_, worker)| worker.id)
            .collect()
    }

    fn under_threshold(&self, worker: WorkerWithDpRank, prompt_blocks: &[u64]) -> bool {
        match self.classify(worker) {
            WorkerRef::Primary(index) => {
                self.primary[index].online
                    && self.primary_occupancy_for_blocks(index, prompt_blocks)
                        <= self.scenario.policy.occupancy_threshold
            }
            WorkerRef::Proxy(_) => true,
        }
    }

    fn load_signals(&self, worker: WorkerWithDpRank) -> testkit::RankSignals {
        let mut signals = testkit::RankSignals::default();
        for request in self.active.values() {
            if request.worker_key != worker {
                continue;
            }
            signals.active_requests += 1;
            if request.phase == Phase::Prefilling {
                signals.active_prefill_tokens += request.remaining_prefill_tokens;
            }
        }
        // The router's active decode footprint is the per-worker union of complete prompt
        // blocks, not a per-request sum of prompt+output blocks. `active_blocks` counts each
        // shared prefix edge once, output-block tracking is off, and a request's prompt blocks
        // are acquired at admission, so prefilling and engine-queued requests count too.
        signals.active_decode_blocks = self.active_prompt_union(self.classify(worker)).len();
        signals
    }

    fn overlap_for(&self, worker: WorkerWithDpRank, blocks: &[u64]) -> usize {
        match self.classify(worker) {
            WorkerRef::Primary(index) => self.primary[index].cache.overlap(blocks),
            WorkerRef::Proxy(index) => self.proxies[index].cache.overlap(blocks, self.time),
        }
    }

    fn build_request(
        &self,
        prompt_tokens: usize,
        prompt_blocks: &[u64],
        excluded: &HashSet<u64>,
    ) -> dynamo_kv_router::SchedulingRequest {
        let mut request = testkit::empty_request(prompt_tokens);
        for (worker_id, config) in &self.workers {
            let worker = WorkerWithDpRank::new(*worker_id, config.dp_start_rank);
            let mut signals = self.load_signals(worker);
            signals.device_overlap_blocks = self.overlap_for(worker, prompt_blocks);
            // `additional_active_blocks` is the arriving request's blocks not already shared
            // with a live sequence on this worker (`query_len - membership overlap depth`).
            // The router takes it from the active-membership trie, not the device KV cache,
            // so a cache-resident prefix that no live request shares still counts in full.
            let membership = self.active_prompt_union(self.classify(worker));
            let active_overlap = prompt_blocks
                .iter()
                .take_while(|hash| membership.contains(hash))
                .count();
            signals.additional_active_blocks = prompt_blocks.len().saturating_sub(active_overlap);
            testkit::set_rank(&mut request, worker, signals, self.block_size);
        }
        if !excluded.is_empty() {
            request.allowed_worker_ids = Some(
                self.workers
                    .keys()
                    .copied()
                    .filter(|worker_id| !excluded.contains(worker_id))
                    .collect(),
            );
        }
        request
    }

    fn fire_turn(&mut self, session: usize, turn: usize) {
        if turn >= self.workload.sessions[session].turns.len() {
            return;
        }
        let mut prompt = self.conversation[session].clone();
        let turn_spec = &self.workload.sessions[session].turns[turn];
        prompt.extend_from_slice(&turn_spec.user_tokens);
        let output_tokens = turn_spec.output_tokens;
        self.route(session, turn, prompt, output_tokens);
    }

    fn route(&mut self, session: usize, turn: usize, prompt: Vec<u32>, output_tokens: usize) {
        let now = self.time;
        let prompt_tokens = prompt.len();
        let prompt_blocks = block_hashes(&prompt, self.block_size as usize);
        let previous = self.last_worker[session];
        let occupancy = self.primary_occupancy_with_blocks(&prompt_blocks);
        let is_followup = turn > 0;
        let previous_under_threshold = match previous {
            None => false,
            Some(worker) => self.under_threshold(worker, &prompt_blocks),
        };
        // Production routes every turn through the policy with no session pin, so stickiness
        // must come from cache overlap alone, which is what the stickiness scenario measures.
        let saturated = self.steering_exclusions();
        let steering_excluded = saturated.len();
        let mut excluded: HashSet<u64> = saturated;
        let mut attempts = 0usize;
        self.occupancy_samples.push((now, occupancy));

        loop {
            let request = self.build_request(prompt_tokens, &prompt_blocks, &excluded);
            let selected = {
                let input = SelectionInput {
                    request: &request,
                    workers: &self.workers,
                    block_size: self.block_size,
                };
                self.selector.select(&input)
            };
            let Some(worker) = selected else {
                self.record_failure(DecisionContext {
                    arrival_time: now,
                    session,
                    turn,
                    prompt,
                    prompt_blocks,
                    output_tokens,
                    cache_hit_blocks: 0,
                    occupancy,
                    is_followup,
                    previous_under_threshold,
                    attempts,
                    steering_excluded,
                });
                return;
            };
            match self.classify(worker) {
                WorkerRef::Primary(index) => {
                    let cache_hit_blocks = self.overlap_for(worker, &prompt_blocks);
                    self.start_primary(
                        index,
                        worker,
                        DecisionContext {
                            arrival_time: now,
                            session,
                            turn,
                            prompt,
                            prompt_blocks,
                            output_tokens,
                            cache_hit_blocks,
                            occupancy,
                            is_followup,
                            previous_under_threshold,
                            attempts,
                            steering_excluded,
                        },
                    );
                    return;
                }
                WorkerRef::Proxy(index) => {
                    if self.proxy_has_capacity(index) {
                        let cache_hit_blocks = self.overlap_for(worker, &prompt_blocks);
                        self.start_proxy(
                            index,
                            worker,
                            DecisionContext {
                                arrival_time: now,
                                session,
                                turn,
                                prompt,
                                prompt_blocks,
                                output_tokens,
                                cache_hit_blocks,
                                occupancy,
                                is_followup,
                                previous_under_threshold,
                                attempts,
                                steering_excluded,
                            },
                        );
                        return;
                    }
                    attempts += 1;
                    self.rate_limited_count += 1;
                    excluded.insert(worker.worker_id);
                    if self.workers.keys().all(|id| excluded.contains(id)) {
                        self.record_failure(DecisionContext {
                            arrival_time: now,
                            session,
                            turn,
                            prompt,
                            prompt_blocks,
                            output_tokens,
                            cache_hit_blocks: 0,
                            occupancy,
                            is_followup,
                            previous_under_threshold,
                            attempts,
                            steering_excluded,
                        });
                        return;
                    }
                }
            }
        }
    }

    fn record_decision(
        &mut self,
        class: &str,
        is_proxy: bool,
        worker: WorkerWithDpRank,
        context: &DecisionContext,
        cache_hit_tokens: u64,
    ) {
        let sticky = self.last_worker[context.session] == Some(worker);
        // Class stickiness counts a follow-up that returns to the previous turn's class whenever
        // that class is still a sensible target: any proxy, or a primary worker under threshold.
        let class_sticky = context.is_followup
            && context.previous_under_threshold
            && self.last_is_proxy[context.session] == Some(is_proxy);
        self.records.push(RequestRecord {
            arrival_time: context.arrival_time,
            class: class.to_string(),
            is_proxy,
            prompt_tokens: context.prompt.len() as u64,
            cache_hit_tokens,
            is_followup: context.is_followup,
            sticky,
            class_sticky,
            previous_under_threshold: context.previous_under_threshold,
            primary_occupancy_at_selection: context.occupancy,
            failed: false,
            rate_limited_attempts: context.attempts,
            steering_excluded: context.steering_excluded,
            admission_529: false,
        });
        self.decision_trace
            .push(format!("{}:{}", worker.worker_id, worker.dp_rank));
        self.last_worker[context.session] = Some(worker);
        self.last_is_proxy[context.session] = Some(is_proxy);
        if is_proxy {
            self.spill_first
                .entry(class.to_string())
                .or_insert(context.arrival_time);
        }
    }

    fn record_failure(&mut self, context: DecisionContext) {
        self.records.push(RequestRecord {
            arrival_time: context.arrival_time,
            class: "none".to_string(),
            is_proxy: false,
            prompt_tokens: context.prompt.len() as u64,
            cache_hit_tokens: 0,
            is_followup: context.is_followup,
            sticky: false,
            class_sticky: false,
            previous_under_threshold: context.previous_under_threshold,
            primary_occupancy_at_selection: context.occupancy,
            failed: true,
            rate_limited_attempts: context.attempts,
            steering_excluded: context.steering_excluded,
            // A failure with steering exclusions means every primary worker was at its
            // margin and no proxy could take the request: the 529 refusal the gate causes.
            admission_529: context.steering_excluded > 0,
        });
        self.decision_trace.push("none".to_string());
    }

    fn start_primary(&mut self, index: usize, worker: WorkerWithDpRank, context: DecisionContext) {
        let block_size = self.block_size as usize;
        let cached_tokens = (context.cache_hit_blocks * block_size).min(context.prompt.len());
        self.record_decision("primary", false, worker, &context, cached_tokens as u64);
        self.primary[index].cache.insert_all(&context.prompt_blocks);
        let id = self.next_id;
        self.next_id += 1;
        let phase = if self.primary[index].active.len() < self.primary[index].max_concurrent {
            Phase::Prefilling
        } else {
            Phase::Queued
        };
        self.active.insert(
            id,
            ActiveRequest {
                worker: WorkerRef::Primary(index),
                worker_key: worker,
                phase,
                remaining_prefill_tokens: context.prompt.len() - cached_tokens,
                output_tokens: context.output_tokens,
                prompt_blocks: context.prompt_blocks,
                decode_rate: 0.0,
                session: context.session,
                turn: context.turn,
                prompt: context.prompt,
            },
        );
        if phase == Phase::Prefilling {
            self.primary[index].active.push(id);
            self.begin_prefill(index, id);
        } else {
            self.primary[index].queue.push_back(id);
        }
    }

    fn start_proxy(&mut self, index: usize, worker: WorkerWithDpRank, context: DecisionContext) {
        let block_size = self.block_size as usize;
        let cached_tokens = (context.cache_hit_blocks * block_size).min(context.prompt.len());
        let tier = self.proxies[index].tier.clone();
        self.record_decision(&tier, true, worker, &context, cached_tokens as u64);
        let ttft = sample_distribution(
            self.proxies[index].ttft,
            self.proxies[index].ttft_jitter,
            &mut self.service_rng,
        );
        let decode_rate = sample_distribution(
            self.proxies[index].decode,
            self.proxies[index].decode_jitter,
            &mut self.service_rng,
        );
        let id = self.next_id;
        self.next_id += 1;
        self.proxies[index].active.push(id);
        self.active.insert(
            id,
            ActiveRequest {
                worker: WorkerRef::Proxy(index),
                worker_key: worker,
                phase: Phase::Prefilling,
                remaining_prefill_tokens: context.prompt.len() - cached_tokens,
                output_tokens: context.output_tokens,
                prompt_blocks: context.prompt_blocks,
                decode_rate,
                session: context.session,
                turn: context.turn,
                prompt: context.prompt,
            },
        );
        self.schedule(self.time + ttft, EventKind::PrefillDone(id));
    }

    fn proxy_has_capacity(&self, index: usize) -> bool {
        self.proxies[index]
            .limit
            .is_none_or(|limit| self.proxies[index].active.len() < limit)
    }

    fn begin_prefill(&mut self, index: usize, id: u64) {
        let batch = self.primary[index].active.len();
        let rate = effective_rate(
            self.primary[index].prefill_rate,
            batch,
            self.primary[index].slowdown,
        );
        let remaining = self.active[&id].remaining_prefill_tokens;
        if remaining == 0 || rate <= 0.0 {
            self.begin_decode(index, id);
            return;
        }
        let duration = remaining as f64 / rate;
        self.schedule(self.time + duration, EventKind::PrefillDone(id));
    }

    fn begin_decode(&mut self, index: usize, id: u64) {
        let batch = self.primary[index].active.len();
        let rate = effective_rate(
            self.primary[index].decode_rate,
            batch,
            self.primary[index].slowdown,
        );
        let output_tokens = self.active[&id].output_tokens;
        self.active.get_mut(&id).unwrap().phase = Phase::Decoding;
        let duration = if rate > 0.0 {
            output_tokens as f64 / rate
        } else {
            0.0
        };
        self.schedule(self.time + duration, EventKind::RequestDone(id));
    }

    fn on_prefill_done(&mut self, id: u64) {
        let Some(request) = self.active.get(&id) else {
            return;
        };
        let worker = request.worker;
        let prompt_blocks = request.prompt_blocks.clone();
        match worker {
            WorkerRef::Primary(index) => {
                self.active.get_mut(&id).unwrap().remaining_prefill_tokens = 0;
                self.begin_decode(index, id);
            }
            WorkerRef::Proxy(index) => {
                // The real proxy publishes prompt blocks when prefill materializes them, not
                // at admission, so concurrent requests cannot hit blocks that do not exist yet.
                self.proxies[index]
                    .cache
                    .insert_all(&prompt_blocks, self.time);
                self.active.get_mut(&id).unwrap().phase = Phase::Decoding;
                self.active.get_mut(&id).unwrap().remaining_prefill_tokens = 0;
                let output_tokens = self.active[&id].output_tokens;
                let rate = self.active[&id].decode_rate;
                let duration = if rate > 0.0 {
                    output_tokens as f64 / rate
                } else {
                    0.0
                };
                self.schedule(self.time + duration, EventKind::RequestDone(id));
            }
        }
    }

    fn on_request_done(&mut self, id: u64) {
        let Some(request) = self.active.remove(&id) else {
            return;
        };
        let output = crate::hash::synth_tokens(
            &format!("out-{}-{}", request.session, request.turn),
            request.output_tokens,
        );
        let mut full = request.prompt.clone();
        full.extend_from_slice(&output);
        let full_blocks = block_hashes(&full, self.block_size as usize);
        match request.worker {
            WorkerRef::Primary(index) => {
                self.primary[index]
                    .active
                    .retain(|active_id| *active_id != id);
                self.primary[index].cache.insert_all(&full_blocks);
                self.admit_queued(index);
            }
            WorkerRef::Proxy(index) => {
                self.proxies[index]
                    .active
                    .retain(|active_id| *active_id != id);
            }
        }
        self.conversation[request.session] = full;
        if request.turn + 1 < self.workload.sessions[request.session].turns.len() {
            let delay = self.workload.sessions[request.session].turns[request.turn + 1].think_time;
            self.schedule(
                self.time + delay,
                EventKind::TurnStart {
                    session: request.session,
                    turn: request.turn + 1,
                },
            );
        }
        let occupancy = self.primary_occupancy();
        self.occupancy_samples.push((self.time, occupancy));
    }

    fn admit_queued(&mut self, index: usize) {
        while self.primary[index].active.len() < self.primary[index].max_concurrent {
            let Some(id) = self.primary[index].queue.pop_front() else {
                break;
            };
            self.active.get_mut(&id).unwrap().phase = Phase::Prefilling;
            self.primary[index].active.push(id);
            self.begin_prefill(index, id);
        }
    }

    fn toggle_primary(&mut self, online: bool) {
        for index in 0..self.primary.len() {
            self.primary[index].online = online;
            let id = self.primary[index].id;
            if online {
                self.workers.insert(
                    id,
                    testkit::SimWorker::primary_with_seq_capacity(
                        self.primary[index].capacity_blocks as u64,
                        self.primary[index].max_concurrent as u64,
                    ),
                );
            } else {
                self.workers.remove(&id);
            }
        }
        let occupancy = self.primary_occupancy();
        self.occupancy_samples.push((self.time, occupancy));
    }
}

/// Primary workers apply a fractional slowdown as concurrency grows.
fn effective_rate(base: f64, concurrency: usize, slowdown: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    base / (1.0 + slowdown * concurrency.saturating_sub(1) as f64)
}

/// Deterministic draw in `base * [1 - jitter, 1 + jitter]`.
fn sample_distribution(base: f64, jitter: f64, rng: &mut Rng) -> f64 {
    if jitter <= 0.0 {
        return base;
    }
    let value = base * (1.0 + jitter * (2.0 * rng.f64() - 1.0));
    value.max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selector::HeuristicSelector;

    fn scenario() -> Scenario {
        Scenario::parse(
            r#"
name: engine_unit
seed: 1
duration_seconds: 60
block_size: 16
arrival_rate:
  - { time: 0, rate: 0.1 }
primary:
  - id: 0
    capacity_blocks: 400
    prefill_tokens_per_second: 100000
    decode_tokens_per_second: 40
    max_concurrent_requests: 1
proxies:
  - tier: X
    dp_rank_start: 1000
    ttft_seconds: 0.1
    decode_tokens_per_second: 40
    cache_ttl_seconds: 60
workload:
  system_prompt_tokens: 64
  user_tokens: { min: 48, max: 48 }
  output_tokens: { min: 64, max: 64 }
  think_time_seconds: { min: 0.1, max: 0.1 }
  turns_per_session: { min: 1, max: 1 }
policy:
  model: test-model
  occupancy_threshold: 0.8
admission:
  primary_queue_margin: 1
"#,
        )
        .unwrap()
    }

    fn active_request(
        worker: WorkerRef,
        worker_key: WorkerWithDpRank,
        phase: Phase,
    ) -> ActiveRequest {
        ActiveRequest {
            worker,
            worker_key,
            phase,
            remaining_prefill_tokens: 16,
            output_tokens: 16,
            prompt_blocks: vec![],
            decode_rate: 0.0,
            session: 0,
            turn: 0,
            prompt: vec![],
        }
    }

    /// r11-1: the policy's decode cost must include the arriving request's own uncached
    /// blocks, exactly as the real prompt registry projects them.
    #[test]
    fn build_request_feeds_the_arrivals_uncached_blocks_to_the_policy() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let engine = Engine::new(&scenario, &mut selector);
        let prompt = crate::hash::synth_tokens("prompt", 64);
        let blocks = crate::hash::block_hashes(&prompt, scenario.block_size as usize);
        let request = engine.build_request(prompt.len(), &blocks, &HashSet::new());
        let primary = WorkerWithDpRank::new(0, 0);
        let load = request.worker_loads.get(&primary).expect("primary load");
        assert_eq!(load.additional_active_blocks, blocks.len());
    }

    /// Finding 1: the reported occupancy must be the one the policy compares against the
    /// threshold at decision time — the worker's active decode blocks plus the arriving
    /// request's own uncached blocks over the advertised KV capacity. A cold 7-block prompt on
    /// an idle 8-block worker is 0.875, over a 0.8 threshold; the old metric (current decode
    /// blocks only) read 0.0.
    #[test]
    fn decision_occupancy_includes_the_arrivals_uncached_blocks() {
        let scenario = Scenario::parse(
            r#"
name: occupancy_unit
seed: 1
duration_seconds: 60
block_size: 16
arrival_rate:
  - { time: 0, rate: 0.1 }
primary:
  - id: 0
    capacity_blocks: 8
    prefill_tokens_per_second: 100000
    decode_tokens_per_second: 40
    max_concurrent_requests: 8
proxies: []
workload:
  system_prompt_tokens: 64
  user_tokens: { min: 48, max: 48 }
  output_tokens: { min: 64, max: 64 }
  think_time_seconds: { min: 0.1, max: 0.1 }
  turns_per_session: { min: 1, max: 1 }
policy:
  model: test-model
  occupancy_threshold: 0.8
"#,
        )
        .unwrap();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let engine = Engine::new(&scenario, &mut selector);
        let prompt = crate::hash::synth_tokens("cold", 112);
        let blocks = block_hashes(&prompt, scenario.block_size as usize);
        assert_eq!(blocks.len(), 7);
        let occupancy = engine.primary_occupancy_for_blocks(0, &blocks);
        assert!(
            (occupancy - 0.875).abs() < 1e-9,
            "expected 7/8, got {occupancy}"
        );
        assert!(occupancy > scenario.policy.occupancy_threshold);
        // The old metric saw no decode load at all and reported the worker as idle.
        assert!(engine.primary_occupancy() < occupancy);
    }

    /// r11-7: an offline primary worker must not contribute decode occupancy or admission
    /// exclusions, even while it still holds queued or in-flight requests.
    #[test]
    fn offline_primary_worker_does_not_contribute_load_or_steering() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let mut engine = Engine::new(&scenario, &mut selector);
        engine.primary[0].online = false;
        engine.workers.remove(&0);
        let primary = WorkerWithDpRank::new(0, 0);
        engine.active.insert(
            1,
            active_request(WorkerRef::Primary(0), primary, Phase::Decoding),
        );
        engine.active.insert(
            2,
            active_request(WorkerRef::Primary(0), primary, Phase::Queued),
        );
        assert_eq!(engine.primary_occupancy_for(0), 0.0);
        assert!(engine.steering_exclusions().is_empty());
    }

    /// r11-8: proxy prompt blocks are published when prefill completes, not at admission, so
    /// a concurrent request cannot score a hit on blocks the proxy has not computed yet.
    #[test]
    fn proxy_cache_is_published_when_prefill_completes_not_at_admission() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let mut engine = Engine::new(&scenario, &mut selector);
        let prompt = crate::hash::synth_tokens("prompt", 64);
        let blocks = crate::hash::block_hashes(&prompt, scenario.block_size as usize);
        let proxy = WorkerWithDpRank::new(1000, 1000);
        let context = DecisionContext {
            arrival_time: 0.0,
            session: 0,
            turn: 0,
            prompt: prompt.clone(),
            prompt_blocks: blocks.clone(),
            output_tokens: 16,
            cache_hit_blocks: 0,
            occupancy: 0.0,
            is_followup: false,
            previous_under_threshold: false,
            attempts: 0,
            steering_excluded: 0,
        };
        engine.start_proxy(0, proxy, context);
        assert_eq!(engine.proxies[0].cache.overlap(&blocks, engine.time), 0);
        let id = engine.proxies[0].active[0];
        engine.on_prefill_done(id);
        assert_eq!(
            engine.proxies[0].cache.overlap(&blocks, engine.time),
            blocks.len()
        );
    }

    fn decoding_request(
        prompt: &[u32],
        block_size: u32,
        worker: WorkerWithDpRank,
    ) -> ActiveRequest {
        let mut request = active_request(WorkerRef::Primary(0), worker, Phase::Decoding);
        request.prompt = prompt.to_vec();
        request.prompt_blocks = block_hashes(prompt, block_size as usize);
        request
    }

    /// S13-1: `active_decode_blocks` is the per-worker union of complete prompt blocks, not a
    /// per-request sum, so a shared prefix is charged once no matter how many live requests
    /// carry it.
    #[test]
    fn active_decode_blocks_is_the_shared_prefix_union() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let mut engine = Engine::new(&scenario, &mut selector);
        let primary = WorkerWithDpRank::new(0, 0);
        // 80 tokens over a 16-token block is 5 complete blocks.
        let prompt = crate::hash::synth_tokens("shared", 80);
        let blocks = block_hashes(&prompt, scenario.block_size as usize);
        assert_eq!(blocks.len(), 5);
        for id in 1..=3 {
            engine
                .active
                .insert(id, decoding_request(&prompt, scenario.block_size, primary));
        }
        assert_eq!(engine.load_signals(primary).active_decode_blocks, 5);
    }

    /// S13-1: output-block tracking is off, and only complete prompt blocks count, so the
    /// decode footprint is `floor(prompt / block_size)` and does not grow with output tokens.
    #[test]
    fn active_decode_blocks_ignores_output_and_partial_blocks() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let mut engine = Engine::new(&scenario, &mut selector);
        let primary = WorkerWithDpRank::new(0, 0);
        // 40 tokens over a 16-token block is 2 complete blocks, partial tail dropped.
        let prompt = crate::hash::synth_tokens("partial", 40);
        let mut request = decoding_request(&prompt, scenario.block_size, primary);
        request.output_tokens = 10_000;
        engine.active.insert(1, request);
        assert_eq!(engine.load_signals(primary).active_decode_blocks, 2);
    }

    /// S13-3: the router acquires a request's prompt blocks at admission
    /// (`add_request_with_prefill_tracking` calls `acquire_prompt`), so a prefilling request
    /// already contributes to the decode footprint the failover threshold reads.
    #[test]
    fn prefilling_prompt_blocks_count_at_admission() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let mut engine = Engine::new(&scenario, &mut selector);
        let primary = WorkerWithDpRank::new(0, 0);
        let prompt = crate::hash::synth_tokens("prefill", 48);
        let mut request = active_request(WorkerRef::Primary(0), primary, Phase::Prefilling);
        request.prompt = prompt.clone();
        request.prompt_blocks = block_hashes(&prompt, scenario.block_size as usize);
        engine.active.insert(1, request);
        assert_eq!(engine.load_signals(primary).active_decode_blocks, 3);
    }

    /// S13-2: `additional_active_blocks` is the arriving request's blocks not already shared
    /// with a live sequence, taken from active membership, not the device KV cache. A prefix
    /// left in the cache by a completed request with no live sharer still counts in full.
    #[test]
    fn additional_active_blocks_uses_active_membership_not_the_device_cache() {
        let scenario = scenario();
        let mut selector = HeuristicSelector::new(scenario.policy.model_parameters());
        let mut engine = Engine::new(&scenario, &mut selector);
        let primary = WorkerWithDpRank::new(0, 0);
        let prompt = crate::hash::synth_tokens("resident", 48);
        let blocks = block_hashes(&prompt, scenario.block_size as usize);
        assert_eq!(blocks.len(), 3);
        // A completed request left every block in the device cache, but no live request holds
        // them: the router's membership overlap is zero, so the full prefix is additional.
        engine.primary[0].cache.insert_all(&blocks);
        let request = engine.build_request(prompt.len(), &blocks, &HashSet::new());
        assert_eq!(
            request
                .worker_loads
                .get(&primary)
                .unwrap()
                .additional_active_blocks,
            blocks.len()
        );
        // A live request now shares the first two blocks; only the third is additional.
        let live_prompt = crate::hash::synth_tokens("resident", 32);
        engine.active.insert(
            1,
            decoding_request(&live_prompt, scenario.block_size, primary),
        );
        let request = engine.build_request(prompt.len(), &blocks, &HashSet::new());
        assert_eq!(
            request
                .worker_loads
                .get(&primary)
                .unwrap()
                .additional_active_blocks,
            1
        );
    }
}
