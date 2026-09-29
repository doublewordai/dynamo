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
use crate::hash::{block_hashes, blocks_for};
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

struct HostedWorker {
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
    Hosted(usize),
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
    seq_blocks: usize,
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
    HostedChange { online: bool },
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

/// Common facts about one decision, shared by the hosted and proxy start paths.
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
    /// Hosted workers excluded from this decision because their engine queue was at
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
    hosted: Vec<HostedWorker>,
    proxies: Vec<ProxyWorker>,
    workers: HashMap<u64, testkit::SimWorker>,
    hosted_capacity: HashMap<u64, f64>,
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
        let mut hosted = Vec::new();
        let mut workers = HashMap::new();
        let mut hosted_capacity = HashMap::new();
        for config in &scenario.hosted {
            hosted.push(HostedWorker {
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
                testkit::SimWorker::hosted(config.capacity_blocks as u64),
            );
            hosted_capacity.insert(config.id, config.capacity_blocks as f64);
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
            hosted,
            proxies,
            workers,
            hosted_capacity,
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
        for change in &scenario.hosted_online {
            engine.schedule(
                change.time,
                EventKind::HostedChange {
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
                EventKind::HostedChange { online } => self.toggle_hosted(online),
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
        if let Some(index) = self.hosted.iter().position(|h| h.id == worker.worker_id) {
            WorkerRef::Hosted(index)
        } else {
            let index = self
                .proxies
                .iter()
                .position(|p| p.worker_id == worker.worker_id)
                .expect("selected worker is neither hosted nor a proxy");
            WorkerRef::Proxy(index)
        }
    }

    fn hosted_occupancy(&self) -> f64 {
        let capacity: f64 = self
            .hosted
            .iter()
            .filter(|worker| worker.online)
            .map(|worker| worker.capacity_blocks as f64)
            .sum();
        let decode: f64 = self
            .active
            .values()
            .filter_map(|request| match request.worker {
                WorkerRef::Hosted(_) if request.phase == Phase::Decoding => {
                    Some(request.seq_blocks as f64)
                }
                _ => None,
            })
            .sum();
        if capacity > 0.0 {
            decode / capacity
        } else {
            0.0
        }
    }

    /// Decode occupancy of a single hosted worker. Proxies have no capacity threshold.
    fn hosted_occupancy_for(&self, index: usize) -> f64 {
        let capacity = self.hosted[index].capacity_blocks as f64;
        let decode: f64 = self
            .active
            .values()
            .filter_map(|request| match request.worker {
                WorkerRef::Hosted(worker)
                    if worker == index && request.phase == Phase::Decoding =>
                {
                    Some(request.seq_blocks as f64)
                }
                _ => None,
            })
            .sum();
        if capacity > 0.0 {
            decode / capacity
        } else {
            0.0
        }
    }

    /// Engine-waiting requests on a hosted worker: admitted but not yet running, matching
    /// the waiting count the real worker reports (`report_engine_waiting`).
    fn hosted_queue_depth(&self, index: usize) -> usize {
        self.active
            .values()
            .filter(|request| {
                request.worker == WorkerRef::Hosted(index) && request.phase == Phase::Queued
            })
            .count()
    }

    /// Hosted workers whose engine queue is at or above their admission margin. Only hosted
    /// workers are eligible: proxies never report engine waiting, so their estimate stays
    /// unenforced, exactly as in the fork.
    fn steering_exclusions(&self) -> HashSet<u64> {
        let Some(admission) = &self.scenario.admission else {
            return HashSet::new();
        };
        self.hosted
            .iter()
            .enumerate()
            .filter(|(index, worker)| {
                self.hosted_queue_depth(*index) as u64 >= admission.margin_for(worker.id)
            })
            .map(|(_, worker)| worker.id)
            .collect()
    }

    fn under_threshold(&self, worker: WorkerWithDpRank) -> bool {
        match self.classify(worker) {
            WorkerRef::Hosted(index) => {
                self.hosted_occupancy_for(index) < self.scenario.policy.occupancy_threshold
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
            match request.phase {
                Phase::Queued => {}
                Phase::Prefilling => {
                    signals.active_requests += 1;
                    signals.active_prefill_tokens += request.remaining_prefill_tokens;
                }
                Phase::Decoding => {
                    signals.active_requests += 1;
                    signals.active_decode_blocks += request.seq_blocks;
                }
            }
        }
        signals
    }

    fn overlap_for(&self, worker: WorkerWithDpRank, blocks: &[u64]) -> usize {
        match self.classify(worker) {
            WorkerRef::Hosted(index) => self.hosted[index].cache.overlap(blocks),
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
        let occupancy = self.hosted_occupancy();
        let is_followup = turn > 0;
        let previous_under_threshold = match previous {
            None => false,
            Some(worker) => self.under_threshold(worker),
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
                    hosted_capacity: &self.hosted_capacity,
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
                WorkerRef::Hosted(index) => {
                    let cache_hit_blocks = self.overlap_for(worker, &prompt_blocks);
                    self.start_hosted(
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
        // that class is still a sensible target: any proxy, or a hosted worker under threshold.
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
            hosted_occupancy_at_selection: context.occupancy,
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
            hosted_occupancy_at_selection: context.occupancy,
            failed: true,
            rate_limited_attempts: context.attempts,
            steering_excluded: context.steering_excluded,
            // A failure with steering exclusions means every hosted worker was at its
            // margin and no proxy could take the request: the 529 refusal the gate causes.
            admission_529: context.steering_excluded > 0,
        });
        self.decision_trace.push("none".to_string());
    }

    fn start_hosted(&mut self, index: usize, worker: WorkerWithDpRank, context: DecisionContext) {
        let block_size = self.block_size as usize;
        let cached_tokens = (context.cache_hit_blocks * block_size).min(context.prompt.len());
        self.record_decision("hosted", false, worker, &context, cached_tokens as u64);
        self.hosted[index].cache.insert_all(&context.prompt_blocks);
        let id = self.next_id;
        self.next_id += 1;
        let phase = if self.hosted[index].active.len() < self.hosted[index].max_concurrent {
            Phase::Prefilling
        } else {
            Phase::Queued
        };
        self.active.insert(
            id,
            ActiveRequest {
                worker: WorkerRef::Hosted(index),
                worker_key: worker,
                phase,
                remaining_prefill_tokens: context.prompt.len() - cached_tokens,
                output_tokens: context.output_tokens,
                seq_blocks: blocks_for(
                    context.prompt.len() + context.output_tokens,
                    self.block_size,
                ),
                decode_rate: 0.0,
                session: context.session,
                turn: context.turn,
                prompt: context.prompt,
            },
        );
        if phase == Phase::Prefilling {
            self.hosted[index].active.push(id);
            self.begin_prefill(index, id);
        } else {
            self.hosted[index].queue.push_back(id);
        }
    }

    fn start_proxy(&mut self, index: usize, worker: WorkerWithDpRank, context: DecisionContext) {
        let block_size = self.block_size as usize;
        let cached_tokens = (context.cache_hit_blocks * block_size).min(context.prompt.len());
        let tier = self.proxies[index].tier.clone();
        self.record_decision(&tier, true, worker, &context, cached_tokens as u64);
        self.proxies[index]
            .cache
            .insert_all(&context.prompt_blocks, self.time);
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
                seq_blocks: blocks_for(
                    context.prompt.len() + context.output_tokens,
                    self.block_size,
                ),
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
        let batch = self.hosted[index].active.len();
        let rate = effective_rate(
            self.hosted[index].prefill_rate,
            batch,
            self.hosted[index].slowdown,
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
        let batch = self.hosted[index].active.len();
        let rate = effective_rate(
            self.hosted[index].decode_rate,
            batch,
            self.hosted[index].slowdown,
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
        match request.worker {
            WorkerRef::Hosted(index) => {
                self.active.get_mut(&id).unwrap().remaining_prefill_tokens = 0;
                self.begin_decode(index, id);
            }
            WorkerRef::Proxy(_) => {
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
            WorkerRef::Hosted(index) => {
                self.hosted[index]
                    .active
                    .retain(|active_id| *active_id != id);
                self.hosted[index].cache.insert_all(&full_blocks);
                self.admit_queued(index);
            }
            WorkerRef::Proxy(index) => {
                self.proxies[index]
                    .active
                    .retain(|active_id| *active_id != id);
                // The real proxy publishes prompt blocks only; its own generated output never
                // joins the virtual cache.
                let prompt_blocks = block_hashes(&request.prompt, self.block_size as usize);
                self.proxies[index]
                    .cache
                    .insert_all(&prompt_blocks, self.time);
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
        let occupancy = self.hosted_occupancy();
        self.occupancy_samples.push((self.time, occupancy));
    }

    fn admit_queued(&mut self, index: usize) {
        while self.hosted[index].active.len() < self.hosted[index].max_concurrent {
            let Some(id) = self.hosted[index].queue.pop_front() else {
                break;
            };
            self.active.get_mut(&id).unwrap().phase = Phase::Prefilling;
            self.hosted[index].active.push(id);
            self.begin_prefill(index, id);
        }
    }

    fn toggle_hosted(&mut self, online: bool) {
        for index in 0..self.hosted.len() {
            self.hosted[index].online = online;
            let id = self.hosted[index].id;
            if online {
                self.workers.insert(
                    id,
                    testkit::SimWorker::hosted(self.hosted[index].capacity_blocks as u64),
                );
            } else {
                self.workers.remove(&id);
            }
        }
        let occupancy = self.hosted_occupancy();
        self.occupancy_samples.push((self.time, occupancy));
    }
}

/// Hosted workers apply a fractional slowdown as concurrency grows.
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
