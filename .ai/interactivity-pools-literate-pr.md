<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# feat(router): add frontend-owned interactivity pools and warm reclassification

Implementation walkthrough · generated 2026-09-08.

## Summary

The Rust frontend now classifies warm workers into two concurrency pools, routes using worker telemetry and its own outstanding-request records, and changes those classifications as demand shifts. **Pool policy lives entirely in the frontend.** Workers report load and execute inference requests. A background task within the frontend manages pool membership.

A request prefers its home pool, may borrow another worker under the smaller class limit, and receives HTTP 503 when the frontend sees no eligible capacity. Each worker has one class and one budget across all DP ranks. Model weights, engine batching settings, and worker processes stay unchanged during a move.

Capacity decisions are best effort: each frontend combines cached worker telemetry with its own outstanding requests. Independent frontends can temporarily oversubscribe a worker or make different classification decisions.

[lib/llm/src/kv_router/push_router/interactivity.rs:4–4](../lib/llm/src/kv_router/push_router/interactivity.rs#L4-L4) — The new policy is part of the Rust push router.

```rust
//! Frontend-owned, best-effort interactivity policy. Workers only publish telemetry.
```

## 1. Configure the frontend’s view of the fleet

`DYN_FRONTEND_INTERACTIVITY_CONFIG` accepts one endpoint configuration or an array of configurations for separate model fleets. Each model fleet registers on its own generate endpoint; independent pools require distinct endpoints. Worker membership comes from runtime discovery; configuration contains no worker assignments. Caps are worker-wide: an interactive worker with two DP ranks and cap 2 can accept two total requests through this frontend, rather than two per rank. Runtime discovery still supplies the separate engine limit for each rank.

The configuration requires two pools, positive caps and minimum memberships, a valid default class, and finite positive timings. Each endpoint has independent membership, observations, outstanding requests, and cooldowns. Stable worker IDs and class names can be reused across endpoints. The frontend can start with no registered workers. Each arrival joins the smaller pool by worker count, with ties going to `default_pool`. This produces an initial split as close to 50/50 as possible. Stable IDs are optional; runtime worker IDs are the fallback. After a removal, an uneven fleet is repaired by draining and reclassifying a worker. Load-driven rebalancing remains active after membership has settled. Restarting a frontend assigns its discovered workers afresh.

[lib/llm/src/kv_router/push_router/interactivity.rs:29–44](../lib/llm/src/kv_router/push_router/interactivity.rs#L29-L44) — Configuration defines class policy independently of worker identities.

```rust
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
```

[lib/llm/src/kv_router/push_router/interactivity.rs:537–558](../lib/llm/src/kv_router/push_router/interactivity.rs#L537-L558) — Only the configured endpoint enables the policy; unconfigured routers retain their existing behavior.

```rust
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
        let period = Duration::from_secs_f64(config.sample_seconds);
        let manager = Arc::new(Self {
            state: Mutex::new(State::new(config)),
        });
        let weak = Arc::downgrade(&manager);
        let configs = chooser.runtime_configs();
        let cancel = endpoint.drt().primary_token();
        let status_path = std::env::var_os("DYN_FRONTEND_INTERACTIVITY_STATUS_PATH").map(|path| {
            if multiple {
                let mut path = path;
```

With exactly one registered worker, the frontend uses ordinary Dynamo routing: pool filters, class caps, and pool rebalancing are bypassed. The worker’s DP ranks still count as a single worker. Pool routing activates automatically when another worker registers; existing engine safeguards remain in effect throughout.

[lib/llm/src/kv_router/push_router/interactivity.rs:677–679](../lib/llm/src/kv_router/push_router/interactivity.rs#L677-L679) — The bypass counts registered workers, not DP ranks.

```rust
    pub(super) fn pools_bypassed(&self) -> bool {
        self.pools.is_some() && self.chooser.workers_with_configs.borrow().len() == 1
    }
```

[lib/llm/src/kv_router/push_router/interactivity.rs:282–296](../lib/llm/src/kv_router/push_router/interactivity.rs#L282-L296) — Each arriving worker joins the smaller class; ties use the default pool.

```rust
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
```

## 2. Discover topology and collect worker observations

Dynamo’s runtime-config watch supplies worker identity, DP topology, per-rank maximum sequences, and KV capacity. The frontend subscribes to the existing endpoint-scoped worker-metrics event stream. The only worker-side change is an optional active-request count in that telemetry payload. Existing readers can still deserialize the flattened `ActiveLoad` fields.

SGLang already reports running slots and waiting requests. Its adapter publishes their sum, alongside KV use and queue depth. Other backends need equivalent active counts before enabling this policy; a worker that never reports them remains ineligible.

[lib/llm/src/kv_router/publisher/worker_metrics.rs:20–25](../lib/llm/src/kv_router/publisher/worker_metrics.rs#L20-L25) — The observation envelope extends the existing worker metrics payload compatibly.

```rust
pub struct WorkerLoadReport {
    #[serde(flatten)]
    pub load: ActiveLoad,
    /// Running plus waiting requests on this rank, observed by the backend.
    #[serde(default)]
    pub num_active_reqs: Option<u64>,
```

[components/src/dynamo/sglang/publisher.py:228–243](../components/src/dynamo/sglang/publisher.py#L228-L243) — The SGLang adapter reports running plus waiting requests without changing engine execution.

```python
                num_waiting = getattr(kv_metrics, "num_requests_waiting", None)
                num_running = getattr(kv_metrics, "request_active_slots", None)
                self.metrics_publisher.publish(
                    dp_rank,
                    kv_used_blocks=active_decode_blocks,
                    num_active_reqs=int(num_running) + int(num_waiting)
                    if num_running is not None and num_waiting is not None
                    else None,
                    num_waiting_reqs=int(num_waiting)
                    if num_waiting is not None
                    else None,
                )
                dp_rank_str = str(dp_rank)
                # Publish total blocks (always available in KvMetrics)
                self.component_gauges.set_total_blocks(dp_rank_str, total_blocks)
                # Publish GPU cache usage percentage (always available in KvMetrics)
```

[lib/llm/src/kv_router/push_router/interactivity.rs:568–573](../lib/llm/src/kv_router/push_router/interactivity.rs#L568-L573) — The frontend consumes the existing metrics event plane.

```rust
                let subscription = tokio::select! {
                    _ = cancel.cancelled() => return,
                    result = EventSubscriber::for_endpoint(&endpoint, crate::kv_router::KV_METRICS_SUBJECT) => result,
                };
                let mut events = match subscription {
                    Ok(sub) => sub.typed::<WorkerLoadReport>(),
```

## 3. Expire observations and estimate occupancy

The frontend requires a fresh observation for every DP rank. A missing rank, expired observation, unknown active count, or duplicated stable identity excludes the whole worker. The TTL is measured from receipt of a new observation revision; transport heartbeats replaying an old revision cannot extend it.

For each rank, the frontend combines reported active work with dispatches made since that observation, then takes at least the number of its own outstanding requests. A newer observation reconciles the local dispatch delta. This avoids systematic double counting, but observation creation and delivery can race with dispatch. The result remains best effort.

[lib/llm/src/kv_router/push_router/interactivity.rs:178–197](../lib/llm/src/kv_router/push_router/interactivity.rs#L178-L197) — Reported occupancy plus a local delta, with local outstanding work as a lower bound.

```rust
    fn rank_occupied(&self, rank: u32) -> u64 {
        let Some(report) = self.reports.get(&rank) else {
            return 0;
        };
        let local = self.local.values().filter(|r| r.rank == rank);
        let pending = local
            .clone()
            .filter(|r| r.report_revision == report.revision)
            .count() as u64;
        // A new report reconciles the local dispatch delta. max avoids counting
        // this frontend's already-observed requests twice. Neither is a global lock.
        report
            .active
            .saturating_add(pending)
            .max(local.count() as u64)
    }

    fn view(&self, id: u64, config: &PoolConfig, now: Instant) -> View {
        let ranks = self.rank_start..self.rank_start.saturating_add(self.rank_count);
        let engine_cap = self.rank_cap.saturating_mul(self.rank_count as u64);
```

[lib/llm/src/kv_router/push_router/interactivity.rs:353–364](../lib/llm/src/kv_router/push_router/interactivity.rs#L353-L364) — An old revision is ignored; only a fresh observation updates the cached load.

```rust
        if member
            .reports
            .get(&load.dp_rank)
            .is_some_and(|old| old.revision >= revision)
        {
            return;
        }
        // Only new observations refresh freshness and reconcile dispatch deltas.
        // A transport heartbeat replay cannot make stale scheduler data fresh.
        member.reports.insert(
            load.dp_rank,
            RankReport {
```

## 4. A request arrives with an interactivity class

A client sets `x-dynamo-interactivity-pool: interactive` or `nvext.interactivity_pool`. The HTTP boundary normalizes and validates the tag; conflicting or malformed values return HTTP 400. The preprocessor carries the class in routing hints. Untagged requests use the configured default, while unknown class names return HTTP 400.

This first version supports aggregated, single-sequence text generation with router queueing disabled. The checks keep the counting unit unambiguous.

[lib/llm/src/protocols/common/preprocessor.rs:28–28](../lib/llm/src/protocols/common/preprocessor.rs#L28-L28) — The routing hint carries the requested interactivity class.

```rust
    pub interactivity_pool: Option<String>,
```

[lib/llm/src/kv_router/push_router/interactivity.rs:699–720](../lib/llm/src/kv_router/push_router/interactivity.rs#L699-L720) — Unsupported request shapes are rejected before pool selection.

```rust
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
            let pool = request
                .routing
                .as_ref()
                .and_then(|r| r.interactivity_pool.clone())
```

## 5. Filter candidates, then use Dynamo’s existing scorer

The frontend computes available capacity for each worker from its class cap, all its DP ranks, and outstanding borrowed requests. It tries home-class workers first, then compatible workers from the other class. Borrowing uses the smaller limit; a locally outstanding interactive request keeps a throughput worker’s effective cap low until that request’s response guard is released.

Eligible worker/rank pairs are passed to the existing router. KV affinity, load scoring, worker pins, and other existing constraints continue to select among those candidates. This change does not replace Dynamo’s engine queue safety or turn concurrency counts into a KV-memory guarantee.

[lib/llm/src/kv_router/push_router/interactivity.rs:204–233](../lib/llm/src/kv_router/push_router/interactivity.rs#L204-L233) — One effective cap covers the worker; every requested class receives an availability estimate.

```rust
        let effective_cap = self
            .local
            .values()
            .fold(config.pools[&self.pool].cap.min(engine_cap), |cap, r| {
                cap.min(config.pools[&r.pool].cap)
            });
        let occupied = ranks.clone().map(|rank| self.rank_occupied(rank)).sum();
        let available_by_pool = config
            .pools
            .iter()
            .map(|(name, p)| {
                (
                    name.clone(),
                    if healthy && self.target.is_none() {
                        effective_cap.min(p.cap).saturating_sub(occupied)
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
            reported_occupied: self.reports.values().map(|r| r.active).sum(),
            local_occupied: self.local.len(),
```

[lib/llm/src/kv_router/push_router/interactivity.rs:727–743](../lib/llm/src/kv_router/push_router/interactivity.rs#L727-L743) — Home candidates precede borrowed candidates, while DP ranks remain concrete destinations.

```rust
        for home in [true, false] {
            let mut eligible: HashSet<_> = views
                .iter()
                .filter(|v| (v.pool == pool) == home && v.available_by_pool[&pool] > 0)
                .flat_map(|v| {
                    v.eligible_ranks
                        .iter()
                        .map(|rank| WorkerWithDpRank::new(v.worker_id, *rank))
                })
                .collect();
            while !eligible.is_empty() {
                let selection = self
                    .select_worker(
                        request.context().id(),
                        request,
                        RoutingRequestParts::new(request),
                        phase,
```

## 6. Record the choice locally and dispatch normally

The frontend rechecks the selected worker under its own state mutex and adds a local outstanding-request record before dispatch. Concurrent requests handled by this frontend see that record immediately. If another local request has consumed the available estimate, selection tries another candidate.

The local `PoolLease` is an RAII accounting handle. It is attached to the existing response guard and ordinary exact worker/rank dispatch follows. Independent frontends do not share this mutex or record.

[lib/llm/src/kv_router/push_router/interactivity.rs:631–664](../lib/llm/src/kv_router/push_router/interactivity.rs#L631-L664) — The atomic check is frontend-local; it cannot serialize admissions from another frontend.

```rust
    fn admit(self: &Arc<Self>, worker: WorkerWithDpRank, pool: &str) -> Option<PoolLease> {
        let mut state = self.state.lock();
        let view = state
            .views(Instant::now())
            .into_iter()
            .find(|v| v.worker_id == worker.worker_id)?;
        if view.available_by_pool.get(pool).copied().unwrap_or(0) == 0
            || !view.eligible_ranks.contains(&worker.dp_rank)
        {
            return None;
        }
        state.next_request += 1;
        let request = state.next_request;
        let member = state.members.get_mut(&worker.worker_id)?;
        let revision = member.reports.get(&worker.dp_rank)?.revision;
        member.local.insert(
            request,
            LocalRequest {
                pool: pool.into(),
                rank: worker.dp_rank,
                report_revision: revision,
            },
        );
        tracing::debug!(endpoint=%state.config.endpoint, worker_id=worker.worker_id, dp_rank=worker.dp_rank, request_pool=pool, node_pool=%view.pool, borrowed=pool != view.pool, occupied=view.occupied+1, cap=view.effective_cap.min(state.config.pools[pool].cap), "Pool routing admitted by frontend");
        Some(PoolLease {
            manager: Arc::downgrade(self),
            worker: worker.worker_id,
            request,
        })
    }
}

fn pool_error(code: u16, message: &str) -> Error {
    DynamoError::builder()
```

[lib/llm/src/kv_router/push_router.rs:621–621](../lib/llm/src/kv_router/push_router.rs#L621-L621) — The existing request lifetime owns the local accounting record.

```rust
        guard.pool_lease = selection.pool_lease;
```

## 7. Execute, complete, cancel, or fail

A worker receives an ordinary inference request. The CPU demo worker below increments its active rank count, emits tokens, and decrements the count when execution finishes. Its periodic metrics report these counts to the frontend.

When the frontend response guard drops, the local record is released, including on dispatch failure or cancellation. Backend work may continue after a frontend disconnect; subsequent telemetry can still report it, but a gap is possible.

[components/src/dynamo/interactivity/demo/worker.py:65–102](../components/src/dynamo/interactivity/demo/worker.py#L65-L102) — The dummy worker executes requests and reports its active count.

```python
    async def generate(request, context):
        routing = request.get("routing") or {}
        rank = routing.get("dp_rank") or 0
        count = min(
            1200, max(1, (request.get("stop_conditions") or {}).get("max_tokens") or 1)
        )
        active[rank] += 1
        logger.info(
            "START node=%s rank=%s actual_occupied=%s tokens=%s",
            stable_id,
            rank,
            sum(active),
            count,
        )
        try:
            for index in range(count):
                await asyncio.sleep(0.1)
                yield {
                    "token_ids": [3],
                    "finish_reason": "stop" if index == count - 1 else None,
                    "meta_info": {
                        "finish_reason": "stop" if index == count - 1 else None
                    },
                }
        finally:
            active[rank] -= 1
            logger.info(
                "FINISH node=%s rank=%s actual_occupied=%s",
                stable_id,
                rank,
                sum(active),
            )

    config = ModelRuntimeConfig()
    config.stable_routing_id = stable_id
    config.max_num_seqs = 8
    config.total_kv_blocks = 4096
    config.context_length = 4096
```

[lib/llm/src/kv_router/push_router/interactivity.rs:526–536](../lib/llm/src/kv_router/push_router/interactivity.rs#L526-L536) — Release is tied to frontend lifetime, not proof of engine termination.

```rust
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
```

## 8. Reject when the frontend sees no eligible capacity

If home and borrowed candidates are exhausted, the frontend returns the typed capacity error. The HTTP layer translates it into a retryable 503 with reason `interactivity_capacity`. A full class budget, a drain in progress, stale telemetry, or incompatible worker/rank constraints can all leave no eligible destination.

The frontend makes this capacity decision before submitting the request to a worker.

[lib/llm/src/kv_router/push_router/interactivity.rs:798–799](../lib/llm/src/kv_router/push_router/interactivity.rs#L798-L799) — Capacity rejection is a frontend decision.

```rust
        tracing::debug!(endpoint=%manager.state.lock().config.endpoint, request_pool=%pool, "Pool routing rejected by frontend: no eligible capacity");
        Err(PoolCapacityRejection.into())
```

## 9. Rebalance inside the frontend

A background task in the same Rust manager compares each pool’s estimated occupied requests with its total effective capacity. A recipient must remain strictly above the threshold while the donor stays below. The currently discovered fleet must have fresh observations, and minimum memberships, donor headroom, and cooldown must allow a move.

Only one worker drains at a time. When both pools exceed the threshold, the frontend records a capacity shortage and retains the current allocation.

[lib/llm/src/kv_router/push_router/interactivity.rs:463–481](../lib/llm/src/kv_router/push_router/interactivity.rs#L463-L481) — Sustained asymmetric pressure drives the frontend’s allocator.

```rust
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
```

[lib/llm/src/kv_router/push_router/interactivity.rs:497–515](../lib/llm/src/kv_router/push_router/interactivity.rs#L497-L515) — The donor must retain enough capacity after losing the selected worker.

```rust
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
```

## 10. Drain the local view, then change the class

Setting a target class makes the donor unavailable for new requests from this frontend. Once its estimated occupancy reaches zero, the manager replaces its class. Every DP rank moves together because the class belongs to the worker. A drain timeout cancels the target and restores eligibility in the original class.

Engine configuration, resident weights, and worker processes remain unchanged. Another frontend may still send requests during this local drain, so zero observed occupancy is not an atomic cluster-wide handoff.

[lib/llm/src/kv_router/push_router/interactivity.rs:401–420](../lib/llm/src/kv_router/push_router/interactivity.rs#L401-L420) — Reclassification changes only frontend-owned membership.

```rust
        if let Some(draining) = views.iter().find(|v| v.target.is_some()) {
            self.pressure_since.clear();
            if draining.occupied == 0 {
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
```

## 11. What this design guarantees—and what it estimates

| Concern | Behavior |
|---|---|
| Same frontend, simultaneous requests | Local check and accounting update are serialized. |
| Multiple DP ranks | One worker class and cap; separate rank destinations and engine ceilings. |
| Missing or old telemetry | Worker excluded after the observation TTL. |
| Multiple frontends | Independent estimates and classifications; occasional oversubscription is accepted. |
| Borrowed-class protection | Applies to requests still tracked by this frontend; worker counts do not carry class ownership. |
| Cancellation or frontend restart | Local accounting may disappear before engine work ends; fresh telemetry helps recover. |
| Reclassification | A local routing change after estimated drain, with unchanged worker execution settings. |

Observation TTLs bound the age of cached load information. Together with local request accounting, they support responsive routing and rebalancing while allowing temporary oversubscription and classification differences between frontends.

## Validation

The CPU replay exercises the actual modified Rust frontend against three ordinary workers, each exposing two simulated DP ranks. It checks home routing, unknown-class errors, class-specific and drain rejection, worker-wide caps, reclassification in both directions, borrowed-class protection, and full-capacity rejection. Worker process identities must remain unchanged during moves.

The attached replay completed successfully on local MicroK8s on 8 September 2026: all six phases passed, including both reclassification directions, with unchanged worker incarnations. Thirteen focused Rust tests passed; `cargo clippy -p dynamo-llm --lib`, formatting, and Python lint checks passed. Rust unit tests cover local accounting across ranks, borrowing, observation reconciliation and expiry, and rebalancing; existing HTTP tag and typed-error tests cover the request boundary. CPU ranks validate routing and accounting behavior, not GPU DP-attention performance or a hard multi-frontend capacity guarantee.

The two-model replay uses one frontend, six workers, identical class names and stable IDs, and distinct caps (2/8 and 3/6). It verifies that reclassification and saturation of either model leave the other model’s membership and accounting unchanged, that the other model continues serving both classes, and that successful probes route within their own fleet. See [the two-model replay](two-model-replay.log).

The registration replay starts a frontend with no workers for one model, then tests singleton bypass above the class cap, automatic 1/1 and 2/1 joins, re-enabled capacity rejection, deletion repair, and rejoin while the second model remains unchanged. See [the registration replay](registration-replay.log).

Configuration and reproduction instructions are in `components/src/dynamo/interactivity/README.md` and its `demo/README.md`. This remains a working-tree PR draft; it has not been published to GitHub.
