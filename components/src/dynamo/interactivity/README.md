<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend interactivity pools

The Rust frontend assigns each discovered worker to one of two concurrency classes. It filters Dynamo's existing KV/load-aware router candidates, counts its own outstanding requests, and changes its own classifications when sustained load differs between pools. Workers do not enforce pool policy. There is no worker gate, reservation RPC, or separate controller.

## Configuration

Set `DYN_FRONTEND_INTERACTIVITY_CONFIG` on the frontend to a JSON file:

```json
{
  "endpoint": "pooldemo.worker.generate",
  "pools": {
    "interactive": {"cap": 2, "minimum": 1},
    "throughput": {"cap": 8, "minimum": 1}
  },
  "default_pool": "throughput",
  "threshold": 0.8,
  "sustained_seconds": 30,
  "cooldown_seconds": 60,
  "drain_seconds": 300,
  "sample_seconds": 0.25,
  "telemetry_ttl_seconds": 5
}
```

For multiple models, supply a JSON array of these configuration objects, each with a distinct generate endpoint and its own caps and timing settings. Each model fleet must register on its own endpoint (for example, `pooldemo.worker.generate` and `pooldemo.worker2.generate`). Independent pools require distinct endpoints; model aliases on a shared endpoint are not separate fleets. Duplicate endpoint entries are rejected. Managers, observations, outstanding requests, cooldowns, and classifications are independent across endpoints, even when stable IDs and class names match.

Workers are assigned automatically when discovered. No `assignments` field is required or accepted. The frontend can start before any workers register. With exactly one registered worker, class filtering, pool caps, and pool rebalancing are bypassed; ordinary Dynamo routing and engine limits apply, including for tagged requests. DP ranks do not count as separate workers.

With two or more workers, each new worker joins the smaller pool by worker count. Ties go to `default_pool`, giving an initial split as close to 50/50 as possible. Batch discovery is ordered by stable routing ID, falling back to runtime worker ID. Stable IDs are optional. Workers need fresh telemetry for every DP rank before becoming eligible for multi-worker pool routing.

Deleting a worker removes it from accounting. If the remaining classes differ by more than one worker, the frontend drains and reclassifies workers to restore an even split. Joins and removals cancel an existing drain and recompute the membership plan. Load-driven rebalancing remains enabled after the discovery balance is established, so sustained demand can subsequently change the split. A restart assigns the currently discovered fleet afresh.

A cap applies to the whole worker, summed across its DP ranks. Engine `max_num_seqs` remains a separate per-rank ceiling. The configuration is read on router creation.

Clients set `x-dynamo-interactivity-pool: interactive` or `nvext.interactivity_pool`. Untagged requests use the default. With multi-worker pool routing active, unknown or conflicting classes return HTTP 400. When no eligible destination fits, the frontend returns HTTP 503 with reason `interactivity_capacity` and a retry hint.

## Worker information and request lifetime

Discovery supplies stable identity, DP topology, `max_num_seqs`, and KV capacity. The existing endpoint-scoped worker-metrics event stream supplies per-rank `num_active_reqs` (running plus waiting), queue length, KV use, and observation revision. The publisher adds an optional field without breaking existing `ActiveLoad` readers. The SGLang adapter derives active requests from `request_active_slots + num_requests_waiting`; other backends must publish equivalent counts before opting in. No worker control endpoint is needed.

The frontend maintains a cached observation for every rank. Missing counts, missing ranks, duplicate identities, or observations older than the configured TTL exclude the worker. Heartbeat replays do not refresh the observation age. Configure actual scheduler reporting more frequently than the TTL.

For each rank, estimated occupancy is the larger of (a) reported active requests plus local dispatches since that observation and (b) all locally outstanding requests. New observations reconcile the dispatch delta. This reduces double counting, but races between observation creation, delivery, and dispatch are intentionally approximate.

The router tries home-class workers before borrowing other workers. A borrowed request uses the smaller of the home and requested class caps. While that frontend still owns a borrowed interactive request, it keeps the worker's effective cap low. Existing pinning, taints, KV scoring, and engine queue safety filters still apply. An atomic frontend-local check records the chosen request before ordinary exact worker/rank dispatch. Dropping the response guard releases that local record, including on dispatch failure or cancellation.

## Reclassification

A background task inside the frontend compares pool occupancy with total effective concurrency capacity. A recipient must remain strictly above the threshold while the donor is below. The currently discovered fleet must have fresh telemetry; minima, donor headroom, and cooldown must permit a move.

The frontend stops choosing one donor worker for new requests, waits until its estimated occupancy reaches zero, then changes its local class. All DP ranks move together. A drain timeout restores eligibility in the original class. Only one move happens at a time. Both pools above threshold produces a capacity-shortage state rather than moving workers back and forth.

Model weights, worker processes, engine batching settings, and Kubernetes replica counts remain unchanged.

## Best-effort contract

Each frontend owns independent state. Two frontends can admit against the same observed spare capacity or disagree about a worker's current class. Worker telemetry has no request-class ownership information, so another frontend cannot enforce this frontend's borrowed-class protection. A restart loses local accounting and classifications. A disconnected response can release local accounting before the engine has stopped. TTLs and fresh worker counts limit stale routing; they do not provide a hard distributed concurrency guarantee or an atomic drain boundary. These soft failures are accepted in this design. A worker gate can be added later if strict enforcement becomes necessary.

Initial scope is a discovered fleet, two pools, aggregated single-sequence text serving with router queueing disabled. This is request-concurrency policy, not a token/KV-memory guarantee. Existing KV/load-aware routing remains responsible for its normal engine-safety decisions.

## Observing and testing

Enable `DYN_LOG=info,dynamo_llm::kv_router::push_router::interactivity=debug` to see admissions, rejections, drain starts, classifications, and state changes. Optional `DYN_FRONTEND_INTERACTIVITY_STATUS_PATH` writes a JSON snapshot for diagnostics; it is not shared admission state. With multiple endpoint configurations, the filename gains an endpoint suffix, for example `/tmp/pool-state.json.pooldemo.worker2.generate`. Decision and state logs include the endpoint.

See [the CPU Kubernetes demo](demo/README.md) for a real frontend with three ordinary two-rank dummy workers and an executable HTTP replay.
