<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend interactivity pools

The Rust frontend classifies discovered workers into two pools with different KV-block budgets. It filters the existing KV/load-aware router's candidates and reclassifies warm workers as demand shifts. Workers and their metrics interfaces are unchanged.

## Configuration

Set `DYN_FRONTEND_INTERACTIVITY_CONFIG` to a JSON file on the frontend:

```json
{
  "endpoint": "serving.worker.generate",
  "pools": {
    "interactive": {"kv_fraction": 0.2, "minimum": 1},
    "throughput": {"kv_fraction": 0.8, "minimum": 1}
  },
  "default_pool": "throughput",
  "home_priority": 100,
  "borrowed_priority": 0,
  "threshold": 0.8,
  "sustained_seconds": 30,
  "cooldown_seconds": 60,
  "drain_seconds": 300,
  "sample_seconds": 0.25,
  "telemetry_ttl_seconds": 5
}
```

Enable `DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true` and `DYN_ROUTER_TRACK_OUTPUT_BLOCKS=true`; configuration is rejected unless both are enabled. Omit `DYN_FRONTEND_INTERACTIVITY_CONFIG` to deploy ordinary routing. An array of configurations enables independent pools for distinct generate endpoints. Model aliases sharing an endpoint share a fleet.

`kv_fraction` is a fraction in (0, 1], not a request count. Each worker's block budget is `floor(kv_fraction * total_kv_blocks_per_rank * number_of_DP_ranks)`. Its load is the sum of the scheduler's existing `active_decode_blocks` across those ranks. For example, a worker with two ranks of 1,000 blocks each and interactive fraction 0.2 has a 400-block budget. Sustained load above 320 blocks exceeds the example 80% rebalance threshold. Both ranks always share one class.

## Membership and routing

The frontend can start before workers register. A single registered worker bypasses pooling, including when it has multiple DP ranks. With more workers, each arrival joins the smaller class by worker count; ties use `default_pool`. This starts the fleet as close to 50/50 as possible. Stable routing IDs are optional. Departures remove membership; an imbalanced remaining fleet is repaired through draining and reclassification.

Clients select a class using `x-dynamo-interactivity-pool` or `nvext.interactivity_pool`. Untagged requests use `default_pool`. Invalid/conflicting tags return HTTP 400. Requests try eligible workers in their home pool first, then borrow spare capacity from the other pool. Borrowing uses the smaller of the worker's class budget and the request's class budget. A locally tracked borrowed stream retains that lower budget until it ends. Full workers are excluded except for the bounded home admission described below. If no eligible worker/rank remains, the response is HTTP 503 with reason `interactivity_capacity`. Pool eligibility is intersected with Dynamo’s existing worker allow-list. Dynamo’s existing scheduler selects the DP rank; the pool policy adds no per-rank eligibility filter. Existing pinning, taints, KV scoring, engine limits, and cancellation handling remain in force.

The scheduler already tracks prompt blocks, shared-prefix ownership and output growth. The pool policy reads an empty load projection directly from that scheduler; it introduces no worker request-count metric, protocol extension or Python publisher change. Cached worker load reports establish freshness using the existing observation revision. Heartbeat replays do not renew an observation. Missing rank observations, expired telemetry or unknown KV capacity exclude the worker and suspend rebalancing.

## Borrowing and engine priority

Every pooled dispatch receives `home_priority` (default 100) when the request class matches the selected worker's class, or `borrowed_priority` (default 0) otherwise. `home_priority` must be greater than `borrowed_priority`. These values override client-supplied engine priority: a throughput request borrowing an interactive worker cannot promote itself above that worker's interactive requests. The rule applies symmetrically to both pools and does not depend on their names. Pooling disabled or single-worker bypass leaves ordinary request priorities unchanged.

For example, a throughput request borrows an interactive worker at priority 0. An interactive request arriving at that worker receives priority 100, allowing an enabled engine priority scheduler to favor it over the borrower. Priority is assigned at dispatch even before contention develops; it does not require changing the priority of a running request.

If borrowers fill the block budget, the frontend allows **one additional home request per worker per frontend** to reach the engine. This requires a locally tracked borrowed stream, fresh telemetry, and a worker that is not draining. The allowance is shared across DP ranks and remains occupied until that home stream ends, including cancellation or dispatch failure cleanup. Further over-budget requests are rejected. New borrowing is blocked while this allowance is occupied; normal home admission resumes whenever tracked load falls below budget. Existing engine/routing constraints still apply.

This exception permits an engine to preempt; it does not force it. Configure SGLang workers with `--enable-priority-scheduling --schedule-policy fcfs --priority-scheduling-preemption-threshold 10`, using a version supporting those settings. The default priority gap of 100 exceeds that threshold. SGLang decides whether it can retract lower-priority running work to schedule the new request. With DP attention this decision is local to the engine rank selected by Dynamo's existing scheduler; there is no cross-rank preemption guarantee. See [SGLang's scheduling implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py).

For vLLM, `--scheduling-policy priority` enables engine priority scheduling; Dynamo already converts priority polarity for that backend. Engine scheduling/preemption behavior depends on the deployed backend version and configuration. Without engine priority scheduling, the tags do not protect home requests from borrowers. See [Dynamo priority scheduling](https://docs.nvidia.com/dynamo/dev/agents/priority-scheduling).

Borrower ownership and the extra admission allowance are local to each frontend, not globally coordinated. A frontend cannot reclaim based solely on another frontend's borrowed streams. Preempted streams may remain in the frontend's block accounting until completion, so additional arrivals can still receive 503 while the engine has temporarily freed memory. CPU policy tests do not establish GPU preemption or latency guarantees.

## Reclassification

Pool pressure is `sum(active_decode_blocks) / sum(effective_block_budgets)`. A pool must remain above `threshold` while the donor is below it. Minimum memberships, donor headroom and cooldown must permit a move. If both pools exceed the threshold, the frontend reports a capacity shortage instead of moving workers.

The frontend excludes one donor from new routing, waits for zero tracked decode blocks and no locally held pool streams, then changes its class. Timeout cancels the drain. Model weights, worker processes, engine batching configuration and Kubernetes replica counts stay unchanged.

## Limits and observability

This is a soft KV-load policy, not a concurrency guarantee or a GPU memory reservation. Admission tests current tracked load; a large prompt or subsequent output growth can cross the budget. Shared prefixes make block load differ from the sum of request lengths. Fractional decay and replica synchronization follow the scheduler's existing behavior. Frontends can disagree temporarily, and cancellation can release local tracking before the engine stops. `kv_used_blocks` remains an independent backend overload signal; it is not substituted for active decode blocks in this policy.

Initial scope is aggregated, single-sequence text generation with router queueing disabled. GPU performance and hard consistency across replicated frontends require separate validation.

Enable `DYN_LOG=info,dynamo_llm::kv_router::push_router::interactivity=debug` for decisions and state. Optional `DYN_FRONTEND_INTERACTIVITY_STATUS_PATH` writes a diagnostic JSON snapshot, with an endpoint suffix for multi-endpoint configurations. `occupied` and `effective_cap` in this snapshot are block counts; `kv_fraction` is active blocks divided by total KV capacity. `local_requests` tracks stream ownership for draining, borrowed budgets and the extra admission allowance. `reclaim_available` means a home request can use that allowance; it is not a count of free blocks. Admission logs include `borrowed`, `priority` and `reclaim`.
