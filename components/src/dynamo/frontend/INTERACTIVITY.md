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

Clients select a class using `x-dynamo-interactivity-pool` or `nvext.interactivity_pool`. Untagged requests use `default_pool`. Invalid/conflicting tags return HTTP 400. Requests prefer home workers, then borrow other workers under the smaller class budget. A locally tracked borrowed stream keeps that smaller budget in effect until completion. Workers at or above the applicable budget are excluded. If no eligible worker/rank remains, the response is HTTP 503 with reason `interactivity_capacity`. Existing pinning, taints, KV scoring, engine limits, and cancellation handling remain in force.

The scheduler already tracks prompt blocks, shared-prefix ownership and output growth. The pool policy reads an empty load projection directly from that scheduler; it introduces no worker request-count metric, protocol extension or Python publisher change. Cached worker load reports establish freshness using the existing observation revision. Heartbeat replays do not renew an observation. Missing rank observations, expired telemetry or unknown KV capacity exclude the worker and suspend rebalancing.

## Reclassification

Pool pressure is `sum(active_decode_blocks) / sum(effective_block_budgets)`. A pool must remain above `threshold` while the donor is below it. Minimum memberships, donor headroom and cooldown must permit a move. If both pools exceed the threshold, the frontend reports a capacity shortage instead of moving workers.

The frontend excludes one donor from new routing, waits for zero tracked decode blocks and no locally held pool streams, then changes its class. Timeout cancels the drain. Model weights, worker processes, engine batching configuration and Kubernetes replica counts stay unchanged.

## Limits and observability

This is a soft KV-load policy, not a concurrency guarantee or a GPU memory reservation. Admission tests current tracked load; a large prompt or subsequent output growth can cross the budget. Shared prefixes make block load differ from the sum of request lengths. Fractional decay and replica synchronization follow the scheduler's existing behavior. Frontends can disagree temporarily, and cancellation can release local tracking before the engine stops. `kv_used_blocks` remains an independent backend overload signal; it is not substituted for active decode blocks in this policy.

Initial scope is aggregated, single-sequence text generation with router queueing disabled. GPU performance and hard consistency across replicated frontends require separate validation.

Enable `DYN_LOG=info,dynamo_llm::kv_router::push_router::interactivity=debug` for decisions and state. Optional `DYN_FRONTEND_INTERACTIVITY_STATUS_PATH` writes a diagnostic JSON snapshot, with an endpoint suffix for multi-endpoint configurations. `occupied` and `effective_cap` in this snapshot are block counts; `kv_fraction` is active blocks divided by total KV capacity. `local_requests` only tracks stream ownership for borrowing/draining and does not set capacity.
