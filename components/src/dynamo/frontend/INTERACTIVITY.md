<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend interactivity pools

An elected Rust frontend assigns discovered workers to two pools. Admission uses **worker-measured decode tokens per second per user**, with a configured minimum for each pool. All frontends read the same persisted assignments and filter the existing KV/load-aware router's candidates. Moving a worker between pools requires a drain acknowledgement from every registered frontend.

This is reactive admission control. A request admitted above its target can slow down afterwards. Idle probes and the bounded home-reclaim allowance intentionally permit exceptions. Targets are admission thresholds, not latency guarantees or GPU reservations.

## Configuration and rollout

Enable `--enable-decode-metrics` on SGLang and vLLM workers. The flag defaults to disabled. vLLM uses Dynamo's `InstrumentedScheduler`; an incompatible custom scheduler is rejected. SGLang requires the [companion engine patch](https://github.com/doublewordai/sglang/pull/6) (`upstream-pr/decode-metrics`) on the SGLang fork and enables its native `--enable-forward-pass-decode-metrics` instrumentation. Dynamo rejects an unpatched SGLang engine when the flag is enabled. The wrapper also enables the forward-pass metrics relay; setting an FPM port or enabling tracing alone does not enable decode measurement.

Deploy capable workers before enabling pooling consistently across frontends. Older workers continue to support ordinary routing, but missing decode snapshots exclude them from speed-based admission. Replace the PR's former `kv_fraction` configuration before enabling this version: it is rejected as an unknown field. The coordination schema is a pre-merge change with no migration from an assumed production deployment.

Set `DYN_FRONTEND_INTERACTIVITY_CONFIG` to a JSON file on each frontend:

```json
{
  "endpoint": "serving.worker.generate",
  "pools": {
    "interactive": {"min_decode_tps_per_user": 50, "minimum": 1},
    "throughput": {"min_decode_tps_per_user": 10, "minimum": 1}
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

**50 and 10 are examples only.** Every deployment must supply finite, positive `min_decode_tps_per_user` values; there are no default targets. Pool names are arbitrary. Two pools and positive minimum memberships are required. An array of configurations enables independent fleets for distinct generate endpoints; model aliases sharing an endpoint share a fleet.

Enable `DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true` and `DYN_ROUTER_TRACK_OUTPUT_BLOCKS=true`. Both remain required for scheduler block accounting and drain barriers. Blocks do not determine speed eligibility. Omit `DYN_FRONTEND_INTERACTIVITY_CONFIG` for ordinary routing.

Coordination requires etcd discovery. Every frontend routing to a pooled endpoint must use the same configuration and register before admitting pooled traffic. An ordinary router can bypass the barrier, so all traffic must pass through participating frontends. Initial scope is aggregated, single-sequence text generation with router queueing disabled.

## Worker measurements

Each global DP rank reports a rolling one-second measurement:

```text
decode tok/s/user = accepted decode output tokens / active decode-sequence seconds
```

A user is an active generation sequence, not an account or HTTP connection. Each sequence starts contributing decode time after its first committed output batch. That initial batch is excluded because no decode time has yet elapsed, including any initial speculative tokens. Subsequent accepted speculative output counts in full. Prompt tokens and first-token latency are excluded. Time spent waiting for subsequent tokens includes prefill interference, preemption, and decode stalls. Sequence counts are integrated over scheduler observation intervals; completed and cancelled sequences stop contributing when the scheduler observes their removal.

An idle rank or a rank without elapsed decode time reports an absent rate, never infinity. A sampled decode sequence that makes no progress reports zero. Reports include running/waiting request counts, a monotonically increasing observation revision, and a Unix timestamp in milliseconds. Observations are produced at most once every 250 ms from scheduler execution, result processing, or scheduler idle checks. vLLM's worker periodically wakes an idle engine with a lightweight utility RPC so the scheduler itself can confirm idleness; the RPC publisher does not infer idle state.

The optional FPM `decode_metrics` snapshot is relayed as endpoint-scoped `ActiveLoad.decode_metrics`. `WorkerMetricsPublisher.publish_decode_metrics` also accepts these snapshots independently of ordinary `publish` KV reports. Decode and KV revisions are independent. Old payloads and FPM heartbeats without a snapshot remain valid. Replaying a snapshot retains its original revision and timestamp, so a publisher heartbeat cannot refresh stale speed or make a stuck scheduler appear idle.

Freshness includes source observation age and transport delay, bounded by `telemetry_ttl_seconds`. Frontend and worker clocks must be synchronized; observations over one second in the future are rejected. Every expected DP rank needs fresh telemetry. A healthy, confirmed idle worker needs no defined decode rate to acknowledge a drain.

## Membership and admission

The frontend can start before workers register. The leader assigns arrivals to the smaller pool by worker count; ties use `default_pool`. Stable routing IDs are optional. Departed workers are excluded while persisted assignments preserve their classification on rediscovery. A single-worker fleet still enforces speed admission and can serve either pool through borrowing, even though it cannot satisfy both membership minima.

Clients select a pool with `x-dynamo-interactivity-pool` or `nvext.interactivity_pool`. Untagged requests use `default_pool`; invalid or conflicting tags return HTTP 400. Requests try their home pool first, then eligible workers in the other pool.

The effective target is the highest target among the worker's home pool, the incoming request's pool, and all locally tracked requests borrowing that worker. A borrower retains its target until stream cleanup. Every active rank must have a fresh rate **strictly above** the effective target. A rate equal to the target is ineligible for ordinary admission. A busy rank with no rate is also ineligible.

Dynamo keeps worker-level filtering and its existing DP selection. It checks the selected rank again under the admission lock before dispatch. Allow-lists, pins, taints, normal backend capacity limits, KV scoring and cancellation handling remain in force. If no candidate can use ordinary admission, an idle probe, or the reclaim exception, the frontend returns HTTP 503 with reason `interactivity_capacity`.

### Idle probes

Fresh, explicitly idle ranks can receive a probe even without a decode rate. There is at most **one outstanding idle probe per worker per frontend**, shared across DP ranks. Dispatch consumes that observation: repeated idle heartbeats or repeated admission attempts cannot issue another probe.

A later scheduler observation can release the restriction by establishing measured decode capacity. Alternatively, after the frontend observes successful stream completion, a subsequent scheduler observation must confirm idleness. A selection that fails before dispatch releases its reservation; a dispatched request without confirmed completion leaves uncertainty and cannot reopen idle probing from idle telemetry alone.

### Borrowing, priority and home reclaim

A home dispatch receives `home_priority` (default 100); borrowing receives `borrowed_priority` (default 0). Home priority must be greater. These values override client-supplied engine priority symmetrically in both pools. Ordinary routing with pooling disabled retains its existing priority behavior.

When speed is at or below the effective target and the frontend owns a local borrower, it may admit **one additional home request per worker per frontend**. This requires fresh, measured telemetry on active ranks and non-draining membership. Missing, expired or unsampled telemetry never qualifies. The allowance remains occupied until stream cleanup. Additional reclaim and new borrowing are blocked while it is occupied; ordinary home admission can resume if measured speed recovers.

Configure SGLang with `--enable-priority-scheduling --schedule-policy fcfs --priority-scheduling-preemption-threshold 10` to allow an eligible home request to preempt lower-priority work. For vLLM use `--scheduling-policy priority`; Dynamo converts priority polarity for that backend. Engines decide whether and how to preempt. DP scheduling/preemption stays local to the selected rank.

Probe, borrower and reclaim state remain local to each frontend. A frontend cannot reclaim solely because another frontend has borrowers. Multiple frontends can each use their own bounded allowance. Engine priority and measured admission do not guarantee that existing requests remain above target.

## Automatic reclassification and coordination

A recipient pool experiences pressure when **all present workers are busy**, and each worker's slowest active rank is at or below `effective_target / threshold`. With the default threshold 0.8, an example 50 tok/s/user target begins pressure tracking at 62.5 tok/s/user. Pressure must persist for `sustained_seconds`; cooldown and minimum memberships must permit the move. Unknown or stale rank observations suspend rebalancing.

Only a freshly idle worker with no locally outstanding or uncertain requests can donate. After removal, its pool must meet minimum membership, and every remaining worker must either be idle or have measured speed strictly above its own headroom boundary. Membership repair after departures follows the same idle/minimum/headroom restrictions. Busy workers are never automatically moved to relieve speed pressure.

The leader persists a drain with a new epoch. Each frontend applies it under its admission lock, stops selecting the donor, and waits for previously selected requests to complete. Acknowledgement requires newer scheduler observations for every rank, confirmed idleness, zero tracked decode blocks, and no local uncertain dispatches. An idle rank can acknowledge with an absent rate. The leader changes the pool only after every registered frontend acknowledges. A frontend joining during a drain must participate before the move completes. Timeout cancels the move under a new epoch and retains the original pool.

Assignments and acknowledgements live at `/dynamo/interactivity/<endpoint>/assignments`. Frontends poll this small document at `sample_seconds`; admission performs no etcd operations. An etcd lease elects the leader at the adjacent `leader` key. Writes compare both leader ownership and the document's modification revision. A successor resumes persisted drains and cooldowns; sustained pressure sampling restarts after leadership changes.

Frontend registrations and acknowledgements survive lease expiry. A paused or disconnected frontend may still reach workers, so its missing acknowledgement blocks movement. Graceful departure stops admission and removes its registration only when no outstanding or uncertain dispatch remains. Failure or cancellation without an observed successful backend completion marks the worker incarnation's local accounting uncertain. It cannot acknowledge later moves. Timed-out moves retain the original pool. Clearing a crashed frontend's registration requires operational confirmation that it cannot dispatch again and that its backend work has finished.

## Diagnostics and validation

Enable `DYN_LOG=info,dynamo_llm::kv_router::push_router::interactivity=debug` for decisions and state. Optional `DYN_FRONTEND_INTERACTIVITY_STATUS_PATH` writes a diagnostic JSON snapshot, with an endpoint suffix for multi-endpoint configurations. Each worker shows:

- Home and drain-target pools, effective target retained by local requests, and slowest active-rank speed.
- Per-rank speed, running/waiting counts, revision, observation timestamp and age; worker `healthy` and `idle` state.
- Per-request-pool eligibility and exclusion reason, including missing/stale telemetry, missing samples, below-target speed, draining, or a pending probe/reclaim.
- `probe_pending`, `reclaim_pending`, `reclaim_available`, `local_requests` and `has_uncertain_requests`.
- Existing scheduler block occupancy and KV capacity diagnostics used for ordinary routing and drains.

Admission logs include the selected rank's measured speed, age, effective target including the incoming request, borrowed/home priority, probe and reclaim decisions. The snapshot also records frontend registration, document revision and capacity shortage. It is diagnostic state, not a new metrics API.

CPU policy and real-etcd coordination tests:

```bash
ETCD_ENDPOINTS=http://localhost:2379 cargo test -p dynamo-llm --no-default-features --features testing-etcd --lib interactivity
PYTHONPATH=components/src python -m pytest components/src/dynamo/common/tests/test_decode_metrics.py components/src/dynamo/common/tests/test_decode_metrics_adapters.py
```

The small-model smoke test uses `Qwen/Qwen3-0.6B`, compares measured speed with streamed output, verifies single-worker HTTP 503 rejection under an intentionally unreachable target, and verifies idle-probe recovery. Run it separately in each capable worker environment, then profile actual VRAM before adding a shared-GPU CI allocation:

```bash
python -m pytest tests/router/test_interactivity_decode_metrics.py -k vllm -xvv
python -m pytest tests/router/test_interactivity_decode_metrics.py -k sglang -xvv
```

These smoke checks establish basic telemetry/admission behavior, not performance or preemption guarantees under production workloads.
