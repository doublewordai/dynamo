<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tuning the spillover policy

This note explains what the policy's knobs do and gives two starting profiles: spill early
(latency first) and spill late (primary utilisation first). Everything here comes from `routing-sim sweep` on the `overload_ramp` scenario
(one primary worker, two X and two Y proxy workers, arrivals ramped to ~18x primary capacity and
back, so the single primary worker is pushed well past its failover point). Sweeps are deterministic and keep the scenario seed, so the numbers below reproduce:

```sh
cargo run -p dw-routing-sim -- sweep lib/spillover/routing-sim/scenarios/overload_ramp.yaml \
    --param failover_penalty_blocks=100,200,400,800,1600 \
    --param occupancy_threshold=0.8,0.9 \
    --jobs 8 --markdown out.md --json out.json
```

The tables below are excerpted columns of that `--markdown` output, not a verbatim paste. The
generator's full header is
`settings | requests | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | worker sticky % | cache hit % | primary cache hit % | steer excl | 529 | failures`
followed by one `<tier> %` column per tier. The main sweep table drops `primary cache hit %`,
`steer excl`, `529` and the `Y %` column; the admission table is a derived subset with its own
column names (`margin`, `primary share %`), so neither can be diffed byte-for-byte against a
fresh `--markdown` run.

Every cost is in KV blocks; lower is better. The baseline scorer (cache affinity plus current
load) runs first and the spillover scorer is stacked on top, so a setting only matters when it
is large enough to change the ordering that the baseline leaves behind.

## What each setting does

| Setting | Effect |
|---|---|
| `occupancy_threshold` | Fraction of a primary worker's capacity above which it earns `failover_penalty_blocks`. Occupancy is the larger of the worker's decode-block fraction (`decode blocks / total_kv_blocks`, fallback `primary_capacity_blocks`) and its projected concurrency fraction (`(active requests + the arriving request) / max_num_seqs`, fallback `primary_max_requests`), so either signal can trip it. A value in `(0, 4]`: `1.0` is exactly full, `0.8` spills before the last 20%, `1.2` accepts 20% queueing. The penalty applies strictly above the threshold, so at exactly `1.0` a full worker is still acceptable. Lower values spill earlier; higher values let primary fill further before any penalty applies. |
| `primary_capacity_blocks` | Optional fallback KV capacity in blocks for a primary worker that does not advertise `total_kv_blocks`. Leave unset to use each worker's advertised capacity. Must be positive and finite if set. |
| `primary_max_requests` | Optional fallback sequence capacity for a primary worker that does not advertise `max_num_seqs`. Leave unset to use each worker's advertised capacity. Must be positive and finite if set. |
| `failover_penalty_blocks` | Cost added to a primary worker once it is above the threshold. At or above `ceil(context_length / kv_block_size)` plus the costliest tier's `penalty_blocks + weight_blocks` (the generator computes it and warns below it), no cached prefix can outweigh it, so the threshold is a hard cap: a primary worker over it never wins while any tier is available. Below that the threshold is a soft cap and conversations with a long cached prefix stay on a full primary worker and queue there. Use the hard-cap value unless that queueing is wanted; to accept queueing, raise `occupancy_threshold` instead, which is exact. |
| `<tier>.penalty_blocks` | Fixed "always full" cost for every worker in a proxy tier. Raising it makes that tier less attractive; the spilling traffic shifts to other tiers or back to primary. |
| `<tier>.weight_blocks` | Tier preference between proxy tiers: smaller is preferred. Ordering X below Y keeps the cheaper/faster tier first. |
| `pending_weight_blocks` | Cost per active request on any worker. It is a load-spreading term; with a single primary worker it mostly moves traffic off a busy proxy or host. |

## End-to-end on curie

The threshold calculus was checked on curie (namespace `spillover-test`) with the production
frontend image and arguments (`--router-temperature 0`, overlap credit 1.0,
`--no-router-track-active-blocks` with tracking enabled on the worker set) and the images built
from this branch. Two primary workers are dw-proxy-workers in front of one inference-lab
simulation of Qwen3-30B-A3B on an H100 (`max_num_seqs` 16), each advertising half of it
(`advertised_capacity: {kv_blocks: 8800, max_requests: 8}`), and two OpenRouter tiers
(penalty 200, weights 8 and 40). The load is 1000 four-turn sessions at 3 sessions/s with 5 s think
time and 128 output tokens: about twice what primary can hold, so every profile spills. Each
run starts from restarted workers (empty caches).

A worker's cap is `floor(max_num_seqs * occupancy_threshold)` requests: 6, 8 and 9 here. The
loadgen records every request's interval and serving rank, which gives each primary worker's
exact concurrency when a request was admitted; the frontend's router-tracked
`active_requests` matched that client-side count to within the request being routed.

| failover penalty | threshold | primary share | spills | admissions over cap | peak primary concurrency | primary TTFT p50 / p95 | errors |
|---|---|---|---|---|---|---|---|
| 200 (soft) | 0.8 | 76.2% | 952 | 573 (19%) | 10 | 35 / 65 ms | 0 |
| 200 (soft) | 1.0 | 85.0% | 599 | 39 | 10 | 34 / 66 ms | 0 |
| 200 (soft) | 1.2 | 88.8% | 448 | 3 | 10 | 37 / 210 ms | 0 |
| 1000 (hard) | 0.8 | 67.1% | 1316 | 0 | 6 | 35 / 61 ms | 0 |
| 1000 (hard) | 1.0 | 83.1% | 677 | 0 | 8 | 35 / 61 ms | 0 |
| 1000 (hard) | 1.2 | 88.9% | 444 | 0 | 9 | 38 / 211 ms | 0 |

Readings:

- **With the hard-cap penalty the threshold is exact.** Peak concurrency per primary worker equals
  the cap at every threshold, and no request was admitted above it. The hard-cap floor for this
  deployment is 512 context blocks + 240 = 752.
- **Spills happen only when primary is full.** Every spill found both primary workers at their
  cap, except 3-5 per run (under 0.5% of spills) where a request was routed in the same
  millisecond as another and the client-side order differs from the router's.
- **The threshold trades primary share against queueing.** 0.8 keeps a 25% headroom and spills a
  third of the traffic; 1.2 holds 18 requests on a 16-slot engine, spills 11%, and the queueing
  shows as primary TTFT p95 rising from about 60 ms to 210 ms. 1.0 fills the engine exactly.
- **A soft penalty leaks most at low thresholds.** At penalty 200 a follow-up turn whose cached
  prefix saves more blocks than the penalty margin stays on a full primary worker: 19% of
  admissions at 0.8 were over the cap, reaching 10 concurrent on a 6 cap.
- **Latency.** Provider tiers add 300-600 ms to TTFT p50 over primary, so the spill share is the
  latency cost of each profile.

Two production caveats this surfaced, both fixed on this branch: the frontend's embedded router
dropped `max_num_seqs` on its way into the selection catalog (so the concurrency signal was
absent in production), and SGLang workers advertised no `max_num_seqs` unless
`--max-running-requests` was set.

## Main sweep: failover penalty x occupancy threshold

Peak columns are the scenario's `peak` phase (t = 70..95 s). `X %` is the share of all requests
served by tier X; Y is a strictly more expensive last resort and carried nothing in any of these
points (the `proxy_rate_limited` scenario exercises it).

<!-- BEGIN SWEEP TABLE -->
| settings | requests | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | worker sticky % | cache hit % | failures | X % |
|---|---|---|---|---|---|---|---|---|---|---|
| failover_penalty_blocks=100, occupancy_threshold=0.8 | 1140 | 54.5 | 66.1 | 117.7 | 136.8 | 75.4 | 63.4 | 47.6 | 0 | 54.5 |
| failover_penalty_blocks=100, occupancy_threshold=0.9 | 1140 | 54.5 | 66.1 | 117.7 | 136.8 | 75.4 | 64.1 | 47.6 | 0 | 54.5 |
| failover_penalty_blocks=200, occupancy_threshold=0.8 | 1140 | 55.5 | 67.9 | 111.9 | 127.1 | 83.9 | 61.5 | 48.4 | 0 | 55.5 |
| failover_penalty_blocks=200, occupancy_threshold=0.9 | 1140 | 55.5 | 67.9 | 111.9 | 127.1 | 83.9 | 62.2 | 48.4 | 0 | 55.5 |
| failover_penalty_blocks=400, occupancy_threshold=0.8 | 1140 | 56.0 | 72.6 | 110.1 | 126.7 | 79.6 | 64.7 | 48.2 | 0 | 56.0 |
| failover_penalty_blocks=400, occupancy_threshold=0.9 | 1140 | 56.0 | 72.6 | 110.1 | 126.7 | 79.6 | 65.8 | 48.2 | 0 | 56.0 |
| failover_penalty_blocks=800, occupancy_threshold=0.8 | 1140 | 58.1 | 74.3 | 92.5 | 104.2 | 84.3 | 63.7 | 47.4 | 0 | 58.1 |
| failover_penalty_blocks=800, occupancy_threshold=0.9 | 1140 | 56.6 | 75.0 | 93.2 | 106.9 | 80.4 | 59.9 | 46.2 | 0 | 56.6 |
| failover_penalty_blocks=1600, occupancy_threshold=0.8 | 1140 | 55.7 | 74.6 | 80.4 | 81.2 | 79.1 | 62.2 | 44.7 | 0 | 55.7 |
| failover_penalty_blocks=1600, occupancy_threshold=0.9 | 1140 | 55.2 | 74.6 | 89.7 | 90.6 | 77.9 | 63.5 | 44.4 | 0 | 55.2 |
<!-- END SWEEP TABLE -->

Readings:

- **The ramp overshoots one worker, so the policy spills at every grid point; the penalty sets
  how early and how deep.** Overall proxy share is 54-58% across the whole grid, because the
  plateau arrival rate is roughly twice what the single primary worker can decode and the
  baseline scorer spills the excess even with a small failover penalty.
- **A small penalty lets the baseline pile onto primary before the policy reacts.** At threshold
  0.8, penalty 100 leaves the peak primary occupancy at 117.7% mean / 136.8% max — over capacity,
  so the primary queue is growing while the policy is nominally in charge. Raising the penalty to
  800 brings the peak back to 92.5% mean / 104.2% max, and to 1600 to 80.4% mean / 81.2% max.
  `failover_penalty_blocks` is therefore the knob that actually holds primary at its failover
  point, not the threshold.
- **Peak proxy share rises with the penalty then flattens.** It climbs from 66% at penalty 100
  to ~74-75% at 800-1600 (a larger penalty spills earlier in the ramp and keeps spilling), so
  use peak *occupancy* to judge the penalty and overall proxy share to judge cost: at threshold
  0.8 the total drifts 54.5% -> 56.0% -> 55.7% as the penalty goes 100 -> 400 -> 1600.
- **The threshold only separates once the penalty is large enough to engage it.** At penalty
  100-400 the 0.8 and 0.9 rows are identical: the baseline load term, not the failover cost,
  decides, and the KV fraction is far past both thresholds anyway. At penalty 800-1600 the
  threshold separates cleanly — 0.9 lets primary fill to 89.7% mean / 90.6% max and spills
  slightly less overall (55.2% vs 55.7%), while 0.8 caps it at 80.4% mean / 81.2% max.
- **Stickiness peaks in the middle of the grid.** Peak class stickiness is 84.3% at threshold
  0.8 / penalty 800 and 83.9% at threshold 0.8 or 0.9 / penalty 200, then falls to ~78-79% at
  the top end: an aggressive failover penalty starts spilling conversations that the baseline
  would have kept on primary. Worker stickiness tracks it at 60-66%.
- **No failures and no Y traffic** at any point: X alone has enough capacity, and the failure
  path is exercised by the `proxy_rate_limited` scenario instead.

## Secondary sweep: proxy tier penalty

Raising the X tier's fixed penalty makes X less attractive, so less traffic spills; with Y kept
strictly more expensive (penalty 2000) X stays the only spill tier and the overall proxy share
falls monotonically (`occupancy_threshold` 0.8, `failover_penalty_blocks` 500):

| X.penalty_blocks | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | X % | Y % |
|---|---|---|---|---|---|---|---|
| 0 | 64.6 | 75.3 | 57.8 | 74.2 | 70.2 | 64.6 | 0.0 |
| 200 | 62.1 | 74.6 | 72.8 | 81.2 | 70.4 | 62.1 | 0.0 |
| 400 | 59.6 | 76.0 | 80.0 | 81.2 | 81.0 | 59.6 | 0.0 |
| 800 | 59.4 | 74.9 | 82.7 | 89.1 | 80.6 | 59.4 | 0.0 |
| 1600 | 54.3 | 66.1 | 117.7 | 136.8 | 75.4 | 54.1 | 0.2 |

This is the monotonicity the test suite checks: a larger tier penalty can only reduce the run's
overall proxy share. The peak-phase share instead wobbles within a point (75.3% -> 74.6% ->
76.0% -> 74.9% -> 66.1%) because the concurrency signal adds feedback — routing changes the
active-request counts, which shift when spill begins inside the peak phase — so the test allows
the peak share a two-point band and requires the run-level share to be strictly monotone. It
also shows the practical range: below ~200 X absorbs traffic that primary could serve (primary
peak occupancy only 58% mean), while at 1600 X is expensive enough that the baseline keeps more
traffic on primary and primary overshoots capacity (118% mean / 137% max) — the point past which
raising the tier penalty is counterproductive.

## Admission margin (backend gate, not a policy setting)

The engine-queue admission margin is **not** a router-policy field. The fork reads a single
`DYN_ADMISSION_QUEUE_MARGIN` from each **worker process**
(`lib/runtime/src/admission_gate.rs`, parsed in `lib/runtime/src/admission_margin.rs`); the
frontend never reads it and there is no per-model override map. While a primary worker's engine
waiting queue is at or above its margin that worker is excluded from selection, so cached
conversations are pushed to a proxy. A worker whose engine has never reported its waiting count
is unenforced, and `dw-proxy-worker` never reports one, so the margin cannot apply to a proxy.
When every primary worker is at its margin and no proxy can take the request, the router refuses
it; the frontend turns the overload error into HTTP 529 (`DYN_HTTP_OVERLOAD_STATUS_CODE`).

The simulation models this with an `admission:` block: `primary_queue_margin` applies to every
primary worker and `primary_queue_margin_overrides` gives individual worker ids their own value
(mirroring one margin per worker process). Stage one as `admission_queue_margin`:

```sh
cargo run -p dw-routing-sim -- sweep lib/spillover/routing-sim/scenarios/admission_margin_high.yaml \
    --param admission_queue_margin=0,1,2,3,4,6,8,10,16,32,1000 \
    --jobs 8 --markdown out.md --json out.json
```

<!-- BEGIN ADMISSION SWEEP TABLE -->
| margin | requests | proxy % | primary share % | primary cache hit % | steer excl | 529 | failures |
|---|---|---|---|---|---|---|---|
| 0 | 570 | 100.0 | 0.0 | 0.0 | 1140 | 0 | 0 |
| 1 | 569 | 55.9 | 44.1 | 39.8 | 812 | 0 | 0 |
| 2 | 566 | 54.8 | 45.2 | 41.1 | 787 | 0 | 0 |
| 3 | 567 | 54.7 | 45.3 | 40.7 | 793 | 0 | 0 |
| 4 | 559 | 54.2 | 45.8 | 41.6 | 771 | 0 | 0 |
| 6 | 555 | 52.6 | 47.4 | 39.6 | 746 | 0 | 0 |
| 8 | 546 | 51.1 | 48.9 | 43.9 | 719 | 0 | 0 |
| 10 | 541 | 50.3 | 49.7 | 42.5 | 686 | 0 | 0 |
| 16 | 528 | 46.6 | 53.4 | 45.2 | 635 | 0 | 0 |
| 32 | 493 | 36.3 | 63.7 | 52.1 | 455 | 0 | 0 |
| 1000 | 431 | 23.2 | 76.8 | 65.4 | 0 | 0 | 0 |
<!-- END ADMISSION SWEEP TABLE -->

Readings:

- **The margin is the dominant dial at the deploy defaults.** With `failover_penalty_blocks` 500
  and an X tier cost of 1200 + 300, an over-threshold primary worker still looks cheaper than X
  (500 vs 1500), so the policy rarely fails over on its own and the gate decides. At margin 0
  every primary worker is always excluded and all traffic goes to a proxy; at 1000 nothing is
  excluded and primary keeps 76.8% of requests. In between, primary share rises monotonically
  44.1% -> 76.8%.
- **Steering away from primary costs cache locality.** As the margin rises, steering exclusions
  fall (1140 -> 0) and the primary cache hit rate rises from 0% (margin 0) to 65.4% with no gate
  at all (and 42.5% even at margin 10). Each steered request pays paid spill and loses the
  primary prefix it already had.
- **No gate is not enough to protect primary.** With the margin above any queue (1000) primary
  occupancy still reaches 189% mean / 270% max — far past the 0.8 threshold — because the
  failover penalty is below the tier cost and the policy does not move the traffic. To hold
  primary at its failover point the policy needs a `failover_penalty_blocks` comparable to
  `<tier>.penalty_blocks + <tier>.weight_blocks`; otherwise raising the admission margin only
  shifts the decision from the gate to the baseline scorer.
- **Set the margin above the policy's failover point.** The policy fails over once either signal
  crosses `occupancy_threshold`; the gate must not exclude the worker before that happens.
  Measure the engine waiting depth when primary crosses the threshold on a representative run
  and choose a margin above it. The deploy default is `256`, far above the single-digit knee
  this sweep shows for a four-concurrent worker, and `spillover-deploy` also emits it as worker
  environment (`admission/<model>/primary.env`).
- **Zero 529s and failures here** because the proxies absorb everything the gate steers away. The
  `admission_margin_low` / `admission_margin_high` scenarios and the
  `margin_above_failover_keeps_more_on_primary` test assert the comparison; an inline scenario in
  the test suite covers the all-saturated 529 path.

## Recommended starting points

These are starting values for one model's policy parameters, to be confirmed with a sweep on
the real scenario shape. Pick the profile by what the model's traffic cares about: a tight TTFT
budget favours spilling early; primary GPU utilisation and paid-spill cost favour spilling late.
The policy reads occupancy from each worker's advertised `total_kv_blocks` and `max_num_seqs`,
so the profiles omit the fallbacks; set `primary_capacity_blocks` / `primary_max_requests` only
for workers that cannot advertise their own capacity.

### Spill early (latency first)

```yaml
occupancy_threshold: 0.8
failover_penalty_blocks: 1600
pending_weight_blocks: 4
tiers:
  - {name: X, dp_ranks: [...], penalty_blocks: 1200, weight_blocks: 300}
  - {name: Y, dp_ranks: [...], penalty_blocks: 2000, weight_blocks: 700}
```

Reasoning: spill begins as soon as primary reaches its failover point so a burst does not push
TTFT up on the primary fleet. In the sweep the 0.8 / 1600 point holds peak primary occupancy at
80.4% mean / 81.2% max while spilling 55.7% overall; the failover penalty is set just above the
cheapest tier's total cost (X: 1200 + 300 = 1500) so an over-threshold host is at least as
expensive as X. At the low end of the sweep (penalty 100) the penalty is far below that and
primary overshoots to 117.7% mean / 136.8% max — raising the penalty, not lowering the threshold,
is what caps primary. Class stickiness here is 79.1%; if keeping conversations together matters
more, 800 is the stickiest point (84.3%) but only caps primary at 92.5% mean / 104.2% max.

### Spill late (utilisation first)

```yaml
occupancy_threshold: 0.9
failover_penalty_blocks: 1600
pending_weight_blocks: 4
tiers:
  - {name: X, dp_ranks: [...], penalty_blocks: 1200, weight_blocks: 300}
  - {name: Y, dp_ranks: [...], penalty_blocks: 2000, weight_blocks: 700}
```

Reasoning: fill the primary GPUs first and pay for proxy spill as late as possible, which is what
the 0.9 / 1600 point shows: primary peak 89.7% mean / 90.6% max, overall proxy 55.2% (against
55.7% for the early profile at 0.8 / 1600), and 77.9% class stickiness. At this threshold the
penalty must still be near the X tier cost (1500): a lower penalty lets the baseline pile onto
primary until the KV fraction overshoots and the policy then spills more overall — 0.9 / 800
peaks at 93.2% mean / 106.9% max but spills 56.6% overall. Do not go below ~400 at this
threshold: penalty 100-200 lets primary reach 117.7% mean / 136.8% max before the policy reacts,
which queues TTFT on the very fleet the profile is meant to fill.

### Things to check before locking values in

- **Occupancy estimate vs real KV use.** The KV signal compares router-tracked decode blocks to
  the worker's advertised `total_kv_blocks` (or the `primary_capacity_blocks` fallback); if that
  estimate is optimistic, spill starts late and primary queues. The concurrency signal compares
  the projected request count to the advertised `max_num_seqs`. Level 2 (`lib/spillover/e2e`)
  compares both with mocker-reported usage.
- **Penalty in the same units as the baseline.** `failover_penalty_blocks` only changes the
  ordering if it is comparable to the baseline prefill/decode cost and to the proxy tiers'
  `penalty_blocks + weight_blocks`. Both scale with `block_size` and prompt length, so re-run a
  sweep after changing the capacity estimates or `block_size`; the profile above assumes 1200
  KV blocks per worker and `block_size` 16.
- **Multiple primary workers.** These numbers come from a one-worker scenario. With several primary
  workers the baseline spreads load across them and large penalties spill more; sweep
  `failover_penalty_blocks` on the realistic host count before choosing a value.
- **Admission margin vs failover point.** `DYN_ADMISSION_QUEUE_MARGIN` is per worker process, not
  a policy value. Confirm the realized engine waiting depth at the moment primary crosses
  `occupancy_threshold`; if the gate fires first it will move cached conversations to a proxy
  before the policy wanted to, and no policy sweep will show it. See the admission sweep above.
- **Thresholds above 1.0 meet the worker admission gate.** Each worker process also runs a
  concurrency gate sized `ceil(1.5 * max_num_seqs * data_parallel_size)` from the same advertised
  capacity (`lib/runtime/src/admission_gate.rs`), and queues requests beyond it in the worker
  process instead of the engine. A threshold up to 1.5 stays under it; above that, set
  `DYN_ENGINE_REQUEST_LIMIT` on the primary workers or the gate, not the policy, sets the queue.
