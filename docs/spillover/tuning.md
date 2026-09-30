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
| `occupancy_threshold` | Fraction of `primary_capacity_blocks` at which a primary worker counts as full. Below it a primary worker gets no failover cost; at or above it every primary worker gets `failover_penalty_blocks`. Lower values spill earlier; higher values let primary fill further before any penalty applies. |
| `failover_penalty_blocks` | Cost added to a primary worker once it is at or over the threshold. Raising it makes primary workers look busier, so more traffic spills, the primary peak occupancy falls, and follow-up turns are more likely to stay on the same class (primary or proxy). This is the main spill/stickiness dial. |
| `<tier>.penalty_blocks` | Fixed "always full" cost for every worker in a proxy tier. Raising it makes that tier less attractive; the spilling traffic shifts to other tiers or back to primary. |
| `<tier>.weight_blocks` | Tier preference between proxy tiers: smaller is preferred. Ordering X below Y keeps the cheaper/faster tier first. |
| `pending_weight_blocks` | Cost per active request on any worker. It is a load-spreading term; with a single primary worker it mostly moves traffic off a busy proxy or host. |

## Main sweep: failover penalty x occupancy threshold

Peak columns are the scenario's `peak` phase (t = 70..95 s). `X %` is the share of all requests
served by tier X; Y is a strictly more expensive last resort and carried nothing in any of these
points (the `proxy_rate_limited` scenario exercises it).

<!-- BEGIN SWEEP TABLE -->
| settings | requests | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | worker sticky % | cache hit % | failures | X % |
|---|---|---|---|---|---|---|---|---|---|---|
| failover_penalty_blocks=100, occupancy_threshold=0.8 | 1140 | 54.5 | 66.1 | 116.2 | 136.8 | 76.2 | 64.8 | 47.6 | 0 | 54.5 |
| failover_penalty_blocks=100, occupancy_threshold=0.9 | 1140 | 54.5 | 66.1 | 116.2 | 136.8 | 76.9 | 65.3 | 47.6 | 0 | 54.5 |
| failover_penalty_blocks=200, occupancy_threshold=0.8 | 1140 | 55.5 | 67.9 | 109.3 | 127.1 | 84.5 | 62.9 | 48.4 | 0 | 55.5 |
| failover_penalty_blocks=200, occupancy_threshold=0.9 | 1140 | 54.8 | 67.7 | 108.8 | 135.3 | 82.8 | 67.3 | 49.7 | 0 | 54.8 |
| failover_penalty_blocks=400, occupancy_threshold=0.8 | 1140 | 55.8 | 72.9 | 103.2 | 121.7 | 77.1 | 62.1 | 46.6 | 0 | 55.8 |
| failover_penalty_blocks=400, occupancy_threshold=0.9 | 1140 | 55.3 | 61.7 | 92.9 | 111.3 | 69.6 | 61.1 | 46.1 | 0 | 55.3 |
| failover_penalty_blocks=800, occupancy_threshold=0.8 | 1140 | 56.5 | 68.9 | 79.7 | 89.0 | 65.7 | 61.8 | 47.6 | 0 | 56.5 |
| failover_penalty_blocks=800, occupancy_threshold=0.9 | 1139 | 49.6 | 58.1 | 88.5 | 89.9 | 61.9 | 55.5 | 46.1 | 0 | 49.6 |
| failover_penalty_blocks=1600, occupancy_threshold=0.8 | 1140 | 51.6 | 67.0 | 78.9 | 79.9 | 61.4 | 56.1 | 46.9 | 0 | 51.6 |
| failover_penalty_blocks=1600, occupancy_threshold=0.9 | 1139 | 49.0 | 58.1 | 88.5 | 89.9 | 61.9 | 56.2 | 46.5 | 0 | 49.0 |
<!-- END SWEEP TABLE -->

Readings:

- **The ramp overshoots one worker, so the policy spills at every grid point; the penalty sets
  how early and how deep.** Overall proxy share is 49-57% across the whole grid, because the
  plateau arrival rate is roughly twice what the single primary worker can decode and the
  baseline scorer spills the excess even with a small failover penalty.
- **A small penalty lets the baseline pile onto primary before the policy reacts.** At threshold
  0.8, penalty 100 leaves the peak primary occupancy at 116% mean / 137% max — over capacity, so
  the primary queue is growing while the policy is nominally in charge. Raising the penalty to
  800 brings the peak back to 80% mean / 89% max, and to 1600 to 79% mean / 80% max.
  `failover_penalty_blocks` is therefore the knob that actually holds primary at its failover
  point, not the threshold.
- **Peak proxy share is not monotone in the penalty on this scenario.** It rises from 66% at
  penalty 100 to 73% at 400 (larger penalty spills earlier in the ramp) and then falls back to
  67% at 1600 (the spill is spread over more of the ramp). Use peak *occupancy* to judge the
  penalty, and overall proxy share to judge cost: at threshold 0.8 the total drifts 54.5% ->
  56.5% -> 51.6% as the penalty goes 100 -> 400 -> 1600.
- **The threshold matters only once the penalty is large enough to engage it.** At penalty 100
  the 0.8 and 0.9 rows are identical: the baseline load term, not the failover cost, decides.
  At penalty 800-1600 the threshold separates cleanly — 0.9 lets primary fill to 90% mean / 90%
  max and spills less overall (49.0% vs 51.6%), while 0.8 caps it at 80%.
- **Stickiness peaks in the middle of the grid.** Peak class stickiness is 84.5% at threshold
  0.8 / penalty 200 and 82.8% at 0.9 / 200, then falls to ~61% at the top end: an aggressive
  failover penalty starts spilling conversations that the baseline would have kept on primary.
  Worker stickiness tracks it at 55-67%.
- **No failures and no Y traffic** at any point: X alone has enough capacity, and the failure
  path is exercised by the `proxy_rate_limited` scenario instead.

## Secondary sweep: proxy tier penalty

Raising the X tier's fixed penalty makes X less attractive, so less traffic spills; with Y kept
strictly more expensive (penalty 2000) X stays the only spill tier and the peak proxy share falls
monotonically (`occupancy_threshold` 0.8, `failover_penalty_blocks` 500):

| X.penalty_blocks | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | X % | Y % |
|---|---|---|---|---|---|---|---|
| 0 | 64.6 | 75.3 | 56.4 | 74.2 | 70.2 | 64.6 | 0.0 |
| 200 | 61.0 | 73.5 | 67.7 | 79.9 | 69.0 | 61.0 | 0.0 |
| 400 | 56.8 | 69.2 | 72.6 | 79.9 | 69.2 | 56.8 | 0.0 |
| 800 | 51.8 | 67.3 | 78.7 | 79.9 | 60.1 | 51.8 | 0.0 |
| 1600 | 54.3 | 66.1 | 116.2 | 136.8 | 76.2 | 54.1 | 0.2 |

This is the monotonicity the test suite checks: a larger tier penalty can only reduce peak proxy
share. It also shows the practical range: below ~200 X absorbs traffic that primary could serve
(primary peak occupancy only 56% mean), while at 1600 X is expensive enough that the baseline
keeps more traffic on primary and primary overshoots capacity (117% mean / 137% max) — the point
past which raising the tier penalty is counterproductive. Note the overall proxy share dips from
64.6% to 51.8% and then rises to 54.3% at 1600 even though the peak share keeps falling:
requests are spread differently across the ramp, so read the peak column for the monotone trend.

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
| 1 | 570 | 56.1 | 43.9 | 41.2 | 821 | 0 | 0 |
| 2 | 566 | 55.1 | 44.9 | 42.8 | 797 | 0 | 0 |
| 3 | 565 | 54.2 | 45.8 | 43.7 | 758 | 0 | 0 |
| 4 | 564 | 54.1 | 45.9 | 38.1 | 778 | 0 | 0 |
| 6 | 556 | 52.9 | 47.1 | 41.6 | 758 | 0 | 0 |
| 8 | 550 | 51.5 | 48.5 | 42.7 | 719 | 0 | 0 |
| 10 | 534 | 49.4 | 50.6 | 44.2 | 693 | 0 | 0 |
| 16 | 524 | 46.4 | 53.6 | 47.9 | 628 | 0 | 0 |
| 32 | 482 | 35.1 | 64.9 | 50.3 | 426 | 0 | 0 |
| 1000 | 424 | 21.2 | 78.8 | 61.3 | 0 | 0 | 0 |
<!-- END ADMISSION SWEEP TABLE -->

Readings:

- **The margin is the dominant dial at the deploy defaults.** With `failover_penalty_blocks` 500
  and an X tier cost of 1200 + 300, an over-threshold primary worker still looks cheaper than X
  (500 vs 1500), so the policy rarely fails over on its own and the gate decides. At margin 0
  every primary worker is always excluded and all traffic goes to a proxy; at 1000 nothing is
  excluded and primary keeps 78.8% of requests. In between, primary share rises monotonically
  43.9% -> 78.8%.
- **Steering away from primary costs cache locality.** As the margin rises, steering exclusions
  fall (1140 -> 0) and the primary cache hit rate rises from 0% (margin 0) to 61.3% with no gate
  at all (and 44.2% even at margin 10). Each steered request pays paid spill and loses the
  primary prefix it already had.
- **No gate is not enough to protect primary.** With the margin above any queue (1000) primary
  occupancy still reaches 166% mean / 268% max — far past the 0.8 threshold — because the
  failover penalty is below the tier cost and the policy does not move the traffic. To hold
  primary at its failover point the policy needs a `failover_penalty_blocks` comparable to
  `<tier>.penalty_blocks + <tier>.weight_blocks`; otherwise raising the admission margin only
  shifts the decision from the gate to the baseline scorer.
- **Set the margin above the policy's failover point.** The policy fails over on primary decode
  occupancy; the gate must not exclude the worker before that happens. Measure the engine
  waiting depth when primary occupancy crosses `occupancy_threshold` on a representative run and
  choose a margin above it. The deploy default is `256`, far above the single-digit knee this
  sweep shows for a four-concurrent worker, and `spillover-deploy` also emits it as worker
  environment (`admission/<model>/primary.env`).
- **Zero 529s and failures here** because the proxies absorb everything the gate steers away. The
  `admission_margin_low` / `admission_margin_high` scenarios and the
  `margin_above_failover_keeps_more_on_primary` test assert the comparison; an inline scenario in
  the test suite covers the all-saturated 529 path.

## Recommended starting points

These are starting values for one model's policy parameters, to be confirmed with a sweep on
the real scenario shape. Pick the profile by what the model's traffic cares about: a tight TTFT
budget favours spilling early; primary GPU utilisation and paid-spill cost favour spilling late.

### Spill early (latency first)

```yaml
occupancy_threshold: 0.8
primary_capacity_blocks: <measured total primary KV blocks / worker count>
failover_penalty_blocks: 1600
pending_weight_blocks: 4
tiers:
  - {name: X, dp_ranks: [...], penalty_blocks: 1200, weight_blocks: 300}
  - {name: Y, dp_ranks: [...], penalty_blocks: 2000, weight_blocks: 700}
```

Reasoning: spill begins as soon as primary reaches its failover point so a burst does not push
TTFT up on the primary fleet. In the sweep the 0.8 / 1600 point holds peak primary occupancy at
78.9% mean / 79.9% max while spilling 51.6% overall; the failover penalty is set just above the
cheapest tier's total cost (X: 1200 + 300 = 1500) so an over-threshold host is at least as
expensive as X. At the low end of the sweep (penalty 100) the penalty is far below that and
primary overshoots to 116% mean / 137% max — raising the penalty, not lowering the threshold,
is what caps primary. Class stickiness here is 61.4%; if keeping conversations together matters
more, 200 is the stickiest point (84.5%) but only caps primary at 109% mean / 127% max.

### Spill late (utilisation first)

```yaml
occupancy_threshold: 0.9
primary_capacity_blocks: <measured total primary KV blocks / worker count>
failover_penalty_blocks: 800
pending_weight_blocks: 4
tiers:
  - {name: X, dp_ranks: [...], penalty_blocks: 1200, weight_blocks: 300}
  - {name: Y, dp_ranks: [...], penalty_blocks: 2000, weight_blocks: 700}
```

Reasoning: fill the primary GPUs first and pay for proxy spill as late as possible, which is what
the 0.9 / 800 point shows: primary peak 88.5% mean / 89.9% max, overall proxy 49.6% (against
51.6% for the early profile at 0.8 / 1600), and 61.9% class stickiness. The penalty sits below
the X tier cost (1500) so an over-threshold host stays attractive and the baseline, not the
policy, decides most spills; raising it to 1600 does not change the 0.9 rows. Do not go below
~400 at this threshold: penalty 100-200 lets primary reach 116% mean / 137% max before the policy
reacts, which queues TTFT on the very fleet the profile is meant to fill.

### Things to check before locking values in

- **Occupancy estimate vs real KV use.** The threshold compares router-tracked decode blocks to
  `primary_capacity_blocks`; if that estimate is optimistic, spill starts late and primary queues.
  Level 2 (`lib/spillover/e2e`) compares it with mocker-reported usage.
- **Penalty in the same units as the baseline.** `failover_penalty_blocks` only changes the
  ordering if it is comparable to the baseline prefill/decode cost and to the proxy tiers'
  `penalty_blocks + weight_blocks`. Both scale with `block_size` and prompt length, so re-run a
  sweep after changing `primary_capacity_blocks` or `block_size`; the profile above assumes
  `primary_capacity_blocks` 1200 and `block_size` 16.
- **Multiple primary workers.** These numbers come from a one-worker scenario. With several primary
  workers the baseline spreads load across them and large penalties spill more; sweep
  `failover_penalty_blocks` on the realistic host count before choosing a value.
- **Admission margin vs failover point.** `DYN_ADMISSION_QUEUE_MARGIN` is per worker process, not
  a policy value. Confirm the realized engine waiting depth at the moment primary crosses
  `occupancy_threshold`; if the gate fires first it will move cached conversations to a proxy
  before the policy wanted to, and no policy sweep will show it. See the admission sweep above.
