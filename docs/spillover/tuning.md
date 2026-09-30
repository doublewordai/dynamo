<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tuning the spillover policy

This note explains what the policy's knobs do and gives two starting profiles: spill early
(latency first) and spill late (hosted utilisation first). Everything here comes from `routing-sim sweep` on the `overload_ramp` scenario
(one hosted worker, two X and two Y proxy workers, arrivals ramped to ~6x hosted capacity and
back). Sweeps are deterministic and keep the scenario seed, so the numbers below reproduce:

```sh
cargo run -p dw-routing-sim -- sweep lib/spillover/routing-sim/scenarios/overload_ramp.yaml \
    --param failover_penalty_blocks=100,200,400,800,1600 \
    --param occupancy_threshold=0.8,0.9 \
    --jobs 8 --markdown out.md --json out.json
```

The tables below are excerpted columns of that `--markdown` output, not a verbatim paste. The
generator's full header is
`settings | requests | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | worker sticky % | cache hit % | hosted cache hit % | steer excl | 529 | failures`
followed by one `<tier> %` column per tier. The main sweep table drops `hosted cache hit %`,
`steer excl`, `529` and the `Y %` column; the admission table is a derived subset with its own
column names (`margin`, `hosted share %`), so neither can be diffed byte-for-byte against a
fresh `--markdown` run.

Every cost is in KV blocks; lower is better. The baseline scorer (cache affinity plus current
load) runs first and the spillover scorer is stacked on top, so a setting only matters when it
is large enough to change the ordering that the baseline leaves behind.

## What each setting does

| Setting | Effect |
|---|---|
| `occupancy_threshold` | Fraction of `hosted_capacity_blocks` at which a hosted worker counts as full. Below it a hosted worker gets no failover cost; at or above it every hosted worker gets `failover_penalty_blocks`. Lower values spill earlier; higher values let hosted fill further before any penalty applies. |
| `failover_penalty_blocks` | Cost added to a hosted worker once it is at or over the threshold. Raising it makes hosted workers look busier, so more traffic spills, the hosted peak occupancy falls, and follow-up turns are more likely to stay on the same class (hosted or proxy). This is the main spill/stickiness dial. |
| `<tier>.penalty_blocks` | Fixed "always full" cost for every worker in a proxy tier. Raising it makes that tier less attractive; the spilling traffic shifts to other tiers or back to hosted. |
| `<tier>.weight_blocks` | Tier preference between proxy tiers: smaller is preferred. Ordering X below Y keeps the cheaper/faster tier first. |
| `pending_weight_blocks` | Cost per active request on any worker. It is a load-spreading term; with a single hosted worker it mostly moves traffic off a busy proxy or host. |

## Main sweep: failover penalty x occupancy threshold

Peak columns are the scenario's `peak` phase (t = 70..95 s). `X %` is the share of all requests
served by tier X; Y carried nothing in any of these points.

<!-- BEGIN SWEEP TABLE -->
| settings | requests | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | worker sticky % | cache hit % | failures | X % |
|---|---|---|---|---|---|---|---|---|---|---|
| failover_penalty_blocks=100, occupancy_threshold=0.8 | 481 | 11.6 | 22.0 | 69.9 | 90.2 | 81.5 | 88.2 | 65.6 | 0 | 11.6 |
| failover_penalty_blocks=100, occupancy_threshold=0.9 | 481 | 10.0 | 21.3 | 72.1 | 92.6 | 82.7 | 90.7 | 66.7 | 0 | 10.0 |
| failover_penalty_blocks=200, occupancy_threshold=0.8 | 481 | 13.7 | 22.8 | 67.9 | 85.6 | 82.6 | 89.4 | 64.0 | 0 | 13.7 |
| failover_penalty_blocks=200, occupancy_threshold=0.9 | 481 | 10.0 | 21.3 | 72.1 | 92.6 | 82.7 | 90.7 | 66.7 | 0 | 10.0 |
| failover_penalty_blocks=400, occupancy_threshold=0.8 | 481 | 11.6 | 23.7 | 67.3 | 81.8 | 82.9 | 90.5 | 64.5 | 0 | 11.6 |
| failover_penalty_blocks=400, occupancy_threshold=0.9 | 481 | 10.0 | 21.3 | 72.1 | 92.6 | 82.7 | 90.7 | 66.7 | 0 | 10.0 |
| failover_penalty_blocks=800, occupancy_threshold=0.8 | 481 | 11.6 | 23.7 | 67.3 | 81.8 | 82.9 | 90.5 | 64.5 | 0 | 11.6 |
| failover_penalty_blocks=800, occupancy_threshold=0.9 | 481 | 10.0 | 21.3 | 72.1 | 92.6 | 82.7 | 90.7 | 66.7 | 0 | 10.0 |
| failover_penalty_blocks=1600, occupancy_threshold=0.8 | 481 | 11.6 | 23.7 | 67.3 | 81.8 | 82.9 | 90.5 | 64.5 | 0 | 11.6 |
| failover_penalty_blocks=1600, occupancy_threshold=0.9 | 481 | 10.0 | 21.3 | 72.1 | 92.6 | 82.7 | 90.7 | 66.7 | 0 | 10.0 |
<!-- END SWEEP TABLE -->

Readings:

- **At threshold 0.8 the penalty matters, then saturates.** Raising it from 100 to 400 moves
  peak proxy share from 22.0% to 23.7% and brings the hosted peak max down from 90.2% to 81.8%,
  so spill happens earlier in the ramp rather than in larger volume (overall proxy share is
  11.6% at both ends, 13.7% at 200). Above ~400 nothing changes: once a hosted worker is over
  the threshold the choice is binary, so a larger penalty does not spill more.
- **At threshold 0.9 the penalty is inert on this scenario.** All five rows are identical. The
  router's observed hosted occupancy rarely crosses 0.9 before selection, so the failover cost is
  never applied and spill is driven by the baseline load term instead. A threshold near 0.9 is
  therefore a "only spill when the baseline already says the host is full" setting, not a way to
  tune spill with the penalty.
- **Stickiness is flat across the grid.** Peak class stickiness is 82.6-82.9% at every point
  except penalty 100 at 0.8 (81.5%), and worker stickiness is 88-91%. Cache affinity keeps
  conversations on their worker whichever threshold is used; the penalty only changes when new
  conversations start to spill.
- **No failures and no Y traffic** in any point: X alone has enough capacity, and the failure
  path is exercised by the `proxy_rate_limited` scenario instead.

## Secondary sweep: proxy tier penalty

Raising the X tier's fixed penalty makes X less attractive, so spill falls until Y starts
absorbing the overflow at the top of the range (`occupancy_threshold` 0.8, `failover_penalty_blocks` 500):

| X.penalty_blocks | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | X % | Y % |
|---|---|---|---|---|---|---|---|
| 0 | 33.5 | 55.8 | 27.9 | 38.4 | 73.3 | 33.5 | 0.0 |
| 200 | 26.6 | 53.7 | 38.6 | 51.9 | 71.4 | 26.6 | 0.0 |
| 400 | 21.6 | 45.3 | 47.0 | 57.1 | 71.4 | 21.6 | 0.0 |
| 800 | 16.8 | 31.2 | 61.3 | 80.7 | 80.0 | 16.8 | 0.0 |
| 1600 | 11.9 | 20.8 | 67.3 | 86.0 | 91.0 | 7.3 | 4.6 |

This is the monotonicity the test suite checks: a larger tier penalty can only reduce peak proxy
share. It also shows the practical range: below ~200 X absorbs too much traffic, above ~800 Y
starts to be used and hosted fills to the point of queueing.

## Admission margin (backend gate, not a policy setting)

The engine-queue admission margin is **not** a router-policy field. The fork reads a single
`DYN_ADMISSION_QUEUE_MARGIN` from each **worker process**
(`lib/runtime/src/admission_gate.rs`, parsed in `lib/runtime/src/admission_margin.rs`); the
frontend never reads it and there is no per-model override map. While a hosted worker's engine
waiting queue is at or above its margin that worker is excluded from selection, so cached
conversations are pushed to a proxy. A worker whose engine has never reported its waiting count
is unenforced, and `dw-proxy-worker` never reports one, so the margin cannot apply to a proxy.
When every hosted worker is at its margin and no proxy can take the request, the router refuses
it; the frontend turns the overload error into HTTP 529 (`DYN_HTTP_OVERLOAD_STATUS_CODE`).

The simulation models this with an `admission:` block: `hosted_queue_margin` applies to every
hosted worker and `hosted_queue_margin_overrides` gives individual worker ids their own value
(mirroring one margin per worker process). Stage one as `admission_queue_margin`:

```sh
cargo run -p dw-routing-sim -- sweep lib/spillover/routing-sim/scenarios/admission_margin_high.yaml \
    --param admission_queue_margin=0,1,2,3,4,6,8,10,16,32,1000 \
    --jobs 8 --markdown out.md --json out.json
```

<!-- BEGIN ADMISSION SWEEP TABLE -->
| margin | requests | proxy % | hosted share % | hosted cache hit % | steer excl | 529 | failures |
|---|---|---|---|---|---|---|---|
| 0 | 568 | 100.0 | 0.0 | 0.0 | 1136 | 0 | 0 |
| 1 | 569 | 55.9 | 44.1 | 41.3 | 817 | 0 | 0 |
| 2 | 565 | 55.0 | 45.0 | 39.6 | 798 | 0 | 0 |
| 3 | 565 | 54.9 | 45.1 | 41.2 | 791 | 0 | 0 |
| 4 | 565 | 54.3 | 45.7 | 41.8 | 793 | 0 | 0 |
| 6 | 556 | 52.3 | 47.7 | 43.3 | 753 | 0 | 0 |
| 8 | 547 | 51.0 | 49.0 | 43.9 | 734 | 0 | 0 |
| 10 | 538 | 50.0 | 50.0 | 45.7 | 710 | 0 | 0 |
| 16 | 531 | 46.7 | 53.3 | 46.6 | 652 | 0 | 0 |
| 32 | 480 | 35.2 | 64.8 | 54.4 | 446 | 0 | 0 |
| 1000 | 371 | 0.0 | 100.0 | 68.6 | 0 | 0 | 0 |
<!-- END ADMISSION SWEEP TABLE -->

Readings:

- **The margin is the spill dial while the policy's failover point is unreachable.** The scenario's
  hosted workers cap decode occupancy far below `occupancy_threshold`, so the policy never fails
  over and the gate alone decides. At margin 0 every hosted worker is always excluded and all
  traffic goes to a proxy; at 1000 nothing is excluded and hosted keeps every request.
- **Steering away from hosted costs cache locality.** As the margin rises, steering exclusions
  fall and the hosted cache hit rate rises from 0% (margin 0, and 41.2% at margin 3) to 68.6%
  with no gate at all. Each steered request pays paid spill and loses the
  hosted prefix it already had.
- **Set the margin above the policy's failover point.** The policy fails over on hosted decode
  occupancy; the gate must not exclude the worker before that happens. Measure the engine
  waiting depth when hosted occupancy crosses `occupancy_threshold` on a representative run and
  choose a margin above it. The deploy default is `256`, far above the single-digit knee this
  sweep shows for a four-concurrent worker, and `spillover-deploy` also emits it as worker
  environment (`admission/<model>/hosted.env`).
- **Zero 529s and failures here** because the proxies absorb everything the gate steers away. The
  `admission_margin_low` / `admission_margin_high` scenarios and the
  `margin_above_failover_keeps_more_on_hosted` test assert the comparison; an inline scenario in
  the test suite covers the all-saturated 529 path.

## Recommended starting points

These are starting values for one model's policy parameters, to be confirmed with a sweep on
the real scenario shape. Pick the profile by what the model's traffic cares about: a tight TTFT
budget favours spilling early; hosted GPU utilisation and paid-spill cost favour spilling late.

### Spill early (latency first)

```yaml
occupancy_threshold: 0.85
hosted_capacity_blocks: <measured total hosted KV blocks / worker count>
failover_penalty_blocks: 400
pending_weight_blocks: 4
tiers:
  - {name: X, dp_ranks: [...], penalty_blocks: 200, weight_blocks: 0}
  - {name: Y, dp_ranks: [...], penalty_blocks: 200, weight_blocks: 50}
```

Reasoning: spill begins before hosted is completely full so a burst does not push TTFT up on the
hosted fleet, and the failover penalty keeps follow-up turns where they started (hosted or
proxy). The sweep
shows the 0.8-threshold points from 400 up holding class stickiness near 83% while capping
hosted peak occupancy near 82%; 0.85 sits between that and the untriggered 0.9 points. Start at 400 and only
raise it if the hosted peak max is still too high; the sweep shows no benefit above ~400.

### Spill late (utilisation first)

```yaml
occupancy_threshold: 0.9
hosted_capacity_blocks: <measured total hosted KV blocks / worker count>
failover_penalty_blocks: 100
pending_weight_blocks: 4
tiers:
  - {name: X, dp_ranks: [...], penalty_blocks: 200, weight_blocks: 0}
  - {name: Y, dp_ranks: [...], penalty_blocks: 200, weight_blocks: 50}
```

Reasoning: fill the hosted GPUs first and pay for proxy spill as late as possible, which is what
the 0.9 rows show (hosted peak mean 72%, max 93%, only 10% proxy). Keep the failover penalty low
so it does not move traffic that the baseline would have kept on hosted; once the baseline says
hosted is full, spill happens anyway. Keep tier penalties around 200 so X is used before Y, and
raise `X.penalty_blocks` toward 800 if X is taking traffic that hosted could serve.

### Things to check before locking values in

- **Occupancy estimate vs real KV use.** The threshold compares router-tracked decode blocks to
  `hosted_capacity_blocks`; if that estimate is optimistic, spill starts late and hosted queues.
  Level 2 (`lib/spillover/e2e`) compares it with mocker-reported usage.
- **Penalty in the same units as the baseline.** The penalty only matters if it is comparable to
  the baseline prefill/decode cost, which scales with `block_size` and prompt length. Re-run a
  sweep after changing `hosted_capacity_blocks` or `block_size`.
- **Multiple hosted workers.** These numbers come from a one-worker scenario. With several hosted
  workers the baseline spreads load across them and large penalties spill more; sweep
  `failover_penalty_blocks` on the realistic host count before choosing a value.
- **Admission margin vs failover point.** `DYN_ADMISSION_QUEUE_MARGIN` is per worker process, not
  a policy value. Confirm the realized engine waiting depth at the moment hosted crosses
  `occupancy_threshold`; if the gate fires first it will move cached conversations to a proxy
  before the policy wanted to, and no policy sweep will show it. See the admission sweep above.
