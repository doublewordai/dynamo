# Tuning the spillover policy

This note explains what the policy's knobs do and gives starting values for the two serving
classes. Everything here comes from `routing-sim sweep` on the `overload_ramp` scenario
(one hosted worker, two X and two Y proxy workers, arrivals ramped to ~6x hosted capacity and
back). Sweeps are deterministic and keep the scenario seed, so these tables are reproducible:

```sh
cargo run -p dw-routing-sim -- sweep crates/routing-sim/scenarios/overload_ramp.yaml \
    --param failover_penalty_blocks=100,200,400,800,1600 \
    --param occupancy_threshold=0.8,0.9 \
    --jobs 8 --markdown out.md --json out.json
```

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
| failover_penalty_blocks=100, occupancy_threshold=0.8 | 481 | 11.9 | 21.2 | 69.5 | 89.6 | 81.0 | 88.8 | 64.6 | 0 | 11.9 |
| failover_penalty_blocks=100, occupancy_threshold=0.9 | 481 | 11.6 | 22.2 | 72.2 | 93.0 | 71.2 | 84.1 | 63.6 | 0 | 11.6 |
| failover_penalty_blocks=200, occupancy_threshold=0.8 | 481 | 12.9 | 24.0 | 68.0 | 84.6 | 82.6 | 88.4 | 64.1 | 0 | 12.9 |
| failover_penalty_blocks=200, occupancy_threshold=0.9 | 481 | 11.6 | 22.2 | 72.2 | 93.0 | 71.2 | 84.1 | 63.6 | 0 | 11.6 |
| failover_penalty_blocks=400, occupancy_threshold=0.8 | 481 | 12.5 | 24.7 | 67.6 | 82.6 | 82.7 | 88.4 | 63.5 | 0 | 12.5 |
| failover_penalty_blocks=400, occupancy_threshold=0.9 | 481 | 11.6 | 22.2 | 72.2 | 93.0 | 71.2 | 84.1 | 63.6 | 0 | 11.6 |
| failover_penalty_blocks=800, occupancy_threshold=0.8 | 481 | 12.5 | 24.7 | 67.6 | 82.6 | 82.7 | 88.4 | 63.5 | 0 | 12.5 |
| failover_penalty_blocks=800, occupancy_threshold=0.9 | 481 | 11.6 | 22.2 | 72.2 | 93.0 | 71.2 | 84.1 | 63.6 | 0 | 11.6 |
| failover_penalty_blocks=1600, occupancy_threshold=0.8 | 481 | 12.5 | 24.7 | 67.6 | 82.6 | 82.7 | 88.4 | 63.5 | 0 | 12.5 |
| failover_penalty_blocks=1600, occupancy_threshold=0.9 | 481 | 11.6 | 22.2 | 72.2 | 93.0 | 71.2 | 84.1 | 63.6 | 0 | 11.6 |
<!-- END SWEEP TABLE -->

Readings:

- **At threshold 0.8 the penalty matters, then saturates.** Going from 100 to 200 adds about one
  point of overall proxy share and almost three points of peak proxy share, while the hosted
  peak max drops from 89.6% to 84.6% and peak class stickiness rises from 81.0% to 82.6%. Above
  ~400 nothing changes: once a hosted worker is over the threshold the choice is binary, so a
  larger penalty does not spill more.
- **At threshold 0.9 the penalty is inert on this scenario.** All five rows are identical. The
  router's observed hosted occupancy rarely crosses 0.9 before selection, so the failover cost is
  never applied and spill is driven by the baseline load term instead. A threshold near 0.9 is
  therefore a "only spill when the baseline already says the host is full" setting, not a way to
  tune spill with the penalty.
- **Stickiness follows the spill, not the threshold.** Class stickiness is ~82.7% for the points
  that spill at 0.8, and ~71.2% for every 0.9 point. Once a conversation is on a proxy, a higher
  failover penalty makes it stay there instead of flipping back and forth; at 0.9 that mechanism
  never engages.
- **No failures and no Y traffic** in any point: X alone has enough capacity, and the failure
  path is exercised by the `proxy_rate_limited` scenario instead.

## Secondary sweep: proxy tier penalty

Raising the X tier's fixed penalty makes X less attractive, so spill falls until Y starts
absorbing the overflow at the top of the range (`occupancy_threshold` 0.8, `failover_penalty_blocks` 500):

| X.penalty_blocks | proxy % | peak proxy % | peak occ mean % | peak occ max % | peak class sticky % | X % | Y % |
|---|---|---|---|---|---|---|---|
| 0 | 33.9 | 58.0 | 28.1 | 39.2 | 55.6 | 33.9 | 0.0 |
| 200 | 26.8 | 51.9 | 37.3 | 46.6 | 58.8 | 26.8 | 0.0 |
| 400 | 23.3 | 45.8 | 47.1 | 60.0 | 67.6 | 23.3 | 0.0 |
| 800 | 16.4 | 34.6 | 62.1 | 77.6 | 79.4 | 16.4 | 0.0 |
| 1600 | 12.5 | 21.1 | 67.4 | 85.4 | 85.9 | 7.1 | 5.4 |

This is the monotonicity the test suite checks: a larger tier penalty can only reduce peak proxy
share. It also shows the practical range: below ~200 X absorbs too much traffic, above ~800 Y
starts to be used and hosted fills to the point of queueing.

## Recommended starting points

These are starting values for a per-deployment policy YAML, to be confirmed with a sweep on the
real scenario shape. The interactive class has a tight TTFT budget and values conversation
locality; the throughput class cares about hosted GPU utilisation and paid-spill cost.

### Interactive (`<model>@interactive`)

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
hosted fleet, and the failover penalty keeps follow-up turns on their existing class. The sweep
shows the 0.8-threshold points holding class stickiness around 82.7% while capping hosted peak
occupancy near 83%; 0.85 sits between that and the untriggered 0.9 points. Start at 400 and only
raise it if the hosted peak max is still too high; the sweep shows no benefit above ~400.

### Throughput (`<model>@throughput`)

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
the 0.9 rows show (hosted peak mean 72%, max 93%, only ~12% proxy). Keep the failover penalty low
so it does not move traffic that the baseline would have kept on hosted; once the baseline says
hosted is full, spill happens anyway. Keep tier penalties around 200 so X is used before Y, and
raise `X.penalty_blocks` toward 800 if X is taking traffic that hosted could serve.

### Things to check before locking values in

- **Occupancy estimate vs real KV use.** The threshold compares router-tracked decode blocks to
  `hosted_capacity_blocks`; if that estimate is optimistic, spill starts late and hosted queues.
  Level 2 (`sim/e2e`) compares it with mocker-reported usage.
- **Penalty in the same units as the baseline.** The penalty only matters if it is comparable to
  the baseline prefill/decode cost, which scales with `block_size` and prompt length. Re-run a
  sweep after changing `hosted_capacity_blocks` or `block_size`.
- **Multiple hosted workers.** These numbers come from a one-worker scenario. With several hosted
  workers the baseline spreads load across them and large penalties spill more; sweep
  `failover_penalty_blocks` on the realistic host count before choosing a value.
