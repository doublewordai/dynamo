# Spillover implementation plan

Design: https://claude.ai/artifact/2hQ78AEYMM6RMRUSNzPqME (version 4).

Every model runs as two Dynamo deployments, `<model>@interactive` and `<model>@throughput`.
Each is one worker set: our SGLang workers plus third-party proxy workers that register as if
they were SGLang. A worker-selection policy (`dw-spillover`) ranks eligible workers by
cache affinity, then hosted-to-proxy failover, then proxy tier preference. Dynamo's source is
not edited: the policy uses the published plugin API, and the proxy is an ordinary worker.

This is Doubleword's Dynamo fork, not the standalone spillover repo the design was first
prototyped in. The work now lives under `lib/` and `docs/spillover/`, and the fork's own
Python build links the policy.

## Repository layout

| Path | What | Owner task |
|---|---|---|
| `lib/router-plugins/spillover` | The policy: baseline (port of the fork's default scorer/picker), tier scorer, params, factory | F1 |
| `lib/router-plugins/catalog` | Registration crate linked into Dynamo's Python bindings | scaffold |
| `lib/spillover/testkit` | Builds router selection inputs without a frontend | scaffold |
| `lib/spillover/proxy-core` | Proxy logic with no Dynamo runtime: request carrier, provider client, renderers, virtual cache, retokenizer, config | F2, F3 |
| `lib/spillover/proxy-worker` | The worker binary: Dynamo runtime wiring | F2 |
| `lib/spillover/routing-sim` | Deterministic in-process simulation, scenarios, CI assertions | F5 |
| `lib/spillover/deploy` | Generates router-policy and proxy configs from one deployment description | F4 |
| `lib/spillover/e2e`, `.github/workflows/spillover.yml` | End-to-end simulation with the real frontend and mock workers; CI | F4 |
| `docs/spillover/` | This plan, tuning notes and image notes | F4 |

## Fork plugin API

The standalone prototype was built against upstream `ai-dynamo/dynamo@494d6e24`, whose plugin
API scores a *batch* of candidates at once. This fork's API is an earlier revision: the scorer
is called per candidate,

```rust
WorkerScorer::score(&mut self, ctx, candidate: &WorkerCandidate) -> Result<f64, _>
```

and the fork's default scorer/picker live privately in
`lib/kv-router/src/scheduling/selector/default.rs`. `lib/router-plugins/spillover/src/baseline/`
is therefore a per-candidate port of *this fork's* `DefaultWorkerScorer` and
`DefaultWorkerPicker` (softmax by `router_temperature`, ties sampled, seedable RNG), using only
public plugin inputs. The default's batch-wide `min_active_prefill_tokens` shift is equal for
all candidates, so it does not change ranking; the port documents that.

The baseline approach is:

- A model **with no entry** in the policy parameters gets `WorkerSelectionPolicy::default(
  config.clone(), role.default_selector_label())` — exactly Dynamo's default. Its decisions
  equal `DefaultWorkerSelector`'s for a fuzzed grid of cache/load shapes and temperatures.
- A model **with parameters** gets scorers `[baseline, TierScorer]` and the baseline picker.
  A parameter set whose tiers match no worker and whose failover never triggers must still pick
  exactly what the default picks.

`build_policy(config, role, model_name, params, rng)` is the seam `routing-sim` drives.

The port and its equivalence tests are specified in
[`docs/spillover/tasks/F1.md`](tasks/F1.md).

## Phases

### Phase A: port and test inside the fork (now, parallel)

| Task | Scope | Done when |
|---|---|---|
| F1 | `tier.rs` per-candidate costs; `policy.rs` baseline/default split; `baseline/` per-candidate port; equivalence and tier tests | For models without parameters, decisions equal `DefaultWorkerSelector` (seeded); for parametered models with no matching tier, likewise |
| F2 | `orig.rs`, `errors.rs`, `upstream.rs`, proxy worker binary | Carrier round-trips; SSE client against a mock server (chunks, `[DONE]`, 429 + Retry-After, 5xx, in-stream error, broken stream, cancel on drop); worker registers and serves end to end |
| F3 | GLM/Hermes and DeepSeek renderers, `vcache.rs`, `retokenize.rs`, `config.rs` | Round-trips through `dynamo-parsers`/`dynamo-parsers-v2`; virtual-cache hashes equal `dynamo-kv-router`'s; retokenized ids decode to the exact text |
| F5 | `routing-sim` | Scenarios below pass as `cargo test`; `routing-sim <scenario.yaml>` prints a report |
| F4 | `deploy`, `e2e`, docs, `.github/workflows/spillover.yml` | Generator tests pass; `run.sh` validated with `bash -n`; workflow runs tests, clippy, `dw-proxy-worker` build and routing-sim scenarios into the summary |

### Phase B: integrate

- Wire proxy-core fully into proxy-worker; run the e2e simulation locally.
- Dockerfile for `dw-proxy-worker` (the fork's `lib/spillover/proxy-worker/Dockerfile`); a
  frontend image built with the catalog linked.
- Policy YAML per deployment (`--router-policy-config`), `--router-track-active-blocks` on.
- Because the crates build inside the fork's workspace, they resolve `dynamo-kv-router` to the
  fork's `lib/kv-router`, so registry types match by construction; no `[patch]` is needed.

### Phase C: ship

- onwards (control-layer): stamp `nvext.extra_fields: ["dw.orig.v1:..."]` for Dynamo endpoints,
  strip any `nvext` echoed in responses, map serving class to `<model>@<class>`.
- Deployment (internal): proxy chart and secrets, class names on SGLang workers, manifests drop
  the OpenRouter deployment, scouter off, fusillade batch concurrency capped.
- Rollout: staging with a proxy-only model; then GLM-5.3 interactive; then the rest.

### Phase D: later

Review what the fork's frontend no longer needs (admission gate, queue margins, 529 steering).

## Simulation

Two levels. Both answer: how does traffic split between hosted workers, proxy X and proxy Y as
load changes, and do conversations stay on the worker that holds their cache?

### Level 1: in-process, deterministic (every PR)

`lib/spillover/routing-sim` drives the real policy (`build_policy` with a seeded picker) through
`WorkerSelector::select_worker`, using `testkit` to build each `SchedulingRequest`. Virtual
time, seeded randomness, runs in seconds.

Model:

- **Hosted worker**: KV capacity (blocks), prefill rate (tokens/s), decode rate (tokens/s per
  request) with a batching slowdown, max concurrent requests. Holds a prefix cache (LRU over
  block hashes, bounded by free capacity). A request that does not fit waits in the worker's
  queue.
- **Proxy worker**: DP rank for its tier, time to first token and decode rate from
  distributions, optional concurrency limit that returns a rate-limit error. Virtual cache with
  TTL, as `vcache` publishes it.
- **Router signals**, as Dynamo's router computes them: overlap per worker from caches; load per
  worker from the router's own tracking (active requests, in-flight prompt tokens until first
  token, active decode blocks).
- **Retries**: a rate-limited proxy request is re-routed with that proxy excluded, as migration
  would.
- **Workload**: sessions with a shared system prompt, multi-turn conversations where each turn
  extends the previous prompt, think time between turns, output length distribution, and an
  arrival-rate profile per scenario.

Report (JSON and a markdown table), per time window and overall: share per worker class, hosted
occupancy, cache hit rate, stickiness (follow-up turns routed to the previous turn's worker),
spill order, failed requests.

Scenarios and assertions (`lib/spillover/routing-sim/scenarios/*.yaml`, run by `cargo test`):

| Scenario | Assertion |
|---|---|
| `low_load` (50% of hosted capacity) | hosted share >= 99% |
| `overload_ramp` (ramp to 200% and back) | proxy share <= 1% while hosted occupancy is under threshold; hosted occupancy >= 80% at peak; X share > Y share; traffic returns to hosted after the ramp |
| `stickiness` (long multi-turn sessions) | class stickiness >= 98% while the previous class is available; worker stickiness within 2 points of `DefaultWorkerSelector` on the same seed |
| `proxy_rate_limited` (X capped) | overflow beyond X goes to Y; no request fails |
| `no_parameters` | every decision equals `DefaultWorkerSelector` with the same seed |
| `hosted_outage` (hosted removed then restored) | proxies carry all traffic during the outage; hosted share recovers after restore |

### Level 2: end to end (nightly and on demand)

`lib/spillover/e2e`: a real Dynamo frontend from *this fork's* Python build, `python -m
dynamo.mocker` hosted workers (GPU-free simulated engines with real KV events), two
`dw-proxy-worker` processes pointed at a fake provider (streams canned output with configurable
latency and 429s), and a multi-turn load generator. File discovery, TCP request plane and ZMQ
event plane, so no etcd or NATS. Per-worker request counts come from worker logs and frontend
metrics, and are compared with the Level 1 prediction for the same scenario within tolerances.
This is the only level that exercises the real frontend, proxy registration and card matching.

## Frontend build includes the policy

No `[patch]` and no `scripts/build-frontend.sh` are needed in the fork:

- `lib/router-plugins/catalog` depends on `dw-spillover-policy` and registers it in
  `RouterPluginRegistry` (`register`).
- `lib/bindings/python/Cargo.toml` lists `dynamo-worker-selection-policy-catalog` as an optional
  dependency and enables it through the `custom-policy` feature, which is in `default`.
- So the fork's documented Python dev build already includes `dw-spillover`:
  `maturin develop` in `lib/bindings/python`, then `pip install -e .` (plus
  `lib/gpu_memory_service`), and `python -m dynamo.frontend --help` succeeds. The policy is
  selected only when a router-policy YAML names `dw-spillover`.

## CI

- `.github/workflows/spillover.yml` on PRs touching `lib/spillover/**`,
  `lib/router-plugins/spillover/**` or `lib/router-plugins/catalog/**`: `cargo fmt --check`,
  `cargo clippy -D warnings` and `cargo test` for the spillover crates, build
  `dw-proxy-worker`, run `routing-sim` scenarios, and write each scenario's markdown to the job
  summary.
- No nightly end-to-end workflow exists yet; `lib/spillover/e2e/run.sh` is run by hand until a
  frontend image ships the catalog.

## Phase B status

| Item | Result |
|---|---|
| Frontend with our catalog | Fork Python build (`maturin develop` + `pip install -e .`) links `lib/router-plugins/catalog`; `custom-policy` is a default feature; the frontend starts with `dw-spillover` selectable. |
| Proxy image | `lib/spillover/proxy-worker/Dockerfile`; see `docs/spillover/images.md`. Not built here (Docker unavailable). |
| Level 2 end to end | `lib/spillover/e2e/run.sh` on the real frontend: 4 workers in one set, 0 failures, shares within tolerance of `config/level1-equivalent.yaml`. |
| Deployment config | `spillover-deploy` generates router-policy YAML and proxy configs from `lib/spillover/deploy/config/deployments.yaml`. |
| Tuning | `routing-sim sweep`, `docs/spillover/tuning.md`: tier penalty is the main dial. |
| Retokenizer | Pre-token boundaries from the model tokenizer; ~99.9% of Chinese ids stream early. |
| Reasoning start | `render::reasoning_start` from `extra_args`; renderers correct with thinking on or off. |

## Remaining follow-ups

- A Chinese (or other boundary-free) run with no punctuation still waits for its end in the
  retokenizer; add a maximum hold length if that matters in practice.
- Neither renderer round-trips a second reasoning block mid-response.
- `lib/spillover/e2e/loadgen.py` stamps `dw.orig` itself until onwards does (Phase C).
- Build and push both images once Docker is available.
- The deploy generator's tests require `dw-spillover-policy` and `dw-proxy-core` to compile;
  until the F1/F2/F3 ports land, only the path handling here can be checked.

## Open questions

- Whether the occupancy estimate (router-tracked decode blocks) is close enough to real KV use;
  Level 2 compares it with mocker-reported usage.
- Per-model hosted capacity is a parameter until plugins can read `total_kv_blocks`.
- Name separator: `@` in Dynamo names avoids the `:` serving-class collision in onwards.
