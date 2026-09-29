# Spillover implementation plan

Design: https://claude.ai/artifact/2hQ78AEYMM6RMRUSNzPqME (version 4).

Every model runs as two Dynamo deployments, `<model>@interactive` and `<model>@throughput`.
Each is one worker set: our SGLang workers plus third-party proxy workers that register as if
they were SGLang. A worker-selection policy (`dw-spillover`) ranks eligible workers by
cache affinity, then hosted-to-proxy failover, then proxy tier preference. Dynamo's source is
not edited: the policy uses the published plugin API, and the proxy is an ordinary worker.

Upstream is pinned at `ai-dynamo/dynamo@494d6e2441bb2374e04ba159dfc1aaa36b726495` (v1.6-dev).

## Repository layout

| Path | What | Owner task |
|---|---|---|
| `crates/spillover-policy` | The policy: baseline (copied default scorer/picker), tier scorer, params, factory | P1, P2 |
| `crates/spillover-catalog` | Registration crate linked into Dynamo's Python bindings | scaffold |
| `crates/testkit` | Builds router selection inputs without a frontend | scaffold |
| `crates/proxy-core` | Proxy logic with no Dynamo runtime: request carrier, provider client, renderers, virtual cache, retokenizer, config | P3, P4a, P4b, P5, P9 |
| `crates/proxy-worker` | The worker binary: Dynamo runtime wiring | P6 |
| `crates/routing-sim` | Deterministic in-process simulation, scenarios, CI assertions | P7 |
| `sim/e2e`, `.github/workflows` | End-to-end simulation with real frontend and mock workers; CI | P8 |

## Phases

### Phase A: build and test everything against pinned upstream (now, parallel)

| Task | Scope | Done when |
|---|---|---|
| P1 | `params.rs` validation and tier lookup; `tier.rs` scorer | Unit tests cover every cost branch, missing load, validation errors |
| P2 | `baseline.rs`: copy upstream's default scorer, picker and parameters; equivalence test | For models without parameters, decisions equal `DefaultWorkerSelector` (seeded) across a fuzzed range of cache and load shapes |
| P3 | `orig.rs`, `errors.rs`, `upstream.rs` | Round-trip carrier tests; SSE client tested against a local mock server: chunks, `[DONE]`, 429 with Retry-After, 5xx, in-stream error, broken stream, cancel on drop |
| P4a | GLM (`glm45` + `glm47`) and Hermes (`qwen3` + `hermes`) renderers | Round-trip through `dynamo-parsers` for content, reasoning, one and several tool calls, streamed in small pieces |
| P4b | DeepSeek V4.1 renderer | Round-trip through `dynamo-parsers-v2` unified parser, same cases |
| P5 | `vcache.rs` | Hashes equal `dynamo_kv_router`'s for the same tokens and options; TTL, refresh, LRU cap, clear |
| P6 | `proxy-worker` binary | Registers a Tokens chat worker at a configured DP rank with a mirrored card; serves a request end to end against a mock provider; publishes KV events; answers health canary locally; maps provider errors to migratable errors |
| P7 | `routing-sim` | Scenarios below pass as `cargo test`; `routing-sim <scenario.yaml>` prints a report |
| P8 | `sim/e2e`, CI workflows | Scripts start frontend + mockers + proxies + fake provider; PR and nightly workflows defined |
| P9 | `retokenize.rs`, `config.rs` | Retokenized ids decode back to the exact text; stability under arbitrary chunking; config load/validate tests |

### Phase B: integrate (after Phase A merges)

- Wire proxy-core fully into proxy-worker; run the e2e simulation locally.
- Dockerfile for `dw-proxy-worker`; frontend image built with the catalog linked
  (`lib/bindings/python/Cargo.toml` optional dependency alias
  `dynamo-worker-selection-policy-catalog`, `--features custom-policy`).
- Policy YAML per deployment (`--router-policy-config`), `--router-track-active-blocks` on.
- When linked into Dynamo's build, our crates must resolve `dynamo-kv-router` to that build's
  `lib/kv-router`, not our pinned git copy, or the registry types don't match. `e2e-sim.yml`
  adds a `[patch]` to the checked-out Dynamo workspace; make that the documented build step.

### Phase C: ship (depends on bringing our Dynamo fork to upstream v1.5+)

- Merge or rebase `doublewordai/dynamo` onto upstream v1.5 or later (human-led; ~280 fork commits).
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

`crates/routing-sim` drives the real policy (`build_policy` with a seeded picker) through
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
- **Router signals**, as Dynamo's router computes them: overlap per worker from caches;
  load per worker from the router's own tracking (active requests, in-flight prompt tokens until
  first token, active decode blocks).
- **Retries**: a rate-limited proxy request is re-routed with that proxy excluded, as migration
  would.
- **Workload**: sessions with a shared system prompt, multi-turn conversations where each turn
  extends the previous prompt, think time between turns, output length distribution, and an
  arrival-rate profile per scenario.

Report (JSON and a markdown table), per time window and overall: share per worker class,
hosted occupancy, cache hit rate, stickiness (follow-up turns routed to the previous turn's
worker), spill order, failed requests.

Scenarios and assertions (`crates/routing-sim/scenarios/*.yaml`, run by `cargo test`):

| Scenario | Assertion |
|---|---|
| `low_load` (50% of hosted capacity) | hosted share >= 99% |
| `overload_ramp` (ramp to 200% and back) | proxy share <= 1% while hosted occupancy is under threshold; hosted occupancy >= 80% at peak; X share > Y share; traffic returns to hosted after the ramp |
| `stickiness` (long multi-turn sessions) | class stickiness >= 98% while the previous class is available; worker stickiness within 2 points of `DefaultWorkerSelector` on the same seed |
| `proxy_rate_limited` (X capped) | overflow beyond X goes to Y; no request fails |
| `no_parameters` | every decision equals `DefaultWorkerSelector` with the same seed |
| `hosted_outage` (hosted removed then restored) | proxies carry all traffic during the outage; hosted share recovers after restore |

### Level 2: end to end (nightly and on demand)

`sim/e2e`: a real Dynamo frontend built with our catalog, `python -m dynamo.mocker` hosted
workers (GPU-free simulated engines with real KV events), two `dw-proxy-worker` processes pointed
at a fake provider (streams canned output with configurable latency and 429s), and a multi-turn
load generator. File discovery, TCP request plane and ZMQ event plane, so no etcd or NATS.
Per-worker request counts come from worker logs and frontend metrics, and are compared with the
Level 1 prediction for the same scenario within tolerances. This is the only level that
exercises the real frontend, proxy registration and card matching.

### CI

- `ci.yml` on every PR: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` (includes
  Level 1 scenarios), build `dw-proxy-worker`, upload the sim report and write it to the job
  summary.
- `e2e-sim.yml` nightly and manual: build Dynamo's Python bindings at the pinned rev with the
  catalog linked (cached), run Level 2, upload the report.

## Upstream bumps

1. Change the pinned `rev` in `Cargo.toml`.
2. Re-copy `lib/router-plugins/builtin/src/default/{scorer,picker,parameters}.rs` into
   `spillover-policy/src/baseline/` and apply the same local edits.
3. `cargo test`: the equivalence test fails if the default changed and the copy did not.
4. Fix plugin API drift (compatibility shims run until 1.7).

## Phase B status: done

| Item | Result |
|---|---|
| Frontend with our catalog | `scripts/build-frontend.sh`: wheel at the pinned rev, `[patch]` in `lib/bindings/python/Cargo.toml` (its own workspace), frontend starts with `dw-spillover`. `docs/build-frontend.md`, `docker/frontend.Dockerfile`. |
| Proxy image | `docker/proxy-worker.Dockerfile`, `docs/images.md`. Not built here (Docker unavailable). |
| Level 2 end to end | `sim/e2e/run.sh` on the real frontend: 4 workers in one set, 0 failures, shares within tolerance of `config/level1-equivalent.yaml`. |
| Deployment config | `spillover-deploy` generates router-policy YAML and proxy configs from `deploy/deployments.yaml`. |
| Tuning | `routing-sim sweep`, `docs/tuning.md`: tier penalty is the main dial. |
| Retokenizer | Pre-token boundaries from the model tokenizer; ~99.9% of Chinese ids stream early. |
| Reasoning start | `render::reasoning_start` from `extra_args`; renderers correct with thinking on or off. |

## Remaining follow-ups

- A Chinese (or other boundary-free) run with no punctuation still waits for its end in the
  retokenizer; add a maximum hold length if that matters in practice.
- Neither renderer round-trips a second reasoning block mid-response.
- `sim/e2e/loadgen.py` stamps `dw.orig` itself until onwards does (Phase C).
- Build and push both images once Docker is available.

## Open questions

- Whether the occupancy estimate (router-tracked decode blocks) is close enough to real KV use;
  Level 2 compares it with mocker-reported usage.
- Per-model hosted capacity is a parameter until plugins can read `total_kv_blocks`.
- Name separator: `@` in Dynamo names avoids the `:` serving-class collision in onwards.
