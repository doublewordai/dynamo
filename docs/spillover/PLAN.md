<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spillover implementation plan

Design: https://claude.ai/artifact/2hQ78AEYMM6RMRUSNzPqME (version 4).

A Dynamo model is one worker set: our SGLang workers plus third-party proxy workers that
register as if they were SGLang. Nothing here depends on how models are named or split into
deployments; each served model name is configured on its own. A worker-selection policy (`dw-spillover`) ranks eligible workers by
cache affinity, then hosted-to-proxy failover, then proxy tier preference. The policy uses the
plugin API and the proxy is an ordinary worker, so Dynamo's routing logic is not changed. The
edits outside the new crates are small and additive:

- `lib/router-plugins/catalog` registers the policy.
- `lib/kv-router/src/plugins/worker_selection.rs` gains read-only accessors for values the
  default selector already computes (see Fork plugin API); `selector/` fills them only for
  custom policies.
- `lib/backend-common` gains an optional `WorkerConfig.router_config` (the proxy's per-set
  active-block tracking), and the Python bindings pass `None`.
- The frontend hands the client's chat request to workers that ask for it (see Chat request to
  proxy workers).

## Chat request to proxy workers

The frontend tokenizes before routing, so a worker normally receives token ids only. A proxy
needs the chat request (messages, tools, response format) to call a provider's chat API. A
worker asks for it with the runtime capability `chat_request` in its card's `runtime_data`
(`CHAT_REQUEST_CAPABILITY`, `lib/llm/src/local_model/runtime_config.rs`). `runtime_data` is not
part of the card checksum, so proxies still share a worker set with SGLang workers.

- The chat preprocessor keeps the normalized request in an `Arc` on the preprocessed request
  (`PreprocessedRequest::chat_request`, `#[serde(skip)]`), so it costs a pointer copy and is
  never serialized with the request.
- After the KV router picks a worker, `attach_chat_request`
  (`lib/llm/src/kv_router/routing_host/kv.rs`) serializes it into
  `extra_args.chat_request` if that worker advertises the capability, and drops
  `multi_modal_data` for it (the chat request already carries any media). Other workers get the
  request unchanged.
- Only the KV router attaches it; the proxy config already requires `router_config.mode: kv`.
  Only chat completions carry it; the proxy registers `endpoint_types: chat` only, so the
  frontend never builds a `/v1/completions` pipeline that can route to it. The hosted SGLang
  workers must be launched with `--endpoint-types chat` as well: `endpoint_types` feeds the
  card's `model_type`, which is part of `worker_set_key`, so a mixed `chat,completions`
  hosted set would no longer share a worker set with the chat-only proxy.
- The frontend writes the field, so a client cannot supply or forge it.

This is Doubleword's Dynamo fork, not the standalone spillover repo the design was first
prototyped in. The work now lives under `lib/` and `docs/spillover/`, and the fork's own
Python build links the policy.

## Repository layout

| Path | What | Owner task |
|---|---|---|
| `lib/router-plugins/spillover` | The policy: baseline (port of the fork's default scorer/picker), tier scorer, params, factory | F1 |
| `lib/router-plugins/catalog` | Registration crate linked into Dynamo's Python bindings | scaffold |
| `lib/spillover/testkit` | Builds router selection inputs without a frontend | scaffold |
| `lib/spillover/proxy-core` | Proxy logic with no Dynamo runtime: chat-request reader, provider client, renderers, virtual cache, retokenizer, config | F2, F3 |
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
public plugin inputs. Four inputs the default computed privately are exposed by small,
additive, read-only accessors in `lib/kv-router/src/plugins/worker_selection.rs`, populated only
on the custom-policy path so default routing is untouched: the batch-wide
`min_active_prefill_tokens` and `has_tier_overlap_blocks` (from the default's own
`DefaultScoringContext`), `WorkerCacheInput::effective_overlap_blocks`,
`WorkerLoadInput::raw_prefill_blocks`, and the request's effective weights after
`router_config_override`. With these the port reproduces the default's logit, cached tokens and
decode blocks, not just its choice, and `tests/equivalence.rs` checks that across cache and load
shapes, busy prefill pools, requests without a tier-overlap map, non-block-aligned prompts,
per-request weight overrides and session affinity.

The baseline approach is:

- A model **with no entry** in the policy parameters gets `WorkerSelectionPolicy::default(
  config.clone(), role.default_selector_label())` — exactly Dynamo's default. Its decisions
  equal `DefaultWorkerSelector`'s for a fuzzed grid of cache/load shapes and temperatures.
- A model **with parameters** gets scorers `[baseline, TierScorer]` and the baseline picker.
  A parameter set whose tiers match no worker and whose failover never triggers picks what the
  default picks, with the same logit, across the equivalence grid (see Fork plugin API).

`build_policy(config, role, model_name, params, rng)` is the seam `routing-sim` drives.

The equivalence tests live in `lib/router-plugins/spillover/tests/equivalence.rs`.

## Phases

### Phase A: port and test inside the fork (now, parallel)

| Task | Scope | Done when |
|---|---|---|
| F1 | `tier.rs` per-candidate costs; `policy.rs` baseline/default split; `baseline/` per-candidate port; equivalence and tier tests | For models without parameters, decisions equal `DefaultWorkerSelector` (seeded); for parametered models with no matching tier, likewise |
| F2 | `chat_request.rs`, `errors.rs`, `upstream.rs`, proxy worker binary | Carrier round-trips; SSE client against a mock server (chunks, `[DONE]`, 429 + Retry-After, 5xx, in-stream error, broken stream, cancel on drop); worker registers and serves end to end |
| F3 | GLM/Hermes and DeepSeek renderers, `vcache.rs`, `retokenize.rs`, `config.rs` | Round-trips through `dynamo-parsers`/`dynamo-parsers-v2`; virtual-cache hashes equal `dynamo-kv-router`'s; retokenized ids decode to the exact text |
| F5 | `routing-sim` | Scenarios below pass as `cargo test`; `routing-sim <scenario.yaml>` prints a report |
| F4 | `deploy`, `e2e`, docs, `.github/workflows/spillover.yml` | Generator tests pass; `run.sh` validated with `bash -n`; workflow runs tests, clippy, `dw-proxy-worker` build and routing-sim scenarios into the summary |

### Phase B: integrate

- Wire proxy-core fully into proxy-worker; run the e2e simulation locally.
- Dockerfile for `dw-proxy-worker` (the fork's `lib/spillover/proxy-worker/Dockerfile`); a
  frontend image built with the catalog linked.
- Policy YAML per deployment (`--router-policy-config`), `--router-track-active-blocks` on
  for each spillover worker set (see Active-block tracking).
- Because the crates build inside the fork's workspace, they resolve `dynamo-kv-router` to the
  fork's `lib/kv-router`, so registry types match by construction; no `[patch]` is needed.

### Phase C: ship

- Deployment (internal): proxy chart and secrets.
- Rollout: staging with a proxy-only model; then one production model; then the rest.

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
| `admission_margin_low` / `admission_margin_high` (same workload, margin below vs above the policy's failover point) | the high margin keeps strictly more cached conversations on hosted and records no steering exclusions; the low margin steers and records none of the 529s |

## Admission margin

The engine-queue admission margin is a **backend** feature (same fork commit as the spillover
work), not a router-policy one. This section records what the fork actually does, because the
G2 brief described a frontend gate with a per-model override that does not exist here.

- **Where it is read.** `DYN_ADMISSION_QUEUE_MARGIN` is read once per **worker process** in
  `BackendAdmissionGate::from_environment` (`lib/runtime/src/admission_gate.rs`, parsed in
  `lib/runtime/src/admission_margin.rs`). There is no frontend gate and **no
  `DYN_ADMISSION_QUEUE_MARGIN_OVERRIDES`**: margins are not per model, they are per hosted
  process, so a deployment sets the variable in each hosted worker's environment.
- **What it bounds.** The engine's own waiting queue, not Dynamo's. The estimate is the engine's
  last reported waiting count (summed over DP ranks) plus every admission since that report that
  has not yet left the queue. A counted admission is returned at its first response item, at
  stream end, or on a dispatch failure; a complete report resets the count. A process whose
  engine never reported is unenforced, which is why `dw-proxy-worker` is never subject to it.
- **At or over the margin.** The worker is excluded from selection, so cached conversations are
  pushed to a proxy tier until its queue drains. If no strictly-lower-priority in-flight request
  can be evicted and no other worker can take the request, the arrival is refused as
  `ErrorType::WorkerOverloaded`; Dynamo's frontend maps a pre-stream overload to HTTP 529
  (`DYN_HTTP_OVERLOAD_STATUS_CODE`, default 529). A strictly-lower-priority victim is evicted
  instead, its stream ending with `ErrorType::ResourceExhausted`.
- **Frontend steering is separate.** The frontend marks workers overloaded from load thresholds
  (`worker_monitor.rs`) and from a bounded request-path overload lease (`push_router.rs`); the
  router excludes them via `RoutingEligibility`, and "all eligible workers overloaded" is
  `KvSchedulerError::AllEligibleWorkersOverloaded` -> 429 in the kv-router, surfaced as 529 by
  the frontend. The admission margin is a distinct, worker-side hard bound that must sit **above
  the policy's own failover point** so the router's occupancy-driven spill happens first.

The simulation models the gate in `routing-sim` (`admission.hosted_queue_margin` plus per-worker
`hosted_queue_margin_overrides`, applied before the policy selects; a hosted worker with a queue
at or above its margin is added to the excluded set, and a refusal with no remaining candidate is
reported as a 529). `spillover-deploy` writes the margin as environment files
(`admission/<model>/hosted.env` and `.../proxy.env`) because that is where the fork reads it, and
`lib/spillover/e2e/run.sh` sets it on the hosted workers and unsets it for the proxies. The
`admission_queue_margin` sweep and the two comparison scenarios live in
`docs/spillover/tuning.md` and `lib/spillover/routing-sim/scenarios/`.

## Active-block tracking

`dw-spillover` estimates a hosted worker's occupancy as router-tracked decode blocks over the
policy's `hosted_capacity_blocks`. The router only counts those blocks when
`KvRouterConfig::router_track_active_blocks` is true
(`lib/kv-router/src/scheduling/config.rs`). The fork's frontend default for that flag is on,
but a deployment that starts its frontend with `--no-router-track-active-blocks` reports zero
occupancy and the failover penalty would never fire. Tracking is therefore enabled **per worker
set** rather than trusted from the frontend global config.

**Per-set override exists and is honoured, but only the SGLang side can use it.** The watcher
builds each worker set's KV router from the model card's `router_config` when present,
otherwise from the frontend's global config
(`effective_router_config`, `lib/llm/src/discovery/watcher.rs:1408`). That helper clones the
card config and overrides only `router_prefill_policy`, `router_decode_policy` and
`session_affinity_mode`; `router_track_active_blocks` and `router_track_output_blocks` come
from the card. The effective config is passed to
`kv_chooser_for_with_plugins_and_client(..., Some(router_config.kv_router_config.clone()), ...)`
(`lib/llm/src/discovery/watcher.rs:617`) and reaches the policy factory as `&KvRouterConfig` in
`lib/kv-router/src/services/selection/core/workers.rs:321`. The card checksum includes
`router_config` (`lib/llm/src/model_card.rs:1297`), so hosted and proxy workers must advertise
identical values or they stop forming one worker set.

SGLang workers can advertise a card `router_config`: `components/src/dynamo/sglang/args.py`
parses `--router-*` into `WorkerRouterConfig` via `parse_worker_router_config`, and
`components/src/dynamo/sglang/register.py` builds it with `build_router_config` and passes it to
`register_model(router_config=...)`. The Rust `dw-proxy-worker` cannot: it registers through
`dynamo_backend_common` (`lib/backend-common/src/worker.rs` `build_local_model`), whose
`EngineConfig`/`WorkerConfig` have no card `router_config` field; only the Python bindings
(`lib/bindings/python/rust/llm/entrypoint.rs`) can set one. A proxy that omitted the flag while
hosted workers set it would also change the card checksum and split the worker set.

We therefore enable tracking **per worker set**. Each deployment advertises
`router_track_active_blocks` (and the mode the policy needs) on its model card instead of on
the frontend:

- The Rust `dw-proxy-worker` now registers through `dynamo_backend_common`, whose
  `WorkerConfig` gained an optional `router_config`
  (`lib/backend-common/src/worker.rs`, applied to the `LocalModel` in `build_local_model`).
  `None`, the default, keeps the old behaviour: the card advertises no `router_config` and the
  worker set inherits the frontend's. The field is the same `RouterConfig` the Python
  bindings accept in `register_model`.
- `ProxyConfig` gained an optional `router_config` (mode plus the tracking flags), and
  `registration.rs` turns it into that `RouterConfig`, so the proxy card carries exactly what
  the SGLang workers advertise. It pins `shared_cache_multiplier` to the SGLang CLI default
  `0.5` unconditionally (`DYN_SHARED_CACHE_MULTIPLIER` must not decide one side of the set),
  because the field is serialized into the card and a mismatch would split the set; a test
  asserts the serialized configs and checksums are equal.
- The proxy also enables its local KV indexer, so the worker advertises a recovery target. A
  live-only proxy cannot re-sync after a frontend restart: a `Stored` that extends a prefix it
  still holds has a `parent_hash` the fresh frontend never indexed, and the router drops the
  chain. With the indexer the fresh frontend pulls the held tree instead.
- SGLang workers use the existing per-set path: `--router-*` args through
  `parse_worker_router_config`/`build_router_config`, exactly as before.

`spillover-deploy` emits both halves from one value: `router/<model>/hosted.args` holds the
SGLang `--router-*` flags for the hosted workers, and every proxy YAML gets the same
`router_config`. The card checksum includes `router_config`
(`lib/llm/src/model_card.rs:1297`), and a test builds a hosted card and a proxy card from the
same advertisement and asserts equal checksums, so the set cannot accidentally split.
It no longer emits a frontend-wide `DYN_ROUTER_TRACK_ACTIVE_BLOCKS`; the note in the generated
`frontend.env` records that a frontend-wide flag also works but changes tracking for every
other model on the frontend. The e2e simulation starts the mocker with the same `--router-*`
flags, gives the proxies the same `router_config`, and runs the frontend without the global
flag.

Either way the policy fails loudly if tracking is off: `build_policy`
(`lib/router-plugins/spillover/src/policy.rs`) logs an error naming the model and
`router_track_active_blocks`, points first at the per-set setting, and returns Dynamo's default
policy for that model, so a misconfigured deployment never silently claims to spill while
failover is dead. The behavior is asserted in
`lib/router-plugins/spillover/tests/equivalence.rs`.

A model key that is absent from a non-empty `models` map also falls back to Dynamo's default,
but now logs a warning naming the partition and the configured keys, so a typo or a
served-model alias cannot silently disable spillover. Installing a parametered model's tier
policy logs one `info!` per partition naming the model, tiers and thresholds; per-candidate
costs are logged at `debug`.

### Level 2: end to end (nightly and on demand)

`lib/spillover/e2e`: a real Dynamo frontend from *this fork's* Python build, `python -m
dynamo.mocker` hosted workers (GPU-free simulated engines with real KV events), two
`dw-proxy-worker` processes pointed at a fake provider (streams canned output with configurable
latency and 429s), and a multi-turn load generator. File discovery, TCP request plane and ZMQ
event plane, so no etcd or NATS. Per-worker request counts come from the load generator's
per-turn records and the fake-provider logs, and are compared with the Level 1 prediction for
the same scenario within tolerances.
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
  `lib/router-plugins/spillover/**`, `lib/router-plugins/catalog/**` or `lib/llm/**` (the
  chat-request integration): `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test`
  for the spillover crates, build `dw-proxy-worker`, run `routing-sim` scenarios, and write
  each scenario's markdown to the job summary.
- `.github/workflows/spillover-nightly.yml` (nightly cron and `workflow_dispatch`) builds the
  fork's Python frontend and runs `lib/spillover/e2e/run.sh` end to end. `run.sh` now passes
  `report.py --require-routing` with `--min-proxy-share`/`--require-tier`, so a run where every
  request succeeds but none reaches a proxy fails even without a Level 1 baseline.

## Phase B status

| Item | Result |
|---|---|
| Frontend with our catalog | Fork Python build (`maturin develop` + `pip install -e .`) links `lib/router-plugins/catalog`; `custom-policy` is a default feature; the frontend starts with `dw-spillover` selectable. |
| Proxy image | `lib/spillover/proxy-worker/Dockerfile`; see `docs/spillover/images.md`. Not built here (Docker unavailable). |
| Level 2 end to end | `lib/spillover/e2e/run.sh` on the real frontend: 4 workers in one set, 0 failures, served-by tags on every proxy response, both proxy metric surfaces live, absolute routing checks (proxy share floor, required tiers) enabled. The Level 1 comparison passes every row with the recalibrated twin (see [Level 2 run](#level-2-run)); `spillover-nightly.yml` runs the harness without a baseline so a routing regression is still caught on its own. |
| Deployment config | `spillover-deploy` generates router-policy YAML and proxy configs from `lib/spillover/deploy/config/deployments.yaml`. |
| Tuning | `routing-sim sweep`, `docs/spillover/tuning.md`: `failover_penalty_blocks` is the main spill/stickiness dial; tier penalty is the preference dial. |
| Retokenizer | Pre-token boundaries from the model tokenizer; ~99.9% of Chinese ids stream early. |
| Reasoning start | `render::reasoning_start` from `extra_args`; renderers correct with thinking on or off. |

## Validation

Final end-to-end validation of the Level 2 harness (G5, 2026-09-29). All commands
run from a checkout of `main` at `448fd98e2` with a fresh venv and
`CARGO_TARGET_DIR=/home/peter/.cache/dw-fork-target`.

### Build the fork's Python package

```bash
uv venv /home/peter/.cache/dw-fork-venv
source /home/peter/.cache/dw-fork-venv/bin/activate
uv pip install pip 'maturin[patchelf]'
cd lib/bindings/python && maturin develop --uv && cd -
uv pip install -e . -e lib/gpu_memory_service
python3 -m dynamo.frontend --help   # exit 0
python3 -m dynamo.mocker --help     # exit 0, lists --router-track-active-blocks
```

The build uses default features, so `custom-policy` is on and
`lib/router-plugins/catalog` is linked. `dw-spillover` is present in the built
`_core.abi3.so` and the catalog registers it on `import dynamo._core`; the run's
generated `router-policy.yaml` then selects it by name.

### Generate the run configs

`run.sh` renders `lib/spillover/e2e/config/deployments.yaml` with the run's
model id, capacities, block sizes and provider ports, then calls
`spillover-deploy generate --out out/run/generated`. The generated tree is what
the frontend (`router-policy.yaml`), proxies (`<tier>-0.yaml`) and mockers
(`router/<model>/hosted.args`, `admission/<model>/hosted.env`) consume. It is
not hand-written, so this run exercises the same generator production uses.

### Level 1 baseline

```bash
routing-sim lib/spillover/e2e/config/level1-equivalent.yaml \
  --json /tmp/l1.json --markdown /tmp/l1.md
```

778 requests, 0 failures; hosted 392 (50.4%), proxy-x 381 (49.0%), proxy-y 5
(0.6%); class stickiness 69.8%.

### Level 2 run

```bash
export PATH=/home/peter/.cache/dw-fork-venv/bin:$PATH
export CARGO_TARGET_DIR=/home/peter/.cache/dw-fork-target
BASELINE=/tmp/l1.json lib/spillover/e2e/run.sh
```

700 requests, 0 failures, 4 workers committed to one WorkerSet (0 `Rejected incompatible
workers`). The proxies received the chat request from the frontend's KV router (the load
generator sends plain chat requests) and served 333 requests.

| metric | Level 2 | Level 1 | delta | band | result |
|---|---:|---:|---:|---:|---|
| failed | 0 | 0 | 0 | — | pass |
| hosted share | 52.4% | 50.4% | +0.020 | 0.050 | pass |
| proxy-x share | 46.0% | 49.0% | -0.030 | 0.049 | pass |
| proxy-y share | 1.6% | 0.6% | +0.009 | 0.02 | pass |
| class stickiness | 64.4% | 69.8% | -0.054 | 0.070 | pass |
| served-by tags | 333 tagged, 0 untagged, 0 mismatched tier | — | — | — | pass |

`run.sh` now passes every comparison row (exit 0). The real stack is unchanged from the
previous run (52.6% hosted). The review fixes that count each arrival's own uncached blocks and
use per-rank capacity dropped the uncalibrated twin to 33.9% hosted; refitting the hosted timing
in `level1-equivalent.yaml` to the mocker's own model (see `CALIBRATION.md`) brings it back to
50.4% hosted, so all four share/stickiness rows are inside the band at the original 0.1
tolerance.

Every proxy response carried `nvext.engine_data {served_by, tier}` and the tier
matched the worker's DP rank. Both `dynamo_component_proxy_requests_total` series
carried the correct `provider` and `tier` labels. The spill ramps with the
arrival profile: hosted share falls from 84% in the first window to ~37% at the
peak and proxy-x absorbs it. The committed set has DP ranks 0, 0, 1000, 2000
(two hosted, one per proxy tier).

### Rust tests

```bash
cargo test -p dw-spillover-policy -p dw-proxy-core -p dw-proxy-worker \
  -p dw-routing-sim -p dw-spillover-deploy -p dw-spillover-testkit \
  -p dynamo-worker-selection-policy-catalog
```

182 passed, 0 failed.

### Coverage and gaps

The run exercises the whole path: the fork's frontend build loading the catalog,
model-card checksum matching, proxy registration and worker-set membership, the
two-tier policy choosing a tier, provider streaming, `engine_data` served-by
stamping and the proxy Prometheus surface. It does not exercise real SGLang/vLLM
engines, disaggregation, RDMA/NIXL, or multi-node placement; the fake provider
and GPU-free mocker stand in for the engine. The second tier carries little
traffic because the providers answer in ~230 ms and the tier penalty band makes
proxy-x the near-threshold choice; `proxy-y` is only reached under burst
concurrency. A hub-id run needs a seeded `HF_HUB_CACHE` or network access. The
nightly workflow (`.github/workflows/spillover-nightly.yml`) now runs this level.

### Defects found and fixed while validating

- `lib/bindings/python/rust/backend.rs`: the `WorkerConfig` literal was missing
the new `router_config` field, so `dynamo-py3` did not compile with default
features. Added `router_config: None` (behavior-preserving: that constructor
never set it; the model-card router config is set on the `register_model`
path).
- `lib/spillover/e2e/run.sh`: the runtime's global `DYN_SYSTEM_PORT=-1` made the
hosted mockers fall back to shared-storage metadata while the proxies self-hosted,
so the two sides advertised different card `extra_files` and the frontend split
the WorkerSet. Each worker process now gets its own system port
(`HOSTED_SYSTEM_PORT` for mockers, `PROXY_*_SYSTEM_PORT` for proxies), and
hosted mockers run one process per worker because a single mocker process cannot
bind one port for several runtime instances. The defaults keep the hosted range
clear of the proxy ports.

## Remaining follow-ups

- The retokenizer holds at most `MAX_HELD_BYTES` (4 KiB) waiting for a pre-token boundary;
  a longer boundary-free run is cut there, so its ids can differ from a one-shot encode at
  that cut. Tokenizers with `add_prefix_space: true` are not supported (none of the pinned
  families use it).
- Neither renderer round-trips a second reasoning block mid-response.
- Build and push both images once Docker is available.

## Open questions

- Whether the occupancy estimate (router-tracked decode blocks) is close enough to real KV use;
  Level 2 compares it with mocker-reported usage.
- Per-model hosted capacity is a parameter until plugins can read `total_kv_blocks`.
