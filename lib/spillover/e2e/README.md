# Level 2 end-to-end simulation

Starts a real Dynamo frontend (built with our `dw-spillover` catalog), GPU-free
mocker primary workers, two `dw-proxy-worker` processes pointed at fake
OpenAI-compatible providers, and a multi-turn load generator. It is the only
level that exercises real discovery, request routing, proxy registration and
model-card matching.

File discovery plus the TCP request plane and ZMQ event plane keep everything on
one machine, so no etcd or NATS is needed. All servers bind `127.0.0.1`.

See `docs/spillover/PLAN.md` "Simulation" (Level 2) for the model and the CI section for
how the workflows use these scripts.

## Files

| Path | Purpose |
|---|---|
| `fake_provider.py` | OpenAI-compatible SSE provider with configurable TTFT, tokens/s, output length, 429s, errors and JSONL request logging |
| `loadgen.py` | Multi-turn sessions, shared system prompt, think time, arrival-rate profile, streams and records the serving worker and served-by tag |
| `run.sh` | Starts providers, frontend, mocker workers and proxies; runs loadgen, scrapes proxy metrics and the report; cleans up |
| `report.py` | Per-worker/per-tier shares over time, stickiness, served-by tags, errors; markdown, optional Level 1 comparison |
| `check_metrics.py` | Validates the proxy Prometheus snapshots `run.sh` scraped (`dynamo_component_proxy_*`, tier/provider labels) |
| `run_helpers.py` | Literal `${NAME}` template rendering, generator-matching model-dir sanitization and the port pre-flight check `run.sh` uses |
| `test_e2e_helpers.py` | Standard-library unit tests for the scripts above (`python3 -m unittest discover -s lib/spillover/e2e -p 'test_*.py'`) |
| `config/deployments.yaml` | Deployment description `spillover-deploy` generates the policy and proxy configs from (tier ranks 1000/2000 via the generator's rank rule) |
| `config/level1-equivalent.yaml` | `routing-sim` scenario twin of the e2e run for `report.py --baseline` |
| `config/arrival-profile.json` | Piecewise-linear arrival rate used by `run.sh` |
| `config/tier-map.json` | Example DP-rank ranges for manual `report.py --tier-map` runs; `run.sh` derives the ranges from the generated proxy configs instead |
| `requirements.txt` | No dependencies (standard library only); kept as a pip-installable no-op |

## Prerequisites

- A Dynamo checkout with the Python bindings built. This fork builds the
  frontend from the checked-out tree: `maturin develop` in `lib/bindings/python`
  plus the `ai-dynamo` package (`pip install -e .`), as in the contribution
  guide. `custom-policy` is a default feature and `lib/router-plugins/catalog`
  registers `dw-spillover`, so the fork's normal Python build already includes
  the policy. `run.sh` invokes `python3 -m dynamo.frontend`, so put that venv's
  `bin/` on `PATH`.
- A built `dw-proxy-worker` and `spillover-deploy` binary; `run.sh` builds both
  into `$CARGO_TARGET_DIR/debug/` unless `SKIP_BUILD=1`.
- `curl` for the proxy metrics scrape (`run.sh` uses it to read `/metrics`).
- No extra Python packages: the scripts use the standard library, so
  `python3 -m pip install -r lib/spillover/e2e/requirements.txt` is an optional
  no-op.
- A tokenizer for the model: `tokenizer.json`, `config.json`,
  `tokenizer_config.json` (and `generation_config.json`). Download
  `Qwen/Qwen3-0.6B` once; for a fully offline run point `MODEL_PATH` at a local
  directory containing those files.

## Run the whole stack

With network access:

```bash
# from the repository root, with the Dynamo Python package importable
lib/spillover/e2e/run.sh
```

Fully offline, using a locally downloaded tokenizer directory and a venv built
from this fork (see the contribution guide's Python dev build):

```bash
export PATH=/path/to/venv/bin:$PATH
MODEL_PATH=/path/to/Qwen3-0.6B \
CARGO_TARGET_DIR=/path/to/cargo-target \
SKIP_BUILD=1 \
lib/spillover/e2e/run.sh
```

`MODEL_PATH` being a directory makes `run.sh` seed the offline HF cache under
`MODEL_ID` (`Qwen/Qwen3-0.6B` by default) and register the workers with that hub
id, because the frontend requires `--model-path` to be a real directory while
workers must advertise a matching `source_path`. `spillover-deploy`, the same generator production deployments use, emits the
run's policy and proxy configs: `run.sh` renders `config/deployments.yaml` with
the run's model id, capacities, block sizes and provider ports, generates into
`out/run/generated/`, and passes the generated `router-policy.yaml`, proxy
configs and primary `--router-*` flags to the frontend, proxies and mockers. A
generated `admission/<model>/primary.env` supplies `DYN_ADMISSION_QUEUE_MARGIN`
and the generated `proxy.env`'s `unset` is what the proxies opt out with. The
default scenario (`PRIMARY_BLOCKS=8`, `SPEEDUP=1`) ramps past the primary capacity
and spills a large share to proxy X.

Outputs land in `lib/spillover/e2e/out/`:

- `logs/` — one log per process; the first place to look when routing looks wrong.
- `reports/loadgen.jsonl` — one JSON object per turn (arrival/start/end times,
  latency, TTFT, status, worker id, DP ranks, usage).
- `reports/provider-x.jsonl`, `provider-y.jsonl` — one line per provider request
  (provider name, prompt size, status, latency).
- `reports/metrics-proxy-x.prom`, `metrics-proxy-y.prom` — timestamped snapshots
  of each proxy's Prometheus endpoint, scraped while load runs.
- `reports/metrics.json` — the `check_metrics.py` verdict on those snapshots.
- `run/generated/` — the `spillover-deploy` output the run used (policy, proxy
  configs, primary `primary.args`, admission env files).
- `reports/e2e-report.md` and `e2e-report.json` — the report described below.

To compare with Level 1, build a `routing-sim` JSON report and pass it as
`BASELINE`:

```bash
# the Level 1 twin must use the same primary capacity, block size, tiers and
# arrival profile as the run (see config/level1-equivalent.yaml)
routing-sim lib/spillover/e2e/config/level1-equivalent.yaml --json /tmp/l1.json
BASELINE=/tmp/l1.json TOLERANCE=0.1 lib/spillover/e2e/run.sh
```

### Environment overrides

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen3-0.6B` | HF id or local dir; a local dir is used by the frontend and seeded into the offline cache |
| `MODEL_ID` | `Qwen/Qwen3-0.6B` | Hub id workers register as `source_path` (must match across mockers and proxies) |
| `FRONTEND_MODEL_PATH` | `$MODEL_PATH` | Directory the frontend loads directly; set separately if `MODEL_PATH` is a hub id |
| `MODEL` | `$MODEL_PATH` | Served model name; must match the model key in the generated `router-policy.yaml` |
| `HF_HUB_CACHE` | `$OUT_DIR/hf-cache` | Offline HF cache `LocalModel::fetch` resolves hub ids from |
| `HF_HUB_OFFLINE` | `1` with a local `MODEL_PATH` | Prevents model resolution from touching the network; a hub-id `MODEL_PATH` leaves it off to allow the one download |
| `FRONTEND_PORT` | `8000` | Frontend HTTP port |
| `PROVIDER_X_PORT` / `PROVIDER_Y_PORT` | `9101` / `9102` | Fake provider ports (also generated into the proxy configs) |
| `PROXY_X_SYSTEM_PORT` / `PROXY_Y_SYSTEM_PORT` | `9211` / `9212` | Proxy metrics/health ports, re-enabled per proxy for scraping; kept clear of the primary range |
| `PRIMARY_SYSTEM_PORT` | `9200` | First primary mocker metrics/health port; worker `i` uses `PRIMARY_SYSTEM_PORT + i`. A system port also makes the worker self-host its model card (see Model-card matching) |
| `METRICS_INTERVAL` | `2` | Seconds between proxy metrics scrapes |
| `DEPLOY_BIN` | `$CARGO_TARGET_DIR/debug/spillover-deploy` | Config generator binary |
| `PRIMARY_WORKERS` | `2` | Number of primary mocker processes, one worker each |
| `PRIMARY_BLOCKS` | `8` | mocker KV blocks, advertised as each worker's `total_kv_blocks` (and echoed as the policy's `primary_capacity_blocks` fallback); small enough that the ramp spills |
| `PRIMARY_QUEUE_MARGIN` | `256` | `DYN_ADMISSION_QUEUE_MARGIN` set on the primary workers; must be above the policy's failover point. Proxies are launched with it unset |
| `BLOCK_SIZE` | `64` | KV block size; must match the proxy configs |
| `CONTEXT_LENGTH` | `32768` | mocker `--max-model-len` and proxy `context_length`; the cohort checksum includes it |
| `ENGINE_TYPE` | `vllm` | Mocker engine; `vllm` accepts `--max-model-len`, `sglang` does not |
| `MAX_SEQS` | `64` | mocker concurrency, advertised as each worker's `max_num_seqs` |
| `SPEEDUP` | `1.0` | mocker `--speedup-ratio`; 1.0 keeps primary workers slow enough to spill |
| `SESSIONS`, `TURNS`, `THINK_TIME` | `200`, `4`, `0.5` | Load shape; sessions are capped by `--duration` |
| `MAX_TOKENS` | `32` | Output length asked of the provider |
| `ARRIVAL_RATE`, `DURATION` | `2.0`, `62` | Arrival rate (sessions/s) and scheduling window; `config/arrival-profile.json` overrides the rate |
| `PROVIDER_X_CONCURRENCY` / `PROVIDER_Y_CONCURRENCY` | `8` | Requests in flight before 429 |
| `PROVIDER_X_TTFT_MS` / `PROVIDER_Y_TTFT_MS` | `30` | Fake-provider time to first token (ms) |
| `PROVIDER_X_TPS` / `PROVIDER_Y_TPS` | `200` | Fake-provider decode tokens/s |
| `PROVIDER_X_ERROR_RATE` / `PROVIDER_Y_ERROR_RATE` | `0.0` | Fraction answered with HTTP 503 |
| `DYN_FAKE_API_KEY` | `fake-key` | Bearer token the generated proxy configs require (`api_key_env`) |
| `BIN_SECONDS` | `10` | Time-window width in the report |
| `WORKER_WAIT` | `600` | Seconds to wait for the model to register |
| `OUT_DIR` | `lib/spillover/e2e/out` | Where logs and reports go |
| `SKIP_BUILD` | `0` | Set to `1` to reuse an existing proxy binary |
| `CARGO_TARGET_DIR` | `target/` | Build directory for the proxy worker |
| `PROXY_BIN` | `$CARGO_TARGET_DIR/debug/dw-proxy-worker` | Proxy binary to run |
| `BASELINE`, `TOLERANCE` | empty, `0.1` | Optional Level 1 report and relative tolerance |
| `REQUIRE_ROUTING` | `1` | Fail on a failed request, an untagged/mis-tiered proxy response, a proxy share below the floor, or a required tier never seen |
| `MIN_PROXY_SHARE` | `0.05` | Floor for the combined proxy share under `REQUIRE_ROUTING` |
| `MAX_PRIMARY_SHARE` | empty | Optional upper bound on the primary share |
| `REQUIRE_TIERS` | `proxy-x` | Space-separated tiers that must each serve a request (`spillover-nightly.yml` requires both) |

## Run the scripts individually

`fake_provider.py` and `loadgen.py` are independent and make a fast smoke test
with no Dynamo:

```bash
python3 lib/spillover/e2e/fake_provider.py --port 9111 --name smoke \
  --ttft-ms 20 --tps 100 --max-tokens 16 \
  --log /tmp/fake.jsonl &
python3 lib/spillover/e2e/loadgen.py --url http://127.0.0.1:9111 --model smoke \
  --sessions 3 --turns 3 --arrival-rate 3 --duration 10 \
  --out /tmp/loadgen.jsonl --summary /tmp/summary.json
python3 lib/spillover/e2e/report.py --loadgen /tmp/loadgen.jsonl \
  --provider-log smoke=/tmp/fake.jsonl
```

`report.py` also takes `--json`, `--markdown`, `--bin-seconds` and
`--tier-map`. With `--require-routing` (and therefore `--tier-map`) it also
asserts where the traffic went: `--min-proxy-share`, `--max-primary-share` and
`--require-tier`. `run.sh` always enables these; without a tier map a rank is
reported as `unknown` rather than guessed as primary.

## Reading the report

- **Per worker class (overall)** — requests, share, failures and latency/TTFT
  percentiles for `primary`, `proxy-x`, `proxy-y` (and `unknown` if a response
  had no `nvext.worker_id`).
- **Shares over time** — the same shares per `--bin-seconds` window, so a spill
  ramp and recovery are visible.
- **Stickiness** — fraction of follow-up turns served by the same worker (and,
  separately, the same class) as the previous turn, taken from `nvext.worker_id`
  on streamed chunks. Class stickiness is the metric Level 1 reports.
- **Fake providers** — requests per provider with HTTP status counts and mean
  prompt size, which shows 429s from a capped proxy.
- **Served-by tags** — proxy responses that carried `nvext.engine_data`
  `{served_by, tier}`; untagged proxy responses or a tag that disagrees with the
  worker's DP rank make the report exit non-zero.
- **Comparison with Level 1** (only with `--baseline`) — e2e vs `routing-sim`
  values for class shares, class stickiness and failures, with a `pass`/`FAIL`
  per metric. `FAIL` makes `report.py` exit non-zero, but `run.sh` still prints
  the report and keeps the artifacts. The Level 1 JSON is matched by metric name
  (`classes.<tier>.share`, `class_stickiness.rate`, `failed_requests`); a tier
  absent from the Level 1 spill order counts as zero. The band is
  `tolerance * |level 1|`, with a 2-percentage-point absolute floor so a share of
  a few percent is not judged relative to a near-zero reference.

### Making the Level 1 twin equivalent

The two levels run the same policy, capacity, tiers and arrival profile, and
`config/level1-equivalent.yaml` is fitted to the e2e run. **The twin is calibrated:**
the measured run is 52.4% primary / 46.0% proxy-x / 1.6% proxy-y with 64.4% class
stickiness, and the twin reports 50.4% / 49.0% / 0.6% with 69.8% stickiness, inside the
`report.py --baseline` band on every row (primary +0.020 within 0.050, proxy-x -0.030
within 0.049, proxy-y +0.009 within 0.02, stickiness -0.054 within 0.070).
`docs/spillover/PLAN.md`'s Level 2 table records all four rows as `pass`; each primary
parameter's derivation from the mocker's timing model is in the twin's header comment. The
residual deltas are the modelling gap below plus run-to-run noise, not a routing bug.
Getting the shapes to line up needs two deliberate choices:

- **Backend speed.** `routing-sim` needs explicit token rates; the mocker does
  not expose an equivalent, so the twin's primary `prefill_tokens_per_second`,
  `decode_tokens_per_second` and `batching_slowdown` are derived from the
  mocker's `aisimulate-core` polynomial timing model (unloaded decode
  `1 / 5.74 ms` = 174 tps; prefill ~5000 tps for this workload's turn sizes;
  `batching_slowdown` 1.3 linearising the quadratic utilisation term) and then
  adjusted within that model to match the measured shares. The proxies keep the
  fake provider's real 200 tps. Without this fit the sim's primary workers are far
  faster than the mockers and spill very differently.
- **Load signal.** Level 1 charges a proxy for its in-flight prefill/decode
  blocks; the frontend cannot observe a `dw-proxy-worker`'s scheduler load
  (there is no `LLMEngine` load hook; load arrives over the runtime's own
  metrics plane, which SGLang/mocker implement), so it scores proxies as idle.
  The fit above keeps the twin's proxy X busy only as much as the real one, so
  both prefer X over Y at the same rate. The residual deltas are this modelling
  gap plus run-to-run noise, not a routing bug: the tier preference, ramp shape,
  stickiness and zero failures all agree.

## How the worker is identified

`loadgen.py` asks for worker identity in the request body:

```json
{"nvext": {"extra_fields": ["worker_id", "engine_data"]}}
```

Dynamo then puts `nvext.worker_id.decode_worker_id`,
`decode_dp_rank` (and the prefill equivalents) on every streamed chunk
(`lib/llm/src/protocols/common/extensions.rs` in the pinned Dynamo checkout),
and copies the proxy's `engine_data {served_by, tier}` into `nvext.engine_data`.
`report.py` maps `decode_dp_rank` through the tier map (derived by `run.sh` from
the generated proxy configs, or passed to a manual run) to a tier; ranks outside
the ranges are `primary`.

The proxies need the chat request itself, not only token ids. Each proxy
advertises the `chat_request` runtime capability, and the frontend's KV router
puts the chat request in `extra_args.chat_request` when it dispatches to one;
primary workers never receive it. The load generator does nothing special for
this.

## Model-card matching

The frontend only admits workers whose `ModelDeploymentCard` checksum matches
the first one in the endpoint's WorkerSet (`lib/llm/src/discovery/controller.rs`,
`DesiredGroup`). `mdcsum()` hashes `source_path`, the tokenizer/config checksums,
`kv_cache_block_size`, `worker_type`, `router_config`, aliases and
`runtime_config.context_length`. `run.sh` therefore makes every worker use the
same hub-id `source_path` (`MODEL_ID`), block size, context length and router
config: a local `MODEL_PATH` directory is seeded into `HF_HUB_CACHE` under
`MODEL_ID`, each mocker runs with `--engine-type vllm --max-model-len
$CONTEXT_LENGTH` and the generated `router/<model>/primary.args`
(`--router-mode kv --router-track-active-blocks ...`) so it advertises the same
context and card `router_config` as the proxies, and the generated
`proxy-x-0.yaml` / `proxy-y-0.yaml` carry the matching `router_config` from the
same `spillover-deploy` run. The frontend is the only process that loads the
local directory, through `--model-path $FRONTEND_MODEL_PATH`, and it runs
**without** a frontend-wide tracking flag: each worker set advertises tracking on
its own model card instead.

`mdcsum()` also hashes the card's `extra_files` (sorted basename + checksum
pairs). A worker with `DYN_SYSTEM_PORT` self-hosts its card over HTTP and
harvests the model directory's sibling files (`merges.txt`, `vocab.json`) into
`extra_files`; a worker without a system port uses shared-storage metadata and
advertises no `extra_files`. Both sides are internally consistent, but mixing
them splits the WorkerSet. `run.sh` gives every worker process a system port
(`PRIMARY_SYSTEM_PORT`/`PROXY_*_SYSTEM_PORT`) so all cards carry the same
`extra_files`. This is also why primary mockers run one process per worker: a
single mocker process with `--num-workers N` starts N runtime instances but only
the first can bind the port, and the rest fall back to shared-storage cards.

A checksum mismatch shows up in `logs/frontend.log` as `Rejected incompatible
workers` and the affected worker never joins the set.

## CI

- `.github/workflows/spillover.yml` runs on pull requests that touch
  `lib/spillover/**`, `lib/router-plugins/spillover/**`, `lib/router-plugins/catalog/**`
  or `lib/llm/**` (the chat-request integration): it runs fmt, clippy and tests for the
  spillover crates, builds `dw-proxy-worker`, and appends each `routing-sim`
  scenario's markdown to the job summary.
- `.github/workflows/spillover-nightly.yml` (nightly cron and `workflow_dispatch`)
  builds the fork's Python frontend, then runs this harness end to end with the
  baseline-independent routing checks enabled, so a regression that keeps requests
  succeeding but sends none of them to a proxy fails the job. See
  `docs/spillover/PLAN.md` Level 2.
