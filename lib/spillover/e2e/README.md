# Level 2 end-to-end simulation

Starts a real Dynamo frontend (built with our `dw-spillover` catalog), GPU-free
mocker hosted workers, two `dw-proxy-worker` processes pointed at fake
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
| `loadgen.py` | Multi-turn sessions, shared system prompt, think time, arrival-rate profile, streams and records the serving worker |
| `run.sh` | Starts providers, frontend, mocker workers and proxies; runs loadgen and the report; cleans up |
| `report.py` | Per-worker/per-tier shares over time, stickiness, errors; markdown, optional Level 1 comparison |
| `config/policy.yaml` | `dw-spillover` router-policy config (tier DP ranks 1000/2000) |
| `config/level1-equivalent.yaml` | `routing-sim` scenario twin of the e2e run for `report.py --baseline` |
| `config/proxy-x.yaml`, `config/proxy-y.yaml` | `dw-proxy-worker` configs, one per tier, with fake provider URLs |
| `config/arrival-profile.json` | Piecewise-linear arrival rate used by `run.sh` |
| `config/tier-map.json` | DP-rank ranges used by `report.py` to label tiers |
| `requirements.txt` | Empty: everything is Python standard library |

## Prerequisites

- A Dynamo checkout with the Python bindings built. This fork builds the
  frontend from the checked-out tree: `maturin develop` in `lib/bindings/python`
  plus the `ai-dynamo` package (`pip install -e .`), as in the contribution
  guide. `custom-policy` is a default feature and `lib/router-plugins/catalog`
  registers `dw-spillover`, so the fork's normal Python build already includes
  the policy. `run.sh` invokes `python3 -m dynamo.frontend`, so put that venv's
  `bin/` on `PATH`.
- A built `dw-proxy-worker` binary; `run.sh` builds one into
  `$CARGO_TARGET_DIR/debug/dw-proxy-worker` unless `SKIP_BUILD=1`.
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
CARGO_TARGET_DIR=/home/peter/.cache/dw-target-b5 \
SKIP_BUILD=1 \
lib/spillover/e2e/run.sh
```

`MODEL_PATH` being a directory makes `run.sh` seed the offline HF cache under
`MODEL_ID` (`Qwen/Qwen3-0.6B` by default) and register the workers with that hub
id, because the frontend requires `--model-path` to be a real directory while
workers must advertise a matching `source_path`. The default scenario
(`HOSTED_BLOCKS=8`, `SPEEDUP=1`) ramps past the hosted capacity and spills a
large share to proxy X.

Outputs land in `lib/spillover/e2e/out/`:

- `logs/` — one log per process; the first place to look when routing looks wrong.
- `reports/loadgen.jsonl` — one JSON object per turn (arrival/start/end times,
  latency, TTFT, status, worker id, DP ranks, usage).
- `reports/provider-x.jsonl`, `provider-y.jsonl` — one line per provider request
  (provider name, prompt size, status, latency).
- `reports/e2e-report.md` and `e2e-report.json` — the report described below.

To compare with Level 1, build a `routing-sim` JSON report and pass it as
`BASELINE`:

```bash
# the Level 1 twin must use the same hosted capacity, block size, tiers and
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
| `MODEL` | `$MODEL_PATH` | Served model name; must match `config/policy.yaml` |
| `HF_HUB_CACHE` | `$OUT_DIR/hf-cache` | Offline HF cache `LocalModel::fetch` resolves hub ids from |
| `HF_HUB_OFFLINE` | `1` with a local `MODEL_PATH` | Prevents model resolution from touching the network; a hub-id `MODEL_PATH` leaves it off to allow the one download |
| `FRONTEND_PORT` | `8000` | Frontend HTTP port |
| `PROVIDER_X_PORT` / `PROVIDER_Y_PORT` | `9101` / `9102` | Fake provider ports |
| `HOSTED_WORKERS` | `2` | mocker `--num-workers` |
| `HOSTED_BLOCKS` | `8` | mocker KV blocks; must match `hosted_capacity_blocks` in the policy (small enough that the ramp spills) |
| `HOSTED_QUEUE_MARGIN` | `256` | `DYN_ADMISSION_QUEUE_MARGIN` set on the hosted workers; must be above the policy's failover point. Proxies are launched with it unset |
| `BLOCK_SIZE` | `64` | KV block size; must match the proxy configs |
| `CONTEXT_LENGTH` | `32768` | mocker `--max-model-len` and proxy `context_length`; the cohort checksum includes it |
| `ENGINE_TYPE` | `vllm` | Mocker engine; `vllm` accepts `--max-model-len`, `sglang` does not |
| `MAX_SEQS` | `64` | mocker concurrency |
| `SPEEDUP` | `1.0` | mocker `--speedup-ratio`; 1.0 keeps hosted workers slow enough to spill |
| `SESSIONS`, `TURNS`, `THINK_TIME` | `200`, `4`, `0.5` | Load shape; sessions are capped by `--duration` |
| `MAX_TOKENS` | `32` | Output length asked of the provider |
| `ARRIVAL_RATE`, `DURATION` | `2.0`, `62` | Arrival rate (sessions/s) and scheduling window; `config/arrival-profile.json` overrides the rate |
| `PROVIDER_X_CONCURRENCY` / `PROVIDER_Y_CONCURRENCY` | `8` | Requests in flight before 429 |
| `PROVIDER_X_ERROR_RATE` / `PROVIDER_Y_ERROR_RATE` | `0.0` | Fraction answered with HTTP 503 |
| `BIN_SECONDS` | `10` | Time-window width in the report |
| `WORKER_WAIT` | `600` | Seconds to wait for the model to register |
| `OUT_DIR` | `lib/spillover/e2e/out` | Where logs and reports go |
| `SKIP_BUILD` | `0` | Set to `1` to reuse an existing proxy binary |
| `CARGO_TARGET_DIR` | `target/` | Build directory for the proxy worker |
| `PROXY_BIN` | `$CARGO_TARGET_DIR/debug/dw-proxy-worker` | Proxy binary to run |
| `BASELINE`, `TOLERANCE` | empty, `0.1` | Optional Level 1 report and relative tolerance |

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
`--tier-map`.

## Reading the report

- **Per worker class (overall)** — requests, share, failures and latency/TTFT
  percentiles for `hosted`, `proxy-x`, `proxy-y` (and `unknown` if a response
  had no `nvext.worker_id`).
- **Shares over time** — the same shares per `--bin-seconds` window, so a spill
  ramp and recovery are visible.
- **Stickiness** — fraction of follow-up turns served by the same worker (and,
  separately, the same class) as the previous turn, taken from `nvext.worker_id`
  on streamed chunks. Class stickiness is the metric Level 1 reports.
- **Fake providers** — requests per provider with HTTP status counts and mean
  prompt size, which shows 429s from a capped proxy.
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
`config/level1-equivalent.yaml` reaches the same split (hosted 0.54 vs 0.53,
proxy-x 0.44 vs 0.45, proxy-y 0.021 vs 0.019 in the current run). Getting there
needs two deliberate choices:

- **Backend speed.** `routing-sim` needs explicit token rates; the mocker does
  not expose an equivalent, so the twin's hosted `prefill_tokens_per_second`,
  `decode_tokens_per_second` and `batching_slowdown` are fitted to the e2e's
  observed hosted latency (p50 ~2.5 s, TTFT ~1.4 s). The proxies keep the fake
  provider's real 200 tps. Without this fit the sim's hosted workers are far
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
{"nvext": {"extra_fields": ["worker_id"]}}
```

Dynamo then puts `nvext.worker_id.decode_worker_id`,
`decode_dp_rank` (and the prefill equivalents) on every streamed chunk
(`lib/llm/src/protocols/common/extensions.rs` in the pinned Dynamo checkout).
`report.py` maps `decode_dp_rank` through `config/tier-map.json` to a tier;
ranks outside the ranges are `hosted`.

The same `extra_fields` array carries the original chat body for the proxies as
`dw.orig.v1:<base64url>`. Dynamo forwards `nvext.extra_fields` to the worker
verbatim, and `dw_proxy_core::orig::from_extra_args` decodes it. In production
that entry is added by onwards; the simulation has no onwards, so `loadgen.py`
stamps the field-selected request itself (`orig_field`, mirroring
`dw_proxy_core::orig::CARRIED_FIELDS`). Without it a proxy rejects the request
with `request is missing the dw.orig chat payload`.

## Model-card matching

The frontend only admits workers whose `ModelDeploymentCard` checksum matches
the first one in the endpoint's WorkerSet (`lib/llm/src/discovery/controller.rs`,
`DesiredGroup`). `mdcsum()` hashes `source_path`, the tokenizer/config checksums,
`kv_cache_block_size`, `worker_type`, `router_config`, aliases and
`runtime_config.context_length`. `run.sh` therefore makes every worker use the
same hub-id `source_path` (`MODEL_ID`), block size, context length and router
config: a local `MODEL_PATH` directory is seeded into `HF_HUB_CACHE` under
`MODEL_ID`, the mocker runs with `--engine-type vllm --max-model-len
$CONTEXT_LENGTH` and `--router-mode kv --router-track-active-blocks` so it
advertises the same context and card `router_config` as the proxies, and both
`proxy-x.yaml` and `proxy-y.yaml` carry the matching `router_config`. The
frontend is the only process that loads the local directory, through
`--model-path $FRONTEND_MODEL_PATH`, and it runs **without** a frontend-wide
tracking flag. A checksum mismatch shows up in `logs/frontend.log` as `Rejected
incompatible workers` and the proxy never joins the set.

## CI

- `.github/workflows/spillover.yml` runs on pull requests that touch
  `lib/spillover/**`, `lib/router-plugins/spillover/**` or
  `lib/router-plugins/catalog/**`: it runs fmt, clippy and tests for the
  spillover crates, builds `dw-proxy-worker`, and appends each `routing-sim`
  scenario's markdown to the job summary.
- No nightly end-to-end workflow exists in the fork yet. Run `run.sh` by hand
  once a frontend with the catalog is available; a nightly workflow can be
  added alongside the fork's images.
