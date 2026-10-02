#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Level 2 end-to-end spillover simulation.
#
# Starts a real Dynamo frontend built with our catalog, N mocker primary workers,
# two dw-proxy-worker processes pointed at fake providers, then drives them with
# loadgen.py and writes a report. File discovery plus TCP request plane and ZMQ
# event plane means no etcd or NATS. Everything runs on 127.0.0.1.
#
# Environment overrides are documented in lib/spillover/e2e/README.md.

set -euo pipefail

E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$E2E_DIR/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
# Hub id the mockers and proxies register as `source_path`. A local `MODEL_PATH`
# directory is seeded into an offline HF hub cache under this id so every worker
# (and the frontend) advertises the same `source_path` and shares one WorkerSet.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
PROVIDER_X_PORT="${PROVIDER_X_PORT:-9101}"
PROVIDER_Y_PORT="${PROVIDER_Y_PORT:-9102}"
PRIMARY_WORKERS="${PRIMARY_WORKERS:-2}"
PRIMARY_BLOCKS="${PRIMARY_BLOCKS:-8}"
# Engine-queue admission margin for each primary worker process
# (`DYN_ADMISSION_QUEUE_MARGIN`). Default matches `spillover-deploy`'s
# DEFAULT_ADMISSION_QUEUE_MARGIN and sits above the policy's failover point; see
# docs/spillover/tuning.md. Proxies are explicitly opted out below.
PRIMARY_QUEUE_MARGIN="${PRIMARY_QUEUE_MARGIN:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_SEQS="${MAX_SEQS:-64}"
ENGINE_TYPE="${ENGINE_TYPE:-vllm}"
SPEEDUP="${SPEEDUP:-1.0}"
SESSIONS="${SESSIONS:-200}"
TURNS="${TURNS:-4}"
THINK_TIME="${THINK_TIME:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-32}"
ARRIVAL_RATE="${ARRIVAL_RATE:-2.0}"
DURATION="${DURATION:-62}"
BIN_SECONDS="${BIN_SECONDS:-10}"
WORKER_WAIT="${WORKER_WAIT:-600}"
OUT_DIR="${OUT_DIR:-$E2E_DIR/out}"
SKIP_BUILD="${SKIP_BUILD:-0}"
BASELINE="${BASELINE:-}"
TOLERANCE="${TOLERANCE:-0.1}"
# Baseline-independent routing assertions. The default scenario ramps past
# primary capacity, so a run where no request reached a proxy means spillover is
# broken even when every request succeeded. `--require-routing` fails on a
# non-zero failed-request count, an untagged/mis-tiered proxy response, a proxy
# share below the floor, or a required tier never being observed.
REQUIRE_ROUTING="${REQUIRE_ROUTING:-1}"
MIN_PROXY_SHARE="${MIN_PROXY_SHARE:-0.05}"
MAX_PRIMARY_SHARE="${MAX_PRIMARY_SHARE:-}"
REQUIRE_TIERS="${REQUIRE_TIERS:-proxy-x}"

RUN_DIR="$OUT_DIR/run"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/reports"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
PROXY_BIN="${PROXY_BIN:-$CARGO_TARGET_DIR/debug/dw-proxy-worker}"
DEPLOY_BIN="${DEPLOY_BIN:-$CARGO_TARGET_DIR/debug/spillover-deploy}"
# Per-process system status ports. The runtime disables them globally below;
# each worker process opts back in on its own port. Besides metrics, a system
# port is what makes a worker self-host its model card instead of using shared
# storage; the two modes produce different `extra_files` and therefore different
# card checksums, which would split the WorkerSet. Every process here self-hosts
# so primary mockers, proxies and (in production) primary workers of any engine share one set.
PRIMARY_SYSTEM_PORT="${PRIMARY_SYSTEM_PORT:-9200}"
# Kept clear of the primary range (`PRIMARY_SYSTEM_PORT + PRIMARY_WORKERS - 1`) so the
# per-worker mocker ports and the proxy ports never collide.
PROXY_X_SYSTEM_PORT="${PROXY_X_SYSTEM_PORT:-9211}"
PROXY_Y_SYSTEM_PORT="${PROXY_Y_SYSTEM_PORT:-9212}"
# Seconds between proxy metrics scrapes while the load generator runs.
METRICS_INTERVAL="${METRICS_INTERVAL:-2}"

export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="$RUN_DIR/dynamo_store_kv"
# Where model files live. A local `MODEL_PATH` directory is copied into an
# offline HF cache under `MODEL_ID`; a hub id is resolved from that cache or
# downloaded. The proxies and mocker resolve the same cache.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$OUT_DIR/hf-cache}"
mkdir -p "$HF_HUB_CACHE"
# -1 disables the per-process metrics/health server; several processes share this host.
export DYN_SYSTEM_PORT=-1
export DYN_FAKE_API_KEY="${DYN_FAKE_API_KEY:-fake-key}"

PIDS=()

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [ "${#PIDS[@]}" -gt 0 ]; then
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in "${PIDS[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    wait 2>/dev/null || true
    exit "$code"
}
trap cleanup EXIT INT TERM

start() {
    local name="$1"
    shift
    "$@" >"$LOG_DIR/$name.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    # A process that cannot bind its port (or otherwise fails at startup)
    # would otherwise only surface much later, after readiness was satisfied
    # by a stale process. Fail fast instead.
    sleep 0.3
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "process $name (pid $pid) exited during startup; see $LOG_DIR/$name.log" >&2
        exit 1
    fi
    echo "started $name (pid $pid)"
}

wait_for_workers() {
    # Wait for the whole worker set, not just any one registration: /v1/models
    # lists a model as soon as one worker registers, so a run where half the
    # mockers or a proxy failed to start would otherwise proceed. The readiness
    # breakdown reports live workers per namespace and type.
    local expected=$((PRIMARY_WORKERS + 2))
    local deadline=$((SECONDS + WORKER_WAIT))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if python3 - "$FRONTEND_PORT" "$MODEL" "$expected" <<'PY'
import json
import sys
import urllib.request

port, model, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    url = f"http://127.0.0.1:{port}/v1/models/{model}/ready"
    with urllib.request.urlopen(url, timeout=2) as response:
        data = json.load(response)
except Exception:
    sys.exit(1)
total = sum(
    int(worker_type.get("workers") or 0)
    for namespace in (data.get("namespaces") or {}).values()
    for worker_type in (namespace.get("worker_types") or {}).values()
)
sys.exit(0 if total >= expected else 1)
PY
        then
            return 0
        fi
        sleep 2
    done
    echo "workers did not register within ${WORKER_WAIT}s" >&2
    return 1
}

rm -rf "$RUN_DIR" "$LOG_DIR" "$REPORT_DIR"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$REPORT_DIR" "$DYN_FILE_KV"

# Seed the offline hub cache. `LocalModel::fetch` resolves a hub id from the
# standard `models--<org>--<name>/snapshots/<rev>` layout. Only config and
# tokenizer files are needed; a placeholder weight file satisfies the
# `ignore_weights=false` check the proxy performs (it never reads weights).
seed_hub_cache() {
    local src="$1" id="$2"
    local repo="$HF_HUB_CACHE/models--${id//\//--}"
    local rev="0000000000000000000000000000000000000000"
    rm -rf "$repo"
    mkdir -p "$repo/refs" "$repo/snapshots/$rev"
    printf '%s' "$rev" >"$repo/refs/main"
    cp "$src"/* "$repo/snapshots/$rev/" 2>/dev/null || true
    : >"$repo/snapshots/$rev/model.safetensors"
}

# A hub id cannot be passed to the frontend `--model-path` (it must be a
# directory), so materialize its tokenizer/config snapshot and print the
# directory. `HF_HUB_OFFLINE=1` makes this resolve from `HF_HUB_CACHE` only.
resolve_model_dir() {
    python3 - "$1" <<'PY'
import sys

from huggingface_hub import snapshot_download

allow = ["*.json", "*.txt", "*.model"]
print(snapshot_download(sys.argv[1], allow_patterns=allow))
PY
}

if [ -d "$MODEL_PATH" ]; then
    # A local tokenizer directory: the frontend loads it directly, while the
    # workers register the hub id so both sides advertise one `source_path`.
    # Offline is safe here because everything is copied into the cache below.
    export HF_HUB_OFFLINE=1
    FRONTEND_MODEL_PATH="${FRONTEND_MODEL_PATH:-$MODEL_PATH}"
    seed_hub_cache "$MODEL_PATH" "$MODEL_ID"
    MODEL_PATH="$MODEL_ID"
else
    # A hub id: download (or, with HF_HUB_OFFLINE, resolve) the snapshot the
    # frontend and the workers all read. The mocker never loads weights, so a
    # placeholder satisfies the card's weight check without a large download.
    FRONTEND_MODEL_PATH="${FRONTEND_MODEL_PATH:-$(resolve_model_dir "$MODEL_PATH")}"
    [ -e "$FRONTEND_MODEL_PATH/model.safetensors" ] || : >"$FRONTEND_MODEL_PATH/model.safetensors"
fi
FRONTEND_MODEL_PATH="${FRONTEND_MODEL_PATH:-$MODEL_PATH}"
MODEL="${MODEL:-$MODEL_PATH}"

if [ "$SKIP_BUILD" != "1" ]; then
    echo "building dw-proxy-worker and spillover-deploy (set SKIP_BUILD=1 to reuse them)"
    CARGO_TARGET_DIR="$CARGO_TARGET_DIR" cargo build \
        -p dw-proxy-worker -p dw-spillover-deploy \
        --manifest-path "$REPO_ROOT/Cargo.toml"
fi
if [ ! -x "$PROXY_BIN" ]; then
    echo "proxy binary not found: $PROXY_BIN" >&2
    exit 1
fi
if [ ! -x "$DEPLOY_BIN" ]; then
    echo "spillover-deploy binary not found: $DEPLOY_BIN" >&2
    exit 1
fi

# Refuse to start if any port the run needs is already bound, or if two of the
# configured ports collide (e.g. PRIMARY_WORKERS large enough that the primary
# system-port range reaches the proxy ports). Without this, a stale process can
# satisfy readiness and be scraped as if it were this run's.
PREFLIGHT_PORTS=(
    "$FRONTEND_PORT"
    "$PROVIDER_X_PORT"
    "$PROVIDER_Y_PORT"
    "$PROXY_X_SYSTEM_PORT"
    "$PROXY_Y_SYSTEM_PORT"
)
for i in $(seq 0 $((PRIMARY_WORKERS - 1))); do
    PREFLIGHT_PORTS+=("$((PRIMARY_SYSTEM_PORT + i))")
done
if ! python3 "$E2E_DIR/run_helpers.py" check-ports --ports "${PREFLIGHT_PORTS[@]}"; then
    echo "run.sh: required ports are not available" >&2
    exit 1
fi

# The proxy and policy configs come from `spillover-deploy`, the same generator
# production deployments use, so the e2e run exercises it rather than
# hand-written configs. The deployment template is rendered with this run's
# model id, block sizes, capacities and provider ports first.
DEPLOY_INPUT="$RUN_DIR/deployments.yaml"
DEPLOY_DIR="$RUN_DIR/generated"
# Render with Python's literal `${NAME}` templating rather than sed, so paths
# containing `&`, `|` or `\` are inserted unchanged.
render_config() {
    python3 "$E2E_DIR/run_helpers.py" render \
        --template "$1" --out "$2" \
        --set "MODEL_PATH=$MODEL_PATH" \
        --set "MODEL=$MODEL" \
        --set "PRIMARY_BLOCKS=$PRIMARY_BLOCKS" \
        --set "BLOCK_SIZE=$BLOCK_SIZE" \
        --set "CONTEXT_LENGTH=$CONTEXT_LENGTH" \
        --set "PRIMARY_QUEUE_MARGIN=$PRIMARY_QUEUE_MARGIN" \
        --set "PROVIDER_X_PORT=$PROVIDER_X_PORT" \
        --set "PROVIDER_Y_PORT=$PROVIDER_Y_PORT"
}
render_config "$E2E_DIR/config/deployments.yaml" "$DEPLOY_INPUT"
"$DEPLOY_BIN" generate --input "$DEPLOY_INPUT" --out "$DEPLOY_DIR"

# `spillover-deploy` names each model directory after the sanitized model name
# (one `_` per *character* outside [A-Za-z0-9._-]). `run_helpers.sanitize_model_dir`
# mirrors that char-wise rule; a byte-wise `tr` would disagree on non-ASCII names.
MODEL_DIR_NAME="$(python3 "$E2E_DIR/run_helpers.py" sanitize --name "$MODEL")"
POLICY_CONFIG="$DEPLOY_DIR/router-policy.yaml"
PROXY_X_CONFIG="$DEPLOY_DIR/$MODEL_DIR_NAME/proxy-x-0.yaml"
PROXY_Y_CONFIG="$DEPLOY_DIR/$MODEL_DIR_NAME/proxy-y-0.yaml"
PRIMARY_ROUTER_ARGS_FILE="$DEPLOY_DIR/router/$MODEL_DIR_NAME/primary.args"
PRIMARY_ENV_FILE="$DEPLOY_DIR/admission/$MODEL_DIR_NAME/primary.env"
PROXY_ENV_FILE="$DEPLOY_DIR/admission/$MODEL_DIR_NAME/proxy.env"
for required in "$POLICY_CONFIG" "$PROXY_X_CONFIG" "$PROXY_Y_CONFIG" \
    "$PRIMARY_ROUTER_ARGS_FILE" "$PRIMARY_ENV_FILE" "$PROXY_ENV_FILE"; do
    if [ ! -f "$required" ]; then
        echo "spillover-deploy did not emit $required" >&2
        exit 1
    fi
done
# `tier` and `provider` are independent config fields; read the provider names
# the generated configs actually carry so the metrics check asserts each.
PROVIDER_X_NAME="$(python3 "$E2E_DIR/run_helpers.py" proxy-provider --config "$PROXY_X_CONFIG")"
PROVIDER_Y_NAME="$(python3 "$E2E_DIR/run_helpers.py" proxy-provider --config "$PROXY_Y_CONFIG")"
# Derive the tier rank ranges from the configs this run generated, so the report
# cannot misclassify a proxy because a checked-in rank table drifted from the
# deployment generator.
TIER_MAP_FILE="$RUN_DIR/tier-map.json"
python3 "$E2E_DIR/run_helpers.py" tier-map \
    --proxy "$PROXY_X_CONFIG" --proxy "$PROXY_Y_CONFIG" >"$TIER_MAP_FILE"
# The primary worker `--router-*` flags become the mocker's, so the primary and
# proxy model cards carry the same router_config and stay one worker set.
PRIMARY_ROUTER_ARGS="$(grep -v '^[[:space:]]*#' "$PRIMARY_ROUTER_ARGS_FILE" | tr '\n' ' ')"
# The engine-queue margin is read per worker process, so source the generated
# primary env and rely on the generated proxy env's explicit opt-out below.
# shellcheck disable=SC1090
source "$PRIMARY_ENV_FILE"
PRIMARY_QUEUE_MARGIN="${DYN_ADMISSION_QUEUE_MARGIN:-$PRIMARY_QUEUE_MARGIN}"

echo "starting fake providers"
start provider-x python3 "$E2E_DIR/fake_provider.py" \
    --port "$PROVIDER_X_PORT" --name proxy-x --model "$MODEL" \
    --ttft-ms "${PROVIDER_X_TTFT_MS:-30}" --tps "${PROVIDER_X_TPS:-200}" \
    --max-tokens "$MAX_TOKENS" --concurrency "${PROVIDER_X_CONCURRENCY:-8}" \
    --error-rate "${PROVIDER_X_ERROR_RATE:-0.0}" \
    --log "$REPORT_DIR/provider-x.jsonl"
start provider-y python3 "$E2E_DIR/fake_provider.py" \
    --port "$PROVIDER_Y_PORT" --name proxy-y --model "$MODEL" \
    --ttft-ms "${PROVIDER_Y_TTFT_MS:-30}" --tps "${PROVIDER_Y_TPS:-200}" \
    --max-tokens "$MAX_TOKENS" --concurrency "${PROVIDER_Y_CONCURRENCY:-8}" \
    --error-rate "${PROVIDER_Y_ERROR_RATE:-0.0}" \
    --log "$REPORT_DIR/provider-y.jsonl"

echo "starting Dynamo frontend on port $FRONTEND_PORT"
# No frontend-wide --router-track-active-blocks: every worker set advertises
# `router_track_active_blocks` on its model card below, so tracking is enabled
# only for the spillover model. A frontend-wide flag would also work but would
# change routing for every other model the frontend serves.
start frontend python3 -m dynamo.frontend \
    --http-port "$FRONTEND_PORT" \
    --model-path "$FRONTEND_MODEL_PATH" \
    --router-mode kv \
    --router-policy-config "$POLICY_CONFIG" \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq

echo "starting $PRIMARY_WORKERS mocker primary worker(s)"
# The admission margin is a worker-process environment value; real primary workers on an
# engine that reports waiting (SGLang, vLLM, TRT-LLM with --publish-metrics) enforce it.
# The mocker never reports an engine waiting queue, so the generator emits `unset
# DYN_ADMISSION_QUEUE_MARGIN` for it and the run does not set one on the mocker processes:
# setting a margin on an engine that cannot enforce it removes admission control rather than
# bounding it.
#
# One mocker process per worker, each with its own system port. A single mocker process
# with `--num-workers N` starts N runtime instances, but only the first can bind the
# system port; the rest fall back to shared-storage metadata and advertise a card
# without `extra_files`. Because `extra_files` participates in the card checksum, that
# split the primary WorkerSet (one member self-hosted, one not) and excluded the proxies.
for i in $(seq 0 $((PRIMARY_WORKERS - 1))); do
    start "mocker-$i" env \
        DYN_SYSTEM_PORT="$((PRIMARY_SYSTEM_PORT + i))" python3 -m dynamo.mocker \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL" \
        --endpoint "dyn://dynamo.backend.generate" \
        --engine-type "$ENGINE_TYPE" \
        --max-model-len "$CONTEXT_LENGTH" \
        --num-workers 1 \
        --num-gpu-blocks-override "$PRIMARY_BLOCKS" \
        --block-size "$BLOCK_SIZE" \
        --max-num-seqs "$MAX_SEQS" \
        --speedup-ratio "$SPEEDUP" \
        $PRIMARY_ROUTER_ARGS \
        --discovery-backend file \
        --request-plane tcp \
        --event-plane zmq
done

echo "starting proxy workers"
# Proxies never report num_waiting_reqs, so the margin is unenforceable on them; clear it
# explicitly so a value cannot leak in from the surrounding launch environment.
# `proxy.env` says `unset DYN_ADMISSION_QUEUE_MARGIN`: proxies never report
# engine waiting, and DYN_SYSTEM_PORT is re-enabled so metrics can be scraped.
start proxy-x env -u DYN_ADMISSION_QUEUE_MARGIN DYN_SYSTEM_PORT="$PROXY_X_SYSTEM_PORT" \
    "$PROXY_BIN" --config "$PROXY_X_CONFIG"
start proxy-y env -u DYN_ADMISSION_QUEUE_MARGIN DYN_SYSTEM_PORT="$PROXY_Y_SYSTEM_PORT" \
    "$PROXY_BIN" --config "$PROXY_Y_CONFIG"

echo "waiting for '$MODEL' to register (up to ${WORKER_WAIT}s)"
wait_for_workers

# Scrape each proxy's Prometheus endpoint while load runs, appending a
# timestamped snapshot to reports/metrics-<tier>.prom. This is how the run
# proves the proxy metrics surface (requests, tokens, TTFT, virtual cache) is
# live.
: >"$REPORT_DIR/metrics-proxy-x.prom"
: >"$REPORT_DIR/metrics-proxy-y.prom"
scrape_metrics() {
    local port="$1" out="$2"
    while [ -f "$RUN_DIR/scrape.on" ]; do
        {
            echo "# scrape $(date +%s)"
            curl -fsS "http://127.0.0.1:$port/metrics" 2>/dev/null || echo "# scrape failed"
        } >>"$out"
        sleep "$METRICS_INTERVAL"
    done
}
touch "$RUN_DIR/scrape.on"
scrape_metrics "$PROXY_X_SYSTEM_PORT" "$REPORT_DIR/metrics-proxy-x.prom" &
SCRAPE_X_PID=$!
scrape_metrics "$PROXY_Y_SYSTEM_PORT" "$REPORT_DIR/metrics-proxy-y.prom" &
SCRAPE_Y_PID=$!
PIDS+=("$SCRAPE_X_PID" "$SCRAPE_Y_PID")

# Wait for the proxy metrics servers to answer before generating load. They are
# started with the workers and may take a moment to bind; a timeout is a real
# failure, not a reason to continue and scrape nothing.
metrics_ready=0
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PROXY_X_SYSTEM_PORT/metrics" >/dev/null 2>&1 && \
       curl -fsS "http://127.0.0.1:$PROXY_Y_SYSTEM_PORT/metrics" >/dev/null 2>&1; then
        metrics_ready=1
        break
    fi
    sleep 1
done
if [ "$metrics_ready" -ne 1 ]; then
    echo "proxy metrics endpoints did not become ready; see $LOG_DIR" >&2
    exit 1
fi

echo "running load generator for up to ${DURATION}s"
loadgen_args=(
    --url "http://127.0.0.1:$FRONTEND_PORT"
    --model "$MODEL"
    --sessions "$SESSIONS"
    --turns "$TURNS"
    --think-time "$THINK_TIME"
    --max-tokens "$MAX_TOKENS"
    --arrival-rate "$ARRIVAL_RATE"
    --duration "$DURATION"
    --arrival-profile "$E2E_DIR/config/arrival-profile.json"
    --out "$REPORT_DIR/loadgen.jsonl"
    --summary "$REPORT_DIR/loadgen-summary.json"
)
loadgen_status=0
python3 "$E2E_DIR/loadgen.py" "${loadgen_args[@]}" || loadgen_status=$?

# Stop scraping and summarize the proxy metrics the run collected.
rm -f "$RUN_DIR/scrape.on"
wait "$SCRAPE_X_PID" 2>/dev/null || true
wait "$SCRAPE_Y_PID" 2>/dev/null || true
metrics_status=0
# Only the tiers the run requires may be asserted to have served traffic; the
# others still prove their metrics surface is live. Without --require-routing no
# tier is required.
metrics_tier_args=()
if [ "$REQUIRE_ROUTING" = "1" ]; then
    for tier in $REQUIRE_TIERS; do
        metrics_tier_args+=(--required-tier "$tier")
    done
else
    metrics_tier_args+=(--no-required-tiers)
fi
python3 "$E2E_DIR/check_metrics.py" \
    --metrics "proxy-x=$REPORT_DIR/metrics-proxy-x.prom" \
    --metrics "proxy-y=$REPORT_DIR/metrics-proxy-y.prom" \
    --providers "proxy-x=$PROVIDER_X_NAME" \
    --providers "proxy-y=$PROVIDER_Y_NAME" \
    "${metrics_tier_args[@]}" \
    --out "$REPORT_DIR/metrics.json" || metrics_status=$?

comparison_args=()
if [ -n "$BASELINE" ]; then
    comparison_args=(--baseline "$BASELINE" --tolerance "$TOLERANCE")
fi
routing_args=()
if [ "$REQUIRE_ROUTING" = "1" ]; then
    routing_args=(--require-routing --min-proxy-share "$MIN_PROXY_SHARE"
        --tier-map "$TIER_MAP_FILE")
    if [ -n "$MAX_PRIMARY_SHARE" ]; then
        routing_args+=(--max-primary-share "$MAX_PRIMARY_SHARE")
    fi
    for tier in $REQUIRE_TIERS; do
        routing_args+=(--require-tier "$tier")
    done
fi
report_status=0
python3 "$E2E_DIR/report.py" \
    --loadgen "$REPORT_DIR/loadgen.jsonl" \
    --provider-log "proxy-x=$REPORT_DIR/provider-x.jsonl" \
    --provider-log "proxy-y=$REPORT_DIR/provider-y.jsonl" \
    --tier-map "$TIER_MAP_FILE" \
    --bin-seconds "$BIN_SECONDS" \
    --json "$REPORT_DIR/e2e-report.json" \
    --markdown "$REPORT_DIR/e2e-report.md" \
    "${comparison_args[@]+"${comparison_args[@]}"}" \
    "${routing_args[@]+"${routing_args[@]}"}" || report_status=$?

echo
cat "$REPORT_DIR/e2e-report.md"
echo
echo "logs:    $LOG_DIR"
echo "reports: $REPORT_DIR"

if [ "$loadgen_status" -ne 0 ] || [ "$report_status" -ne 0 ] || [ "$metrics_status" -ne 0 ]; then
    exit 1
fi
