#!/usr/bin/env bash
# Level 2 end-to-end spillover simulation.
#
# Starts a real Dynamo frontend built with our catalog, N mocker hosted workers,
# two dw-proxy-worker processes pointed at fake providers, then drives them with
# loadgen.py and writes a report. File discovery plus TCP request plane and ZMQ
# event plane means no etcd or NATS. Everything runs on 127.0.0.1.
#
# Environment overrides are documented in sim/e2e/README.md.

set -euo pipefail

E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$E2E_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
# Hub id the mockers and proxies register as `source_path`. A local `MODEL_PATH`
# directory is seeded into an offline HF hub cache under this id so every worker
# (and the frontend) advertises the same `source_path` and shares one WorkerSet.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
PROVIDER_X_PORT="${PROVIDER_X_PORT:-9101}"
PROVIDER_Y_PORT="${PROVIDER_Y_PORT:-9102}"
HOSTED_WORKERS="${HOSTED_WORKERS:-2}"
HOSTED_BLOCKS="${HOSTED_BLOCKS:-8}"
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

RUN_DIR="$OUT_DIR/run"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/reports"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
PROXY_BIN="${PROXY_BIN:-$CARGO_TARGET_DIR/debug/dw-proxy-worker}"

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
    echo "started $name (pid $pid)"
}

wait_for_workers() {
    local deadline=$((SECONDS + WORKER_WAIT))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if python3 - "$FRONTEND_PORT" "$MODEL" <<'PY'
import json
import sys
import urllib.request

port, model = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
        data = json.load(response)
except Exception:
    sys.exit(1)
ids = {entry.get("id") for entry in data.get("data", [])}
sys.exit(0 if model in ids else 1)
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

# The proxy and policy configs are templates: substitute the registered model id
# and served name so mockers and proxies advertise one card and the policy keys
# on the same routing partition.
PROXY_X_CONFIG="$RUN_DIR/proxy-x.yaml"
PROXY_Y_CONFIG="$RUN_DIR/proxy-y.yaml"
POLICY_CONFIG="$RUN_DIR/policy.yaml"
render_config() {
    sed -e "s|\${MODEL_PATH}|$MODEL_PATH|g" \
        -e "s|\${MODEL}|$MODEL|g" \
        -e "s|\${HOSTED_BLOCKS}|$HOSTED_BLOCKS|g" \
        -e "s|\${BLOCK_SIZE}|$BLOCK_SIZE|g" \
        -e "s|\${CONTEXT_LENGTH}|$CONTEXT_LENGTH|g" "$1" >"$2"
}
render_config "$E2E_DIR/config/proxy-x.yaml" "$PROXY_X_CONFIG"
render_config "$E2E_DIR/config/proxy-y.yaml" "$PROXY_Y_CONFIG"
render_config "$E2E_DIR/config/policy.yaml" "$POLICY_CONFIG"

if [ "$SKIP_BUILD" != "1" ]; then
    echo "building dw-proxy-worker (set SKIP_BUILD=1 to reuse $PROXY_BIN)"
    CARGO_TARGET_DIR="$CARGO_TARGET_DIR" cargo build -p dw-proxy-worker --manifest-path "$REPO_ROOT/Cargo.toml"
fi
if [ ! -x "$PROXY_BIN" ]; then
    echo "proxy binary not found: $PROXY_BIN" >&2
    exit 1
fi

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
start frontend python3 -m dynamo.frontend \
    --http-port "$FRONTEND_PORT" \
    --model-path "$FRONTEND_MODEL_PATH" \
    --router-mode kv \
    --router-policy-config "$POLICY_CONFIG" \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq \
    --router-track-active-blocks

echo "starting $HOSTED_WORKERS mocker hosted worker(s)"
start mocker python3 -m dynamo.mocker \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL" \
    --endpoint "dyn://dynamo.backend.generate" \
    --engine-type "$ENGINE_TYPE" \
    --max-model-len "$CONTEXT_LENGTH" \
    --num-workers "$HOSTED_WORKERS" \
    --num-gpu-blocks-override "$HOSTED_BLOCKS" \
    --block-size "$BLOCK_SIZE" \
    --max-num-seqs "$MAX_SEQS" \
    --speedup-ratio "$SPEEDUP" \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq

echo "starting proxy workers"
start proxy-x "$PROXY_BIN" --config "$PROXY_X_CONFIG"
start proxy-y "$PROXY_BIN" --config "$PROXY_Y_CONFIG"

echo "waiting for '$MODEL' to register (up to ${WORKER_WAIT}s)"
wait_for_workers

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

comparison_args=()
if [ -n "$BASELINE" ]; then
    comparison_args=(--baseline "$BASELINE" --tolerance "$TOLERANCE")
fi
report_status=0
python3 "$E2E_DIR/report.py" \
    --loadgen "$REPORT_DIR/loadgen.jsonl" \
    --provider-log "proxy-x=$REPORT_DIR/provider-x.jsonl" \
    --provider-log "proxy-y=$REPORT_DIR/provider-y.jsonl" \
    --tier-map "$E2E_DIR/config/tier-map.json" \
    --bin-seconds "$BIN_SECONDS" \
    --json "$REPORT_DIR/e2e-report.json" \
    --markdown "$REPORT_DIR/e2e-report.md" \
    "${comparison_args[@]+"${comparison_args[@]}"}" || report_status=$?

echo
cat "$REPORT_DIR/e2e-report.md"
echo
echo "logs:    $LOG_DIR"
echo "reports: $REPORT_DIR"

if [ "$loadgen_status" -ne 0 ] || [ "$report_status" -ne 0 ]; then
    exit 1
fi
