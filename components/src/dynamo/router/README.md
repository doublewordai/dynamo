<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Standalone Router

A backend-agnostic standalone KV-aware router service for Dynamo deployments. For details on how KV-aware routing works, see [Routing Concepts](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md).

## Overview

The standalone router provides configurable KV-aware routing for any set of workers in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing to prefill workers), multi-tier architectures, or any scenario requiring intelligent KV cache-aware routing decisions.

This component is **fully configurable** and works with any Dynamo backend (vLLM, TensorRT-LLM, SGLang, etc.) and any worker endpoint.

## Usage

### Command Line

```bash
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --router-block-size 64 \
    --no-router-track-active-blocks
```

### Arguments

**Required:**
- `--endpoint`: Full endpoint path for workers in the format `namespace.component.endpoint` (e.g., `dynamo.prefill.generate`)

**Router Configuration:**
Most KV tuning options use the `--router-*` prefix, but shared options such as
`--load-aware`, `--serve-indexer`, `--use-remote-indexer`, and `--shared-cache-*` do
not. Standalone-only options include `--endpoint` and `--router-block-size`. Legacy
names such as `--block-size` and `--kv-events` are still accepted but deprecated.
Run `python -m dynamo.router --help` for the standalone command surface. The
[Frontend Configuration Reference](../../../../docs/fern/pages/reference/components/frontend-configuration.mdx#router)
is the canonical reference for shared embedded-router flags and environment variables;
see [Configuration and Tuning](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md) for
behavioral guidance.

## Architecture

The standalone router exposes three endpoints via the Dynamo runtime:

1. **`generate`**: Routes requests to the best worker and streams back generation results (KV-aware routing).
2. **`best_worker_id`**: Given token IDs, returns the best worker ID for the request without routing; useful for debugging or custom routing logic.
3. **`get_overlap_scores`**: Given token IDs, returns per-worker/per-DP-rank matched block counts for device, host-pinned, disk, and configured shared-cache tiers without routing.

Clients call the `generate` endpoint to stream completions, call `best_worker_id` to decide which worker to use and then contact that worker directly, or call `get_overlap_scores` when an external scheduler wants the raw tiered overlap signal.

## Example: Manual Disaggregated Serving (Alternative Setup)

> [!Note]
> **This is an alternative advanced setup.** The recommended approach for disaggregated serving is to use the frontend's automatic prefill routing, which activates when you register workers with `WorkerType.Prefill`. See [Disaggregated Serving](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md) for the default setup.
>
> Use this manual setup if you need explicit control over prefill routing configuration or want to manage prefill and decode routers separately.

For an integrated frontend disaggregated example, see [`examples/backends/vllm/launch/disagg_router.sh`](/examples/backends/vllm/launch/disagg_router.sh). For explicit multi-router composition, see the [Global Router README](../global_router/README.md).

```bash
# Start frontend router for decode workers
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-kv-overlap-score-credit 0  # Pure load balancing for decode

# Start standalone router for prefill workers
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --router-block-size 64 \
    --no-router-track-active-blocks

# Start decode workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 &

# Start prefill workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 --disaggregation-mode prefill &
```

For event-driven prefix-cache state, add the backend-specific KV event publishing flags to the workers that the router indexes. Use `--no-router-kv-events` on the router only when approximate cache-state prediction is acceptable.

>[!Note]
> **Why `--no-router-track-active-blocks` for prefill routing?**
> Active block tracking is used for load balancing across decode (generation) phases. For prefill-only routing, decode load is not relevant, so disabling this reduces overhead and simplifies the router state.
>
> **When should I use `--no-router-track-prefill-tokens`?**
> Use it on decode-only routers that should ignore already-completed prompt work. This keeps `active_prefill_tokens`, queue pressure, and load estimates focused on decode-side work after a prefill-to-decode handoff.
>
> **Why `--router-block-size` should be set for standalone routers:**
> Standalone routers default to block size `128`, but they do not infer block size from the ModelDeploymentCard (MDC) during worker registration. Set the value explicitly so routing decisions match the backend worker block size.

## Configuration Best Practices

>[!Note]
> **Block Size Matching:**
> The block size must match across:
> - Standalone router (`--router-block-size`)
> - All worker instances (backend-specific, e.g. `--block-size` for vLLM)
>
> **Endpoint Matching:**
> The `--endpoint` argument must match where your target workers register. For example:
> - vLLM prefill workers: `dynamo.prefill.generate`
> - vLLM decode workers: `dynamo.backend.generate`
> - Custom workers: `<your_namespace>.<your_component>.<your_endpoint>`

## Integration with Backends

To integrate the standalone router with a backend:

1. Workers should register at the endpoint specified by the `--endpoint` argument
2. Clients call the `router.generate` endpoint to stream completions (router selects the best worker), or call `router.best_worker_id` to get the best worker ID and then send requests to that worker
3. Router state is updated automatically as requests are routed; no separate "free" call is required

See [`components/src/dynamo/vllm/handlers.py`](../vllm/handlers.py) for a reference implementation (search for `prefill_router_client`).

## See Also

- [Router Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/router-guide.md) - Deployment modes and quick start
- [Configuration and Tuning](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md) - CLI flags, transport modes, and metrics
- [Disaggregated Serving](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md) - Prefill and decode routing setups
- [Router Design](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/router-design.md) - Architecture details and event transport modes
- [Frontend Router](../frontend/README.md) - Main HTTP frontend with integrated routing
- [Router Benchmarking](../../../../benchmarks/router/README.md) - Performance testing and tuning

## Rolling worker generations

A standalone router normally targets one exact worker endpoint. A supervisor
that changes its worker namespace to the newest Ready Deployment cuts off the
old generation even when most of the old workers still serve. Use a membership
snapshot to keep all live generations available through the same router endpoint:

```json
{
  "block_size": 256,
  "endpoints": [
    "pool-oldhash.backend.generate",
    "pool-newhash.backend.generate"
  ]
}
```

```bash
DYN_NAMESPACE=pool python -m dynamo.router \
  --endpoint pool.backend.generate \
  --router-block-size 256 \
  --worker-generations-file /run/router/generations.json \
  --router-replica-sync
```

The rollout controller must write the whole JSON document to a temporary file
and atomically rename it over the snapshot. The router reloads it every second.
Unchanged generations retain their router and warm KV index. New generations get
independent clients, KV indexes and schedulers. Removing a generation stops new
selection into it; existing streams keep its state until they finish. An empty
list deliberately withdraws all generations. Invalid or unreadable updates retain
the previous serving table and log an error. Neither membership updates nor the
arrival of the first replacement worker restart the public router endpoint.

Selection previews the native KV-aware scheduler in every eligible generation.
It compares the returned KV-overlap/load costs without reserving work in the
losing generations, then dispatches through the winning generation's ordinary
scheduler, including eligibility revalidation, admission, replica synchronization
and request cleanup. Equal-cost generations are sampled in proportion to their
eligible discovered worker counts. Cache locality can outweigh that proportion;
load can outweigh locality. This is not a fixed traffic split. Preview is advisory,
so concurrent requests can change load before final worker selection.

All listed endpoints must belong to `<base>-<generation>` namespaces and have
the same component and endpoint as `--endpoint`. They must serve the same
preprocessed-request contract: model/tokenizer, compatible context limits, worker
role, KV block size and scheduling configuration. The snapshot's block size must
match the CLI; the controller is responsible for ensuring it matches the actual
workers. This mode does not compare unlike block sizes or independent cost
models. It does not merge cache entries across generations. External/shared
indexer service modes are rejected because each generation needs its own index.
Different generations of one pool are supported; joint optimization of a
prefill/decode pair across pools is outside this mode.

Explicit worker pins and allowlists restrict generation candidates before
preview. If all eligible generations are overloaded, the selected generation's
normal scheduler handles queuing; admission is never bypassed. Selection does
not retry a stream after output starts. The `best_worker_id` endpoint previews
across generations. In this mode `get_overlap_scores` returns a `generations`
map from endpoint path to the native overlap response, preserving each
generation's shared-cache diagnostics separately. Single-endpoint mode keeps
its existing response format.

### Kubernetes companion

For a CPU-side router outside the GPU graph, the packaged companion can own the
membership file and keep one router child running:

```bash
python -m dynamo.router.supervisor \
  graph-name Worker pool 256 --router-temperature 0
```

Its arguments are the DynamoGraphDeployment name, component name, stable Dynamo
namespace and block size, followed by optional router flags. It requires
`POD_NAMESPACE`, the standard Kubernetes service environment and service-account
mount, and read-only `get` permission for that graph and its component
Deployments. It checks component ownership, worker hash and namespace environment,
and includes **every** owned generation with Ready workers. API errors preserve
the previous snapshot. The Kubernetes poll interval is ten seconds; actual
worker eligibility and load continue to come from Dynamo discovery/scheduling.

To adopt this in an existing deployment, build an image containing this change
and replace the previous namespace-switching supervisor command with the module
above, retaining the graph/component/base/block-size arguments. Updating the
image alone while keeping the old supervisor will not fix the cutoff. No worker
wire-format change is required. Rollback is the previous image and supervisor
command; that restores the previous single-generation cutover behavior.

### CPU integration test

Build the Python binding and install the Python components, then point the test
at **isolated** etcd and NATS services:

```bash
RUN_ROUTER_GENERATIONS_E2E=1 \
ETCD_ENDPOINTS=http://127.0.0.1:2379 NATS_SERVER=nats://127.0.0.1:4222 \
python -m pytest -xvs components/src/dynamo/router/tests/test_generations_e2e.py
```

This starts real frontend, router and CPU token-worker processes. Workers publish
real KV events for different prefixes. It checks cache locality in both
generations, spillover under active-request load, pinned streams finishing after
membership withdrawal, one-by-one replacement, invalid snapshot recovery and
stable ingress identity. It loads no model weights and requires no GPU.
