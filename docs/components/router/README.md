---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
subtitle: KV cache-aware router that picks workers by combined prefill and decode cost to maximize throughput and minimize latency.
---

<p align="left">
  <a href="./README.zh-CN.md" hreflang="zh-CN"><img src="../../assets/img/readme-zh-cn-link.svg" alt="简体中文" height="28" /></a>
</p>

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## Quick Start

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

For Kubernetes, set `DYN_ROUTER_MODE=kv` on the Frontend service. Token-input
models use event-driven or approximate prefix-cache state. Text-input chat and
completions models use backend-reported KV occupancy and queue depth instead.

| Argument | Default | Description |
|----------|---------|-------------|
| `--router-mode kv` | `round-robin` | Use token-aware KV routing for token input or reported-load KV routing for text input |
| `--load-aware` | disabled | Token-input preset for KV active-load routing without cache-reuse signals; implies `--router-mode kv` on the frontend |
| `--router-kv-overlap-score-credit` | `1.0` | Credit multiplier for device-local prefix overlap, from 0.0 to 1.0 |
| `--router-prefill-load-scale` | `1.0` | Scale adjusted prompt-side prefill load before adding decode blocks |
| `--router-kv-events` / `--no-router-kv-events` | `--router-kv-events` | Consume worker KV events, or fall back to approximate routing without events |
| `--router-queue-threshold` | `16.0` | Backpressure queue threshold; priority hints only reorder requests while this queue is non-empty |
| `--router-queue-policy` | `fcfs` | Queue scheduling policy: `fcfs` (tail TTFT), `wspt` (avg TTFT), or `lcfs` (comparison-only reverse ordering) |
| `--no-router-track-prefill-tokens` | disabled | Ignore prompt-side prefill tokens in router load accounting; useful for decode-only routing paths |

### Standalone Router

You can also run the KV router as a standalone service (without the Dynamo frontend). See the [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/) for more details.

For deployment modes and quick start steps, see the [Router Guide](router-guide.md). For CLI arguments and tuning guidelines, see [Configuration and Tuning](router-configuration.md). For A/B benchmarking, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

## Prerequisites and Limitations

**Requirements:**
- **Dynamic endpoints only**: Workers must use `register_model()` so the frontend can discover their model input type and runtime topology.
- Token-input workers receive pre-tokenized requests and publish KV events or opt into approximate cache-state prediction.
- Text-input chat and completions workers keep tokenization in the backend and publish per-rank KV occupancy and queue-depth reports.
- The standalone router remains a token-input component; reported-load text routing is embedded in the frontend.

**Multimodal Support:**
- **Image routing via multimodal hashes**: Supported in the documented TRT-LLM and vLLM router paths.
- **Other backend or modality combinations**: Check the backend-specific multimodal docs before relying on multimodal hash routing.

**Limitations:**
- Static endpoints are not supported with KV routing; use dynamic discovery so the router can track worker instances and KV cache state

For basic model registration without KV routing, use `--router-mode round-robin`, `--router-mode random`, `--router-mode least-loaded`, or `--router-mode device-aware-weighted` with both static and dynamic endpoints.

## Next Steps

- **[Router Guide](router-guide.md)**: Deployment modes, quick start, and page map
- **[Routing Concepts](router-concepts.md)**: Cost model and worker-selection behavior
- **[Router Filtering](router-filtering.md)**: Candidate eligibility, DP-rank filtering, and busy-threshold overload handling
- **[Configuration and Tuning](router-configuration.md)**: Router flags, transport modes, and metrics
- **[Deficit Round Robin Queue Scheduling](deficit-round-robin.md)**: Weighted policy-class arbitration, cursor movement, and bulk virtual rounds
- **[Priority Scheduling](priority-scheduling.md)**: Router queue, backend engine, and cache priority behavior
- **[Disaggregated Serving](router-disaggregated-serving.md)**: Prefill and decode routing setups
- **[Router Operations](router-operations.md)**: Replicas, persistence, and recovery
- **[Router Examples](router-examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Testing](router-testing.md)**: Test layers from Rust unit tests to fixture-backed replay and full process E2E
- **[Standalone Indexer](standalone-indexer.md)**: Run the KV indexer as a separate service for independent scaling
- **[Standalone Selection Service](standalone-selection.md)**: Expose KV-aware selection and reservation accounting over HTTP
- **[Standalone Slot Tracker](standalone-slot-tracker.md)**: Run active-request load accounting as a separate HTTP service
- **[Router Design](../../design-docs/router-design.md)**: Architecture details, algorithms, and event transport modes
