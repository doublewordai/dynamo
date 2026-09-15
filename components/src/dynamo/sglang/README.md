<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

### Multinode planner telemetry

SGLang emits forward-pass metrics from attention-TP leaders on the final
pipeline stage. The adapter subscribes to those exact local DP ranks, resolves
the routable leader by worker-group identity, and attributes non-leader samples
to that leader. This gives Planner all 16 decode DP ranks for TP16/DP16 across
four nodes, and all four prefill DP ranks on the final stage for PP4/TP4/DP4.
The relay preserves scheduler counters and payload fields. FPM-only leader
resolution retries after a cold-start timeout without terminating GPU ranks.
