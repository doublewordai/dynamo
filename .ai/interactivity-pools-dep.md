<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DEP draft: Frontend-owned interactivity pools

## Summary

Add two concurrency classes to the Rust frontend's view of a discovered fleet. Use discovered worker topology and existing worker metrics, combine observed occupancy with local outstanding requests, filter the KV/load-aware router's candidates, and reject requests when no eligible capacity remains. Workers publish telemetry but do not enforce classes.

## Motivation

Interactive and throughput workloads need different concurrency operating points on the same warm fleet. For this version, occasional oversubscription and stale classification decisions are acceptable; an authoritative worker gate is unnecessary complexity.

## Proposal

Support typed request tags and a frontend configuration of class caps and minima. Start without registered workers, bypass pool routing with one worker, and assign arrivals to the smaller class (ties use the request default). Stable IDs are optional. A worker-wide budget covers every DP rank. Preserve per-rank engine ceilings and existing routing constraints. Try home workers first, then borrow with the smaller class cap. Track local requests until the frontend response guard drops. Extend the existing worker metrics payload with optional running-plus-waiting request count. Expire cached observations and exclude missing or stale ranks.

Run rebalancing within the Rust frontend. Sustained recipient pressure above 80%, donor headroom, minimum memberships, fresh discovered-fleet telemetry, and cooldown authorize one local drain. Stop routing to that worker, wait for estimated zero occupancy, then change its local class. Timeout cancels the move. No worker mutation, model reload, engine configuration change, Python controller, or Kubernetes scaling occurs.

## Consistency and alternatives

Each frontend has independent classifications and accounting. Observations and local dispatch deltas are estimates. Multiple frontends, cancellation, restart, and delayed observations can oversubscribe or disagree during a drain. This is the explicitly accepted best-effort contract. TTLs constrain stale information but do not serialize admission. A future worker gate or shared coordinator could provide stricter enforcement if operational experience justifies it.

## Scope and validation

Initial scope: exactly two classes, dynamic worker registration and removal, aggregated single-sequence text generation, router queueing disabled. CPU replay uses the actual Rust frontend and ordinary telemetry-only dummy workers with two ranks each. Unit tests cover DP-wide caps, borrowing, telemetry reconciliation and expiry, and reclassification. The demo checks routing, HTTP rejection, bidirectional moves, and unchanged worker incarnations.

This is a local draft; no upstream DEP has been filed.
