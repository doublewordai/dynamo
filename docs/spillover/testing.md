<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing spillover before production

This is the plan for trusting spillover with production traffic. Each stage has a
different job, so the list says what the stage proves, what exists today, and what is
still missing.

## What we need to prove

1. **Correct answers.** A request routed to a proxy returns what the client would have got
   from a primary worker: the same text, tool calls, reasoning, usage, finish reason and
   limits, or a clear error.
2. **Correct routing.** Traffic stays on primary workers while they have room, keeps
   conversations where their cache is, and spills to the cheapest acceptable tier when
   primary fills.
3. **Safe failure.** Provider errors, rate limits, stalls and outages move requests to other
   workers without duplicate output, without taking healthy workers out of routing, and
   without leaks.
4. **Operability.** On-call can see why traffic spills, which provider is failing and what
   it costs, and can roll out or roll back one model at a time.

## Stage 1: unit tests (every PR, `spillover` workflow)

Exists:

- Policy: tier costs, parameter validation, and equivalence of the baseline scorer and
  picker with Dynamo's default selector across a fuzzed grid.
- Chat request in core: the snapshot is never serialized, only workers advertising
  `chat_request` receive it, media fields are cleared together, replays are marked, and
  non-object `extra_args` are left alone.
- Proxy: the request fields forwarded or refused, provider SSE parsing and error
  classification, timeouts, renderers round-tripped through the frontend's own parsers for
  each model family, the retokenizer against real tokenizers, the virtual cache, KV events,
  model-card parity, error mapping (only a rejected key reports the proxy down), and
  metrics.

Missing:

- Equivalence of the unseeded picker production constructs, at temperature 0 with tied
  costs and eight or more workers (review s11-2).
- A dispatch-level test that runs a real `RoutingHost` dispatch to a proxy and to a primary
  worker and inspects what each receives.

## Stage 2: simulation (every PR, `routing-sim` scenarios)

Proves routing behaviour under load shapes we cannot easily create for real: ramps to many
times primary capacity, primary outages, provider rate limits, stickiness, admission margins.

Exists: eight scenarios with assertions, and sweeps for tuning.

Calibrated: the simulator's load signals follow the router's own accounting after the round-2
fixes, and the primary timing in `lib/spillover/e2e/config/level1-equivalent.yaml` was then
refitted to the mocker's `aisimulate-core` model and the measured run. The twin now predicts
50.4% primary / 49.0% proxy-x / 0.6% proxy-y with 69.8% class stickiness against the real
stack's 52.4% / 46.0% / 1.6% / 64.4% — inside the `report.py --baseline` band on every row.
Keep it calibrated with the nightly comparison in stage 3; the derivation of each primary
parameter is in that file's header comment.

Missing:

- A scenario where a primary worker fails mid-response and the retry goes through the
  policy, asserting the retry never lands on a proxy.
- A scenario where a long conversation's worker crosses its admission margin, asserting the
  policy spills instead of queueing.

## Stage 3: end to end with mocks (nightly, `spillover-nightly` workflow)

Proves the real frontend, KV router, policy, proxy workers and generator work together:
all workers form one worker set, the frontend hands proxies the chat request, spill follows
the arrival ramp, responses are tagged with the serving tier, and metrics carry the right
labels. It uses Dynamo's mocker for primary workers and a local fake provider.

Exists: `lib/spillover/e2e/run.sh` with absolute routing assertions (no failures, every
proxy response tagged, a minimum proxy share) and an optional comparison with the Level 1
twin. The last manual run served 700 requests with 0 failures.

Missing:

- A Python-to-Rust card checksum test: build the SGLang worker's router config from the
  Python argument defaults and assert the proxy's card checksum matches (review s11-5).
- Fault injection in the fake provider: 429 storms, 5xx, stalls before headers and mid-stream,
  truncated streams, moderation 403, a 401 on one provider.
- A kill of a primary mocker mid-response, asserting no duplicated output reaches the client.
- A frontend restart while proxies keep running, asserting the proxies' cache is recovered
  from their local indexer.

## Stage 4: staging with real engines and providers

Proves what mocks cannot: real SGLang engines, real provider APIs, and real model output.

Run one model with its SGLang workers and two real provider tiers, then:

1. **Fidelity.** Send the same prompts (plain chat, multi-turn, tools, parallel tools,
   reasoning on and off, JSON output, long context, images) pinned first to primary and then to
   each provider. Compare parsed output shape, usage, finish reasons and length limits.
2. **Routing.** Replay a recorded production traffic shape at 0.5x, 1x and 2x primary
   capacity. Check primary share, spill timing, conversation stickiness and provider spend
   against the simulation.
3. **Failure.** Revoke one provider's key, rate-limit a provider, kill a primary engine
   mid-response, restart the frontend. Check that requests move, nothing duplicates, and the
   proxy with the revoked key is reported down while the others stay in routing.
4. **Thinking dialect.** For every provider tier, send thinking on, off, an effort and a
   budget through its proxy and check the response reasons or not as asked. This confirms
   the tier's `thinking_dialect` before it takes traffic; in production
   `proxy_thinking_total{event="ignored"}` catches a provider that stops honouring it.
5. **Provider cache key.** For every provider tier, send the same multi-turn conversation with
   `cache_key` off and on, and compare the cached prompt tokens the provider reports. Turn it on
   only where it raises cache hits.
6. **Operations.** Walk the on-call questions with only dashboards and logs: why is traffic
   spilling, which provider is failing, what did it cost. Roll one model forward and back.

## Stage 5: load and soak

Run at two to three times primary capacity for several hours with primary restarts and
provider errors. Check that proxy memory, virtual-cache size and event buffers stay bounded,
the 529 rate stays within the SLA, no requests leak across migrations, and the admission
margin does not make routing flap.

## Production rollout

Start with a proxy-only staging model, then one production model with conservative settings
(spill late), watching the stage 4 dashboards, then the rest. Each model has its own policy
entry and proxies, so rollback is per model: remove its policy entry or scale its proxies to
zero.
