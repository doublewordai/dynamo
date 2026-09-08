<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CPU frontend pool demonstration

The isolated `dynamo-interactivity-demo` namespace contains etcd, NATS, the modified Rust Dynamo frontend, and three CPU dummy workers. Each worker exposes two DP ranks and publishes actual running counts plus synthetic KV metrics. Workers have no pool configuration, gate, reservation endpoint, or classification logic. There is no controller pod.

Build the modified Python bindings with maturin, putting the wheel in `/tmp/dynamo-interactivity-wheels`, then:

```bash
DEMO_IMAGE=localhost:32000/dynamo-interactivity:frontend-v1 ./components/src/dynamo/interactivity/demo/build.sh
python components/src/dynamo/interactivity/demo/manifests.py --image localhost:32000/dynamo-interactivity:frontend-v1 | kubectl --context microk8s apply -f -
./components/src/dynamo/interactivity/demo/run.sh
```

Use a new image tag for each rebuild. The build context is disposable and must contain the current components only.

`run.sh` resets this demo’s frontend membership, waits for fresh rank telemetry, and runs the appropriate HTTP replay inside the frontend pod. Run it while the workers are idle. It checks home routing, worker-wide DP budgets, class-specific rejection, rejection during draining, reclassification in both directions, borrowed interactive protection, full saturation, and unchanged worker process identities during moves. Successful requests generate CPU tokens; no GPU or model weights are required.

```bash
python3 components/src/dynamo/interactivity/demo/watch.py
./components/src/dynamo/interactivity/demo/forward.sh
```

The watcher shows frontend routing, cached state, draining, and class changes alongside ordinary worker START/FINISH events. The frontend writes `/tmp/pool-state.json` inside its pod for the replay to inspect. Worker `/status` endpoints report only actual counts and process identity. The latter are demo diagnostics, not an admission protocol.

The example uses interactive cap 2, throughput cap 8 **per worker across both ranks**, 80% threshold, four seconds of sustained pressure, five-second cooldown and telemetry TTL, and 90-second drain timeout. These timings are chosen to make decisions visible. The two ranks are CPU simulations, not a benchmark of GPU DP-attention performance.

Each frontend makes independent best-effort decisions. The replay validates one frontend's behavior and does not claim a hard multi-frontend concurrency guarantee.

## Two independent model fleets

Render with `manifests.py --two-models --image <rebuilt-image>` to run one frontend with six workers. `pool-demo` uses `pooldemo.worker.generate` with caps 2/8; `pool-demo-2` uses `pooldemo.worker2.generate` with caps 3/6. Both fleets deliberately reuse the stable IDs `worker-a`, `worker-b`, `worker-c` and the class names `interactive` and `throughput`.

After applying the manifest and waiting for all workers, restart only the idle frontend to restore initial memberships, then run:

```bash
kubectl --context microk8s -n dynamo-interactivity-demo exec deployment/frontend -- python -m dynamo.interactivity.demo.two_models
```

The replay loads and rebalances each model separately, checks that the other fleet's membership and accounting remain unchanged, then saturates each model and verifies that the other still accepts both classes. It verifies every successful probe routes within the requested model's fleet and checks that no worker restarted. This mode uses endpoint-suffixed diagnostic files; use `two_models` instead of the single-model `traffic` driver.

## Sustained single-class demand

With the two-model demo running and idle, run the sustained demand test from the checkout:

```bash
kubectl --context microk8s -n dynamo-interactivity-demo exec -i deployment/frontend -- python - < components/src/dynamo/interactivity/demo/single_class.py
```

It repeatedly sends only one class per model for 40 seconds, then reverses the classes for another 40 seconds, allowing in-flight work to finish between phases. Offered concurrency grows with the requested pool plus one borrowing request. It checks that both fleets reach two workers in the requested class, retain at least one worker in the opposite class, and preserve worker incarnations. This uses the two-model demo's configured 2/8 and 3/6 caps. Both pool minima are one; this policy therefore cannot reclassify all three workers into one pool.

## Automatic worker registration

`registration.py` runs on the local host against the MicroK8s context. With the two-model demo deployed, it scales the first model to zero, restarts the frontend with no assignments, then brings workers back individually. It checks one-worker bypass with four simultaneous interactive requests, reactivation at two workers with a capacity rejection, balanced joins, deletion repair, and rejoin. The other model remains unchanged. Run while both model fleets are idle:

```bash
python3 components/src/dynamo/interactivity/demo/registration.py
```

The test restores three workers for the first model on success. No worker names appear in the pool configuration; the test names Kubernetes deployments only to control arrival and departure events.

## Live growth and demand reversal

With the two-model demo idle and `forward.sh` forwarding the frontend to local port 18000, run on a host with `aiohttp` installed:

```bash
python3 components/src/dynamo/interactivity/demo/growth.py
```

This restarts the frontend and registers the first model's workers individually. It verifies singleton bypass, activation at two workers, and a third worker joining the smaller class. Sustained interactive load changes the initial 1 interactive / 2 throughput split to 2 / 1; switching to throughput load changes it back to 1 / 2. Worker identities stay unchanged during those moves, and the other model's memberships remain unchanged.
