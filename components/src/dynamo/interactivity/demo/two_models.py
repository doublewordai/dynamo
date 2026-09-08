# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove independent pools for two models sharing one frontend and class names."""

import asyncio
import json
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout

FLEETS = {"pool-demo": "worker", "pool-demo-2": "worker2"}


def say(message):
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


async def main():
    tasks = []
    async with ClientSession(timeout=ClientTimeout(total=120)) as session:

        async def states():
            result = {}
            for model, component in FLEETS.items():
                path = Path(f"/tmp/pool-state.json.pooldemo.{component}.generate")
                if not path.exists():
                    return {}
                snapshot = json.loads(path.read_text())
                workers = {w["stable_id"]: w for w in snapshot["workers"]}
                if len(workers) != 3 or not all(w["healthy"] for w in workers.values()):
                    return {}
                for suffix in "abc":
                    async with session.get(
                        f"http://{component}-{suffix}:8081/status"
                    ) as response:
                        response.raise_for_status()
                        workers[f"worker-{suffix}"].update(await response.json())
                result[model] = workers
            return result

        async def wait_for(predicate, description, timeout=30):
            deadline = time.monotonic() + timeout
            value = {}
            while time.monotonic() < deadline:
                value = await states()
                if value and predicate(value):
                    return value
                await asyncio.sleep(0.15)
            raise AssertionError(f"{description}: {value}")

        async def request(model, pool, tokens=1, worker=None, rank=0):
            headers = {"x-dynamo-interactivity-pool": pool}
            if worker:
                current = await states()
                headers.update(
                    {
                        "x-dynamo-worker-instance-id": str(
                            current[model][worker]["worker_id"]
                        ),
                        "x-dynamo-dp-rank": str(rank),
                    }
                )
            async with session.post(
                "http://frontend:8000/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "demo"}],
                    "max_tokens": tokens,
                    "stream": False,
                    "nvext": {"extra_fields": ["worker_id"]},
                },
            ) as response:
                return response.status, await response.json()

        async def expect(model, pool, code, worker=None):
            actual, body = await request(model, pool, worker=worker)
            assert actual == code, (model, pool, actual, body)
            if code == 503:
                assert "interactivity_capacity" in json.dumps(body), body
            if code == 200:
                selected = body["nvext"]["worker_id"]["decode_worker_id"]
                current = await states()
                assert selected in {
                    w["worker_id"] for w in current[model].values()
                }, body
            say(
                f"HTTP {code} model={model} class={pool} target={worker or 'automatic'}"
            )

        async def hold(model, worker, pool, count, tokens=180):
            group = []
            for i in range(count):
                before = (await states())[model][worker]["actual_occupied"]
                task = asyncio.create_task(request(model, pool, tokens, worker, i % 2))
                tasks.append(task)
                group.append(task)
                await wait_for(
                    lambda s: s[model][worker]["actual_occupied"] == before + 1,
                    f"request admitted on {model}/{worker}",
                    5,
                )
            return group

        def classes(workers):
            return {
                name: (w["pool"], w["target"], w["effective_cap"])
                for name, w in workers.items()
            }

        async def idle(model):
            return await wait_for(
                lambda s: all(
                    w["occupied"] == 0 and w["actual_occupied"] == 0
                    for w in s[model].values()
                ),
                f"{model} idle",
            )

        async def finish(group, model):
            for code, body in await asyncio.gather(*group):
                assert code == 200, (code, body)
            await idle(model)

        try:
            initial = await wait_for(
                lambda s: all(
                    w["occupied"] == 0 for fleet in s.values() for w in fleet.values()
                ),
                "both fleets ready",
            )
            for fleet in initial.values():
                assert (
                    sum(w["pool"] == "interactive" for w in fleet.values()) == 1
                ), "Restart the idle frontend before rerunning."
            incarnations = {
                (model, name): w["incarnation"]
                for model, fleet in initial.items()
                for name, w in fleet.items()
            }
            ids = [
                {w["worker_id"] for w in fleet.values()} for fleet in initial.values()
            ]
            assert ids[0].isdisjoint(ids[1])
            say(
                "Two independent models, same class names and stable worker IDs, one frontend; two DP ranks per worker"
            )
            for model in FLEETS:
                await expect(model, "interactive", 200)
                await expect(model, "throughput", 200)
                await idle(model)

            for model, other in [
                ("pool-demo", "pool-demo-2"),
                ("pool-demo-2", "pool-demo"),
            ]:
                current = await states()
                baseline = classes(current[other])
                home = next(
                    name
                    for name, w in current[model].items()
                    if w["pool"] == "interactive"
                )
                cap = current[model][home]["effective_cap"]
                say(
                    f"PRESSURE {model}: fill interactive worker to {cap}/{cap}; {other} must stay unchanged"
                )
                group = await hold(model, home, "interactive", cap)
                await expect(model, "interactive", 503, home)
                changed = await wait_for(
                    lambda s: sum(w["pool"] == "interactive" for w in s[model].values())
                    == 2,
                    f"{model} reclassification",
                )
                assert classes(changed[other]) == baseline
                assert all(
                    w["occupied"] == 0 and w["local_occupied"] == 0
                    for w in changed[other].values()
                )
                moved = next(
                    name
                    for name, w in changed[model].items()
                    if w["pool"] == "interactive"
                    and current[model][name]["pool"] != "interactive"
                )
                say(
                    f"ISOLATED MOVE {model}: {moved} throughput -> interactive; {other} membership and accounting unchanged"
                )
                await expect(other, "interactive", 200)
                await finish(group, model)
                await idle(other)

            for model, other in [
                ("pool-demo", "pool-demo-2"),
                ("pool-demo-2", "pool-demo"),
            ]:
                current = await states()
                baseline = classes(current[other])
                say(
                    f"SATURATE {model}: fill every worker across both ranks; {other} must keep serving"
                )
                group = []
                # Fill the larger cap first, then low caps before sustained pressure can move a worker.
                ordered = sorted(
                    current[model].items(), key=lambda pair: -pair[1]["effective_cap"]
                )
                for worker, w in ordered:
                    group += await hold(
                        model, worker, w["pool"], w["effective_cap"], tokens=140
                    )
                await expect(model, "interactive", 503)
                await expect(model, "throughput", 503)
                await expect(other, "interactive", 200)
                await expect(other, "throughput", 200)
                await idle(other)
                unchanged = await states()
                assert classes(unchanged[other]) == baseline
                assert all(w["local_occupied"] == 0 for w in unchanged[other].values())
                say(
                    f"ISOLATED CAPACITY {model}: rejects both classes; {other}: accepts both classes"
                )
                await finish(group, model)

            final = await states()
            assert all(
                w["incarnation"] == incarnations[model, name]
                for model, fleet in final.items()
                for name, w in fleet.items()
            )
            say(
                "PASS: two models independently classify, account, rebalance and reject; no cross-model routing or worker restarts; no worker gates"
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
