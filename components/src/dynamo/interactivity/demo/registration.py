# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run on the local host: exercise zero/one/many workers on MicroK8s only."""

import asyncio
import collections
import json
import time

from aiohttp import ClientError, ClientSession, ClientTimeout

K = ["kubectl", "--context", "microk8s", "-n", "dynamo-interactivity-demo"]


def say(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


async def k(*args):
    proc = await asyncio.create_subprocess_exec(
        *K, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(err.decode())
    return out.decode()


async def state(component="worker"):
    return json.loads(
        await k(
            "exec",
            "deployment/frontend",
            "--",
            "cat",
            f"/tmp/pool-state.json.pooldemo.{component}.generate",
        )
    )


async def wait_count(count):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            s = await state()
            if len(s["workers"]) == count and all(w["healthy"] for w in s["workers"]):
                split = collections.Counter(w["pool"] for w in s["workers"])
                if count < 2 or (
                    abs(split["interactive"] - split["throughput"]) <= 1
                    and all(w["target"] is None for w in s["workers"])
                ):
                    assert s["pool_routing_enabled"] == (count > 1)
                    say(
                        f"REGISTERED {count}: split={dict(split)} routing_enabled={s['pool_routing_enabled']}"
                    )
                    return s
        except (RuntimeError, json.JSONDecodeError):
            pass
        await asyncio.sleep(0.3)
    raise AssertionError(f"Fleet did not settle at {count} workers")


async def scale(name, replicas):
    pods = json.loads(await k("get", "pods", "-l", f"app={name}", "-o", "json"))[
        "items"
    ]
    await k("scale", f"deployment/{name}", f"--replicas={replicas}")
    if replicas == 0:
        # Workers are idle at each removal; SIGINT closes their Dynamo runtime.
        for pod in pods:
            try:
                await k(
                    "exec",
                    pod["metadata"]["name"],
                    "--",
                    "python",
                    "-c",
                    "import os,signal; os.kill(1,signal.SIGINT)",
                )
            except RuntimeError:
                pass
    else:
        await k("rollout", "status", f"deployment/{name}", "--timeout=60s")


async def main(after_join=None):
    async with ClientSession(timeout=ClientTimeout(total=20)) as session:

        async def request(tokens=1):
            async with session.post(
                "http://127.0.0.1:18000/v1/chat/completions",
                headers={"x-dynamo-interactivity-pool": "interactive"},
                json={
                    "model": "pool-demo",
                    "messages": [{"role": "user", "content": "demo"}],
                    "max_tokens": tokens,
                    "stream": False,
                },
            ) as response:
                return response.status, await response.json()

        for name in ["worker-a", "worker-b", "worker-c"]:
            await scale(name, 0)
        await k(
            "wait",
            "--for=delete",
            "pod",
            "-l",
            "app in (worker-a,worker-b,worker-c)",
            "--timeout=60s",
        )
        await k("rollout", "restart", "deployment/frontend")
        await k("rollout", "status", "deployment/frontend", "--timeout=60s")
        say(
            "Frontend started with no workers for pool-demo and no configured assignments"
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                async with session.get("http://127.0.0.1:18000/v1/models") as response:
                    models = {m["id"] for m in (await response.json())["data"]}
                if "pool-demo" not in models and "pool-demo-2" in models:
                    break
            except (ClientError, KeyError):
                pass
            await asyncio.sleep(0.5)
        else:
            raise AssertionError("Frontend did not start with the first fleet absent")
        await scale("worker-a", 1)
        one = await wait_count(1)
        assert one["workers"][0]["pool"] == "throughput"
        baseline_other = [
            (w["worker_id"], w["pool"]) for w in (await state("worker2"))["workers"]
        ]
        # Four real interactive requests exceed the pool cap of two. All must run
        # when there is a single worker, even though it exposes two DP ranks.
        jobs = [asyncio.create_task(request(60)) for _ in range(4)]
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (await state())["workers"][0]["reported_occupied"] >= 4:
                    break
                await asyncio.sleep(0.15)
            else:
                raise AssertionError(
                    "Single worker did not run four requests concurrently"
                )
            for code, body in await asyncio.gather(*jobs):
                assert code == 200, (code, body)
        finally:
            for task in jobs:
                task.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        say(
            "PASS singleton bypass: four simultaneous interactive requests succeeded above class cap 2"
        )
        await scale("worker-b", 1)
        await wait_count(2)
        jobs = [asyncio.create_task(request(60)) for _ in range(4)]
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if sum(w["reported_occupied"] for w in (await state())["workers"]) == 4:
                    break
                await asyncio.sleep(0.15)
            else:
                raise AssertionError(
                    "Two workers did not fill their interactive capacity"
                )
            code, body = await request()
            assert code == 503 and "interactivity_capacity" in json.dumps(body), (
                code,
                body,
            )
            for code, body in await asyncio.gather(*jobs):
                assert code == 200, (code, body)
        finally:
            for task in jobs:
                task.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        say(
            "PASS two-worker activation: fifth interactive request rejected after four active requests"
        )
        await scale("worker-c", 1)
        three = await wait_count(3)
        if after_join is not None:
            await after_join(session, three)
        else:
            minority = next(
                w["pool"]
                for w in three["workers"]
                if sum(x["pool"] == w["pool"] for x in three["workers"]) == 1
            )
            victim = next(
                w["stable_id"] for w in three["workers"] if w["pool"] == minority
            )
            say(
                f"Removing sole {minority} worker {victim}; remaining two must rebalance to 1/1"
            )
            await scale(victim, 0)
            await wait_count(2)
            await scale(victim, 1)
            await wait_count(3)
        current_other = [
            (w["worker_id"], w["pool"]) for w in (await state("worker2"))["workers"]
        ]
        assert current_other == baseline_other
        say("PASS: registration and pool transitions verified; other model unchanged")


if __name__ == "__main__":
    asyncio.run(main())
