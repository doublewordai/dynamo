# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sustained single-class traffic against both independent local model fleets."""

import asyncio
import collections
import json
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout

FLEETS = {"pool-demo": "worker", "pool-demo-2": "worker2"}
CAPS = {
    "pool-demo": {"interactive": 2, "throughput": 8},
    "pool-demo-2": {"interactive": 3, "throughput": 6},
}


def say(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def state(model):
    endpoint = f"pooldemo.{FLEETS[model]}.generate"
    return json.loads(Path(f"/tmp/pool-state.json.{endpoint}").read_text())["workers"]


async def exercise(session, model, pool, seconds=40):
    stop = asyncio.Event()
    counts = collections.Counter()
    clients = []
    reached = None

    async def client():
        while not stop.is_set():
            async with session.post(
                "http://frontend:8000/v1/chat/completions",
                headers={"x-dynamo-interactivity-pool": pool},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "demo"}],
                    "max_tokens": 100,
                    "stream": False,
                },
            ) as response:
                body = await response.json()
                assert response.status in (200, 503), (response.status, body)
                counts[response.status] += 1
                if response.status == 503:
                    assert "interactivity_capacity" in json.dumps(body)
                    await asyncio.sleep(0.25)

    start = time.monotonic()
    last = -1
    say(
        f"START model={model} requests={pool} only; initial={dict(collections.Counter(w['pool'] for w in state(model)))}"
    )
    try:
        while time.monotonic() - start < seconds:
            workers = state(model)
            split = collections.Counter(w["pool"] for w in workers)
            assert split["interactive"] >= 1 and split["throughput"] >= 1, split
            if split[pool] == 2 and reached is None:
                reached = time.monotonic()
            # Keep the home pool busy, with one additional request able to borrow.
            # Increase offered concurrency when its membership grows.
            target = split[pool] * CAPS[model][pool] + 1
            if len(clients) < target:
                clients.append(asyncio.create_task(client()))
            for task in clients:
                if task.done():
                    task.result()
            elapsed = int(time.monotonic() - start)
            if elapsed // 5 != last:
                last = elapsed // 5
                say(
                    f"LOAD model={model} class={pool} elapsed={elapsed}s split={dict(split)} occupied={sum(w['occupied'] for w in workers)} clients={len(clients)} completed={dict(counts)}"
                )
            await asyncio.sleep(0.15)
        assert reached is not None, f"{model} never reached two {pool} workers"
        assert (
            time.monotonic() - reached >= 15
        ), "Insufficient observation at the membership limit"
    finally:
        stop.set()
        results = await asyncio.gather(*clients, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    split = collections.Counter(w["pool"] for w in state(model))
    assert split[pool] == 2 and sum(split.values()) == 3, split
    say(
        f"PASS model={model} class={pool} only: final={dict(split)} completed={dict(counts)}; opposite pool retained"
    )


async def main():
    async with ClientSession(timeout=ClientTimeout(total=30)) as session:
        original = {}
        for component in FLEETS.values():
            for suffix in "abc":
                async with session.get(
                    f"http://{component}-{suffix}:8081/status"
                ) as response:
                    original[component, suffix] = (await response.json())["incarnation"]
        for phase in [
            (("pool-demo", "interactive"), ("pool-demo-2", "throughput")),
            (("pool-demo", "throughput"), ("pool-demo-2", "interactive")),
        ]:
            await asyncio.gather(
                *(exercise(session, model, pool) for model, pool in phase)
            )
            await asyncio.sleep(1)
        for (component, suffix), incarnation in original.items():
            async with session.get(
                f"http://{component}-{suffix}:8081/status"
            ) as response:
                assert (await response.json())["incarnation"] == incarnation
        say(
            "PASS: repeated single-class demand tested in both directions on both models; neither fleet became a single pool; no worker restarts"
        )


if __name__ == "__main__":
    asyncio.run(main())
