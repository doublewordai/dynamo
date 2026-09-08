# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run inside the frontend pod. Fail unless real HTTP and worker state prove each case."""

import asyncio
import json
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout

NODES = ("worker-a", "worker-b", "worker-c")


def say(message):
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


async def main():
    async with ClientSession(timeout=ClientTimeout(total=150)) as session:

        async def states():
            try:
                snapshot = json.loads(Path("/tmp/pool-state.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
            result = {w["stable_id"]: w for w in snapshot["workers"]}
            if len(result) != len(NODES):
                return {}
            for node in NODES:
                async with session.get(f"http://{node}:8081/status") as response:
                    response.raise_for_status()
                    result[node].update(await response.json())
            return result

        async def wait_for(predicate, description, timeout=45):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                value = await states()
                if value and predicate(value):
                    return value
                await asyncio.sleep(0.2)
            raise AssertionError(f"Timed out: {description}; state={value}")

        async def request(pool, tokens=1, node=None, rank=None):
            headers = {"x-dynamo-interactivity-pool": pool}
            if node:
                headers["x-dynamo-worker-instance-id"] = str(
                    (await states())[node]["worker_id"]
                )
            if node and rank is None:
                rank = 0  # Dynamo requires a rank when pinning a multi-rank worker.
            if rank is not None:
                headers["x-dynamo-dp-rank"] = str(rank)
            body = {
                "model": "pool-demo",
                "messages": [{"role": "user", "content": "demo"}],
                "max_tokens": tokens,
                "stream": False,
                "nvext": {"extra_fields": ["worker_id"]},
            }
            async with session.post(
                "http://frontend:8000/v1/chat/completions", json=body, headers=headers
            ) as response:
                data = await response.json()
                return response.status, data

        async def expect(pool, code, node=None):
            actual, body = await request(pool, node=node)
            assert actual == code, (pool, actual, body)
            if code == 503:
                assert "interactivity_capacity" in json.dumps(body), body
            say(
                f"HTTP {actual} class={pool} target={node or 'automatic'} "
                f"{json.dumps(body.get('nvext', {}) if code == 200 else body)}"
            )

        async def hold(node, pool, count, tokens):
            # Reserve sequentially, then overlap generation. Check worker state
            # rather than assuming a timer means requests have reached admission.
            tasks = []
            for _ in range(count):
                before = (await states())[node]["actual_occupied"]
                task = asyncio.create_task(
                    request(pool, tokens=tokens, node=node, rank=before % 2)
                )
                tasks.append(task)
                await wait_for(
                    lambda s: s[node]["actual_occupied"] >= before + 1,
                    f"admission on {node}",
                    5,
                )
            return tasks

        async def finished(tasks):
            for code, body in await asyncio.gather(*tasks):
                assert code == 200, (code, body)
            await wait_for(
                lambda s: all(r["occupied"] == 0 for r in s.values()),
                "all requests complete",
            )

        initial = await wait_for(
            lambda s: all(w["healthy"] for w in s.values()),
            "fresh telemetry for every DP rank",
        )
        initial_interactive = next(
            n for n in NODES if initial[n]["pool"] == "interactive"
        )
        initial_throughput = [n for n in NODES if initial[n]["pool"] == "throughput"]
        assert len(initial_throughput) == 2, "Reset the frontend before rerunning."
        assert all(
            r["occupied"] == 0 for r in initial.values()
        ), "Workers must start idle"
        incarnations = {n: initial[n]["incarnation"] for n in NODES}
        say(
            "PHASE 1: home-pool routing; interactive cap=2, throughput cap=8 per worker"
        )
        await expect("interactive", 200)
        await expect("throughput", 200)
        await expect("unknown", 400)
        await asyncio.sleep(0.5)

        say("PHASE 2: fill interactive A (2/2); throughput B and C each hold 3/8")
        tasks = await hold(initial_interactive, "interactive", 2, 180)
        tasks += await hold(initial_throughput[0], "throughput", 3, 180)
        tasks += await hold(initial_throughput[1], "throughput", 3, 180)
        await asyncio.sleep(0.3)  # allow the next worker telemetry sample
        say(
            "Interactive must reject: every worker is at or above its class cap of 2. Throughput still fits."
        )
        await expect("interactive", 503)
        await expect("throughput", 200, initial_throughput[1])
        draining = await wait_for(
            lambda s: any(r["target"] == "interactive" for r in s.values()),
            "automatic drain",
        )
        donor = next(n for n, r in draining.items() if r["target"] == "interactive")
        say(
            f"PHASE 3: frontend chose {donor} to drain toward interactive; new requests to it must reject"
        )
        await expect("throughput", 503, donor)
        await finished(tasks)
        changed = await wait_for(
            lambda s: s[donor]["pool"] == "interactive", "reclassification after drain"
        )
        say(
            f"RECLASSIFIED {donor}: throughput -> interactive, cap={changed[donor]['effective_cap']}; same process/incarnation"
        )
        assert all(changed[n]["incarnation"] == incarnations[n] for n in NODES)
        await expect("interactive", 200, donor)

        say(
            "PHASE 4: reverse pressure: fill the remaining throughput worker to 7/8 (>80%)"
        )
        throughput = next(n for n, r in changed.items() if r["pool"] == "throughput")
        tasks = await hold(throughput, "throughput", 7, 180)
        reversed_state = await wait_for(
            lambda s: sum(r["pool"] == "throughput" for r in s.values()) == 2,
            "reverse reclassification",
        )
        promoted = next(
            n
            for n, r in reversed_state.items()
            if r["pool"] == "throughput" and n != throughput
        )
        say(
            f"RECLASSIFIED {promoted}: interactive -> throughput, cap={reversed_state[promoted]['effective_cap']}"
        )
        assert all(reversed_state[n]["incarnation"] == incarnations[n] for n in NODES)
        await expect("throughput", 200, promoted)
        await finished(tasks)

        say(
            "PHASE 5: borrowed interactive request lowers a throughput node's effective cap to 2"
        )
        tasks = await hold(promoted, "interactive", 1, 50)
        borrowed = (
            await wait_for(lambda s: s[promoted]["effective_cap"] == 2, "borrowed cap")
        )[promoted]
        assert borrowed["pool"] == "throughput" and borrowed["effective_cap"] == 2
        say(
            f"BORROWED {promoted}: node class remains throughput, cap={borrowed['effective_cap']}, local requests={borrowed['local_occupied']}"
        )
        tasks += await hold(promoted, "throughput", 1, 40)
        await expect("throughput", 503, promoted)
        await finished(tasks)

        say("PHASE 6: saturate all current pools; both request classes must reject")
        current = await states()
        tasks = []
        for node, replica in current.items():
            tasks += await hold(node, replica["pool"], replica["effective_cap"], 160)
        await asyncio.sleep(0.3)
        await expect("interactive", 503)
        await expect("throughput", 503)
        await finished(tasks)
        say(
            "PASS: home routing, class-specific rejection, drain rejection, automatic reclassification in both directions, borrowing protection, and full-capacity rejection. Worker-wide budgets span two DP ranks. No worker restarted."
        )


if __name__ == "__main__":
    asyncio.run(main())
