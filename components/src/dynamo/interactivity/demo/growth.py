# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run on the host: register replicas, then reverse load on the CPU demo."""

import asyncio
import collections
import time

from registration import main, say, state


async def wait_for(predicate, description, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = await state()
        if predicate(snapshot["workers"]):
            return snapshot["workers"]
        await asyncio.sleep(0.2)
    raise AssertionError(f"Timed out: {description}; last state: {snapshot}")


async def reverse_load(session, initial):
    ids = {w["worker_id"] for w in initial["workers"]}
    assert collections.Counter(w["pool"] for w in initial["workers"]) == {
        "interactive": 1,
        "throughput": 2,
    }
    for pool, cap in [("interactive", 2), ("throughput", 8)]:
        workers = await wait_for(
            lambda ws: all(w["occupied"] == 0 for w in ws), "idle fleet"
        )
        home = next(w for w in workers if w["pool"] == pool)
        assert sum(w["pool"] == pool for w in workers) == 1
        say(f"LOAD {pool}: fill {home['stable_id']} to {cap}/{cap}; hold 18 seconds")

        async def request(rank):
            async with session.post(
                "http://127.0.0.1:18000/v1/chat/completions",
                headers={
                    "x-dynamo-interactivity-pool": pool,
                    "x-dynamo-worker-instance-id": str(home["worker_id"]),
                    "x-dynamo-dp-rank": str(rank),
                },
                json={
                    "model": "pool-demo",
                    "messages": [{"role": "user", "content": "demo"}],
                    "max_tokens": 180,
                    "stream": False,
                },
            ) as response:
                return response.status, await response.json()

        jobs = []
        try:
            for i in range(cap):
                jobs.append(asyncio.create_task(request(i % 2)))
                await wait_for(
                    lambda ws: next(
                        w for w in ws if w["worker_id"] == home["worker_id"]
                    )["reported_occupied"]
                    == i + 1,
                    f"{pool} request {i + 1} running",
                    timeout=5,
                )
            moved = await wait_for(
                lambda ws: sum(w["pool"] == pool for w in ws) == 2
                and all(w["target"] is None for w in ws),
                f"second worker reclassified to {pool}",
            )
            assert {w["worker_id"] for w in moved} == ids
            before = {w["worker_id"]: w["pool"] for w in workers}
            for w in moved:
                if before[w["worker_id"]] != w["pool"]:
                    say(
                        f"RECLASSIFIED {w['stable_id']}: "
                        f"{before[w['worker_id']]} -> {w['pool']}"
                    )
            say(
                f"PASS load-driven split: {dict(collections.Counter(w['pool'] for w in moved))}; "
                "worker identities unchanged"
            )
            for code, body in await asyncio.gather(*jobs):
                assert code == 200, (code, body)
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
    await wait_for(lambda ws: all(w["occupied"] == 0 for w in ws), "final idle fleet")
    say("PASS live growth: bypass -> 1/1 -> 1I/2T -> 2I/1T -> 1I/2T")


if __name__ == "__main__":
    asyncio.run(main(after_join=reverse_load))
