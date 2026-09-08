# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Follow decision logs, reconnecting when this demo's pods are replaced."""

import asyncio
import json
import re
import signal

KUBECTL = ["kubectl", "--context", "microk8s", "-n", "dynamo-interactivity-demo"]
SELECTOR = "app in (frontend,worker-a,worker-b,worker-c,worker2-a,worker2-b,worker2-c)"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
EVENTS = re.compile(
    r"Pool worker|Pool fleet|Pool routing|Frontend pool state|Pool drain|Pool reclassified|START node=|FINISH node=|READY node=|ERROR|Traceback|Pool status unavailable"
)


async def follow(pod):
    process = await asyncio.create_subprocess_exec(
        *KUBECTL,
        "logs",
        "-f",
        pod,
        "--tail=12",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async for raw in process.stdout:
            line = ANSI.sub("", raw.decode(errors="replace")).strip()
            if EVENTS.search(line):
                # Drop repeated HTTP span fields after the routing decision.
                line = re.split(r"\s+method=POST", line)[0]
                print(f"[{pod}] {line}", flush=True)
        await process.wait()
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


async def main():
    followers = {}
    try:
        while True:
            process = await asyncio.create_subprocess_exec(
                *KUBECTL,
                "get",
                "pods",
                "-l",
                SELECTOR,
                "-o",
                "json",
                stdout=asyncio.subprocess.PIPE,
            )
            output, _ = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError("Could not discover demo pods")
            pods = {
                p["metadata"]["name"]
                for p in json.loads(output)["items"]
                if p["status"]["phase"] == "Running"
                and not p["metadata"].get("deletionTimestamp")
            }
            for pod in followers.keys() - pods:
                task = followers.pop(pod)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for pod in pods - followers.keys():
                followers[pod] = asyncio.create_task(follow(pod))
            await asyncio.sleep(2)
    finally:
        for task in followers.values():
            task.cancel()
        await asyncio.gather(*followers.values(), return_exceptions=True)


async def entrypoint():
    task = asyncio.create_task(main())
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(entrypoint())
