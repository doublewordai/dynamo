# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU integration: real HTTP frontend, RPC router, token workers and KV events.

Run with RUN_ROUTER_GENERATIONS_E2E=1, ETCD_ENDPOINTS and NATS_SERVER pointing
at isolated test services, and the freshly built Dynamo packages on PYTHONPATH.
"""

import asyncio
import collections
import json
import os
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu_0, pytest.mark.pre_merge]


async def serve_worker(directory, namespace, label):
    from dynamo.llm import (
        KvEventPublisher,
        ModelInput,
        ModelRuntimeConfig,
        ModelType,
        WorkerType,
        register_model,
    )
    from dynamo.runtime import DistributedRuntime

    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "nats")
    endpoint = runtime.endpoint(f"{namespace}.backend.generate")
    config = ModelRuntimeConfig()
    config.total_kv_blocks = 1024
    config.max_num_seqs = 32
    config.kv_event_publishing_enabled = True
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        str(directory / "model"),
        "probe",
        kv_cache_block_size=2,
        runtime_config=config,
        worker_type=WorkerType.Aggregated,
    )
    publisher = KvEventPublisher(endpoint, kv_block_size=2)
    old = label.startswith("old")

    async def publish():
        # Distinct real KV event prefixes in each generation; no assumed cache.
        while True:
            publisher.publish_stored(
                [1 if old else 4] * 16, [2] * 8, list(range(100, 108))
            )
            await asyncio.sleep(0.5)

    async def generate(request):
        with (directory / "traffic.jsonl").open("a") as output:
            output.write(json.dumps({"worker": label}) + "\n")
        count = request.get("stop_conditions", {}).get("max_tokens", 1)
        for i in range(count):
            if count > 1:
                await asyncio.sleep(0.1)
            yield {
                "token_ids": [2 if old else 3],
                "finish_reason": "stop" if i == count - 1 else None,
            }

    task = asyncio.create_task(publish())
    try:
        await endpoint.serve_endpoint(generate)
    finally:
        task.cancel()
        runtime.shutdown()


async def serve_gateway(directory, namespace, pool):
    from dynamo.llm import (
        ModelInput,
        ModelRuntimeConfig,
        ModelType,
        WorkerType,
        register_model,
    )
    from dynamo.runtime import DistributedRuntime

    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "nats")
    endpoint = runtime.endpoint(f"{namespace}.backend.generate")
    config = ModelRuntimeConfig()
    config.total_kv_blocks = 1024
    config.max_num_seqs = 32
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        str(directory / "model"),
        "probe",
        kv_cache_block_size=2,
        runtime_config=config,
        worker_type=WorkerType.Aggregated,
    )
    client = await runtime.endpoint(f"{pool}.router.generate").client()

    async def generate(request):
        stream = await client.generate(request)
        async for response in stream:
            yield response.data()

    await endpoint.serve_endpoint(generate)


@pytest.mark.skipif(
    os.environ.get("RUN_ROUTER_GENERATIONS_E2E") != "1",
    reason="requires isolated etcd/NATS and a built Dynamo runtime",
)
@pytest.mark.asyncio
async def test_generation_rollout_keeps_kv_locality_and_spills_load(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from dynamo.runtime import DistributedRuntime

    model = tmp_path / "model"
    model.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "old": 2, "new": 3, "fresh": 4, "cold": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model / "tokenizer.json"))
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 6,
                "max_position_embeddings": 1024,
                "eos_token_id": 0,
            }
        )
    )
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "model_max_length": 1024,
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        )
    )
    pool = "rollout-" + uuid.uuid4().hex[:8]
    public = pool + "-public"
    membership = tmp_path / "generations.json"
    processes = []
    results = []
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "nats")

    def start(name, arguments, **env):
        log = (tmp_path / (name + ".log")).open("w")
        child = subprocess.Popen(
            [sys.executable, *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "DYN_NAMESPACE": pool,
                "DYN_REQUEST_PLANE": "nats",
                "DYN_SYSTEM_PORT": "-1",
                "DYN_SELF_HOST_METADATA": "0",
                **env,
            },
        )
        log.close()
        processes.append(child)
        return child

    def update(names):
        pending = membership.with_suffix(".tmp")
        pending.write_text(
            json.dumps(
                {
                    "block_size": 2,
                    "endpoints": [f"{pool}-{name}.backend.generate" for name in names],
                }
            )
        )
        pending.replace(membership)

    async def wait_until(function, timeout=30):
        async def poll():
            while True:
                try:
                    value = await function()
                    if value:
                        return value
                except (OSError, RuntimeError):
                    pass
                await asyncio.sleep(0.2)

        return await asyncio.wait_for(poll(), timeout)

    async def instance_count(client, count):
        return len(client.instance_ids()) == count

    # Bind port selection only for the test; frontend owns the actual listener.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def http(path, body=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}" + path,
            data=json.dumps(body).encode() if body else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    async def batch(prompt, expected, count=20):
        texts = []
        for _ in range(count):
            response = await asyncio.to_thread(
                http,
                "/v1/completions",
                {
                    "model": "probe",
                    "prompt": (prompt + " ") * 16,
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            texts.append(response["choices"][0]["text"].strip())
        counts = dict(collections.Counter(texts))
        results.append({"prompt": prompt, "counts": counts})
        assert counts == {expected: count}, (results, tmp_path)

    try:
        old = [
            start(
                f"old{i}", [__file__, "worker", str(tmp_path), pool + "-old", f"old{i}"]
            )
            for i in range(3)
        ]
        old_client = await runtime.endpoint(f"{pool}-old.backend.generate").client()
        await wait_until(lambda: instance_count(old_client, 3))
        update(["old"])
        router = start(
            "router",
            [
                "-m",
                "dynamo.router",
                "--endpoint",
                f"{pool}.backend.generate",
                "--router-block-size",
                "2",
                "--worker-generations-file",
                str(membership),
                "--router-temperature",
                "0",
                "--router-decode-active-request-weight",
                "100",
            ],
        )
        start("gateway", [__file__, "gateway", str(tmp_path), public, pool])
        start(
            "frontend",
            [
                "-m",
                "dynamo.frontend",
                "--namespace",
                public,
                "--http-host",
                "127.0.0.1",
                "--http-port",
                str(port),
                "--request-plane",
                "nats",
                "--router-mode",
                "round-robin",
            ],
        )

        async def models_ready():
            return (await asyncio.to_thread(http, "/v1/models"))["data"]

        await wait_until(models_ready, 60)
        await asyncio.sleep(2)
        await batch("hello", "old")
        start("new0", [__file__, "worker", str(tmp_path), pool + "-new", "new0"])
        new_client = await runtime.endpoint(f"{pool}-new.backend.generate").client()
        await wait_until(lambda: instance_count(new_client, 1))
        update(["old", "new"])
        await asyncio.sleep(3)
        await batch("hello", "old")
        await batch("fresh", "new")
        # Exercise the stable RPC endpoint for explicit pins and long streams.
        client = await runtime.endpoint(f"{pool}.router.generate").client()
        await client.wait_for_instances()
        ingress_ids = client.instance_ids()
        streams = []
        for worker_id in old_client.instance_ids():
            stream = await client.generate(
                {
                    "model": "probe",
                    "token_ids": [1] * 16,
                    "stop_conditions": {"max_tokens": 80},
                    "sampling_options": {},
                    "output_options": {},
                    "routing": {"backend_instance_id": worker_id},
                }
            )
            first = await anext(stream)
            assert first.data()["token_ids"] == [2]
            streams.append(stream)
        # All warm workers carry load: the cold replacement now wins on cost.
        await batch("hello", "new", count=10)
        # Withdraw old generation while its existing streams still run.
        update(["new"])
        await asyncio.sleep(2)
        await batch("hello", "new")

        async def finish(stream):
            return [x.data() async for x in stream]

        tails = await asyncio.gather(*(finish(s) for s in streams))
        assert all(
            len(tail) == 79 and tail[-1]["finish_reason"] == "stop" for tail in tails
        )
        # Reintroduce old generation, then replace workers one by one.
        update(["old", "new"])
        await asyncio.sleep(3)
        await batch("hello", "old")
        for remaining in [2, 1, 0]:
            old[remaining].terminate()
            await asyncio.to_thread(old[remaining].wait, 15)
            await wait_until(
                lambda remaining=remaining: instance_count(old_client, remaining)
            )
            if remaining:
                start(
                    f"new{3 - remaining}",
                    [
                        __file__,
                        "worker",
                        str(tmp_path),
                        pool + "-new",
                        f"new{3 - remaining}",
                    ],
                )
                await wait_until(
                    lambda remaining=remaining: instance_count(
                        new_client, 4 - remaining
                    )
                )
            update(["old", "new"] if remaining else ["new"])
            await asyncio.sleep(2)
            await batch("hello", "old" if remaining else "new")
            await batch("fresh", "new")
        # A torn/invalid controller update preserves serving state.
        membership.write_text("{")
        await asyncio.sleep(2)
        await batch("fresh", "new")
        assert router.poll() is None
        assert client.instance_ids() == ingress_ids
        (tmp_path / "results.json").write_text(json.dumps(results, indent=2))
        print("generation rollout evidence:", tmp_path, results)
    finally:
        for child in reversed(processes):
            if child.poll() is None:
                child.terminate()
                try:
                    await asyncio.to_thread(child.wait, 15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    await asyncio.to_thread(child.wait)
        runtime.shutdown()


if __name__ == "__main__":
    kind, directory, namespace, label = sys.argv[1:]
    function = serve_worker if kind == "worker" else serve_gateway
    asyncio.run(function(Path(directory), namespace, label))
