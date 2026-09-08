# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the isolated CPU demo as Kubernetes JSON on stdout."""

import argparse
import json
from pathlib import Path

NAMESPACE = "dynamo-interactivity-demo"


def manifests(image, two_models=False):
    objects = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NAMESPACE}}
    ]
    config = json.loads(Path(__file__).with_name("pools.json").read_text())
    if two_models:
        second = json.loads(json.dumps(config))
        second["endpoint"] = "pooldemo.worker2.generate"
        second["pools"]["interactive"]["cap"] = 3
        second["pools"]["throughput"]["cap"] = 6
        config = [config, second]
    objects.append(
        dict(
            apiVersion="v1",
            kind="ConfigMap",
            metadata=dict(name="pools", namespace=NAMESPACE),
            data={"pools.json": json.dumps(config)},
        )
    )
    env = {
        "ETCD_ENDPOINTS": "http://etcd:2379",
        "NATS_SERVER": "nats://nats:4222",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_RUNTIME_NUM_WORKER_THREADS": "2",
        "DYN_RUNTIME_MAX_BLOCKING_THREADS": "4",
        "DYN_COMPUTE_THREADS": "2",
        "DYN_SYSTEM_PORT": "9090",
        "DYN_ROUTER_USE_KV_EVENTS": "false",
        "DYN_DISABLE_FRONTEND_NVEXT": "false",
        "DYN_LOG": "warn,dynamo_llm::kv_router::push_router::interactivity=debug",
    }

    def add(
        name,
        container_image,
        command,
        port,
        *,
        runtime=False,
        extra_env=None,
        probe_path=None,
    ):
        labels = {"app": name, "demo": "interactivity-pools"}
        container = dict(
            name=name,
            image=container_image,
            imagePullPolicy="IfNotPresent",
            command=command,
            resources={
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "2", "memory": "2Gi"},
            },
        )
        if runtime:
            container["env"] = [
                dict(name=k, value=v) for k, v in (env | (extra_env or {})).items()
            ]
        if name == "frontend":
            container["volumeMounts"] = [
                dict(name="pools", mountPath="/etc/pools", readOnly=True)
            ]
        container["readinessProbe"] = (
            dict(httpGet=dict(path=probe_path, port=port))
            if probe_path
            else dict(tcpSocket=dict(port=port))
        ) | dict(periodSeconds=2, failureThreshold=60)
        pod = dict(containers=[container], terminationGracePeriodSeconds=130)
        if name == "frontend":
            pod["volumes"] = [dict(name="pools", configMap=dict(name="pools"))]
        objects.append(
            dict(
                apiVersion="apps/v1",
                kind="Deployment",
                metadata=dict(name=name, namespace=NAMESPACE),
                spec=dict(
                    replicas=1,
                    strategy=dict(type="Recreate"),
                    selector=dict(matchLabels={"app": name}),
                    template=dict(metadata=dict(labels=labels), spec=pod),
                ),
            )
        )
        objects.append(
            dict(
                apiVersion="v1",
                kind="Service",
                metadata=dict(name=name, namespace=NAMESPACE),
                spec=dict(
                    selector={"app": name}, ports=[dict(port=port, targetPort=port)]
                ),
            )
        )

    add(
        "etcd",
        "quay.io/coreos/etcd:v3.5.21",
        [
            "/usr/local/bin/etcd",
            "--listen-client-urls=http://0.0.0.0:2379",
            "--advertise-client-urls=http://etcd:2379",
        ],
        2379,
    )
    add("nats", "nats:2.10.26-alpine", ["nats-server", "-js"], 4222)
    fleets = [("worker", "pool-demo")]
    if two_models:
        fleets.append(("worker2", "pool-demo-2"))
    for component, model in fleets:
        for suffix in "abc":
            add(
                f"{component}-{suffix}",
                image,
                ["python", "-m", "dynamo.interactivity.demo.worker"],
                8081,
                runtime=True,
                extra_env={
                    "WORKER_ID": f"worker-{suffix}",
                    "WORKER_ENDPOINT": f"pooldemo.{component}.generate",
                    "MODEL_NAME": model,
                    "DYN_WORKER_METRICS_HEARTBEAT_SECS": "1",
                },
                probe_path="/status",
            )
    add(
        "frontend",
        image,
        [
            "python",
            "-m",
            "dynamo.frontend",
            "--router-mode",
            "kv",
            "--http-port",
            "8000",
            "--http-host",
            "0.0.0.0",
            "--namespace",
            "pooldemo",
        ],
        8000,
        runtime=True,
        probe_path="/v1/models",
        extra_env={
            "DYN_FRONTEND_INTERACTIVITY_CONFIG": "/etc/pools/pools.json",
            "DYN_FRONTEND_INTERACTIVITY_STATUS_PATH": "/tmp/pool-state.json",
        },
    )
    return dict(apiVersion="v1", kind="List", items=objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="localhost:32000/dynamo-interactivity:local")
    parser.add_argument("--two-models", action="store_true")
    args = parser.parse_args()
    print(json.dumps(manifests(args.image, args.two_models), indent=2))
