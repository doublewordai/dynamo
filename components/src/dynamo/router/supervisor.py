# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish all Ready DGD generations to one long-lived standalone router.

Usage: python -m dynamo.router.supervisor GRAPH COMPONENT BASE BLOCK_SIZE [router flags]
Requires read-only access to the graph and its component Deployments.
"""

import json
import os
import signal
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def ready_runtime_namespaces(graph, component, base, deployments):
    """Return every owned generation with Ready workers, retaining old capacity."""
    statuses = graph.get("status", {}).get("components", {})
    status = (
        next((c for c in statuses if c.get("name") == component), {})
        if isinstance(statuses, list)
        else statuses.get(component, {})
    )
    names = set(status.get("componentNames", []))
    candidates = []
    for deployment in deployments:
        metadata = deployment.get("metadata", {})
        if metadata.get("name") not in names or metadata.get("deletionTimestamp"):
            continue
        if deployment.get("status", {}).get("readyReplicas", 0) < 1:
            continue
        labels = metadata.get("labels", {})
        if labels.get("nvidia.com/dynamo-namespace") != base:
            continue
        suffix = labels.get("nvidia.com/dynamo-worker-hash")
        if not suffix:
            continue
        containers = (
            deployment.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for container in containers:
            env = {e["name"]: e.get("value") for e in container.get("env", [])}
            # Labels alone do not establish the actual discovery namespace.
            if env.get("DYN_NAMESPACE") not in (base, "$(DW_PLANE_PREFIX)" + base):
                continue
            if env.get("DYN_NAMESPACE_WORKER_SUFFIX") != suffix:
                continue
            candidates.append(base + "-" + suffix)
    return sorted(set(candidates))


def main():
    graph, component, base, block_size, *router_args = sys.argv[1:]
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("BLOCK_SIZE must be positive")
    directory = tempfile.TemporaryDirectory(prefix="dynamo-generations-")
    membership = Path(directory.name) / "generations.json"
    root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    context = ssl.create_default_context(cafile=str(root / "ca.crt"))
    url = (
        f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:443/apis/nvidia.com/v1beta1/"
        f"namespaces/{os.environ['POD_NAMESPACE']}/dynamographdeployments/{graph}"
    )
    deployment_url = (
        f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:443/apis/apps/v1/"
        f"namespaces/{os.environ['POD_NAMESPACE']}/deployments/"
    )

    def read_json(target):
        request = urllib.request.Request(
            target,
            headers={"Authorization": "Bearer " + (root / "token").read_text().strip()},
        )
        with urllib.request.urlopen(request, context=context, timeout=10) as response:
            return json.load(response)

    stopping = False
    child = None
    active = None

    def stop(*_):
        nonlocal stopping
        stopping = True

    def terminate():
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=310)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            try:
                graph_status = read_json(url)
                components = graph_status.get("status", {}).get("components", {})
                status = (
                    next((c for c in components if c.get("name") == component), {})
                    if isinstance(components, list)
                    else components.get(component, {})
                )
                deployments = []
                for name in status.get("componentNames", []):
                    try:
                        deployments.append(read_json(deployment_url + name))
                    except urllib.error.HTTPError as error:
                        if error.code != 404:
                            raise
                namespaces = ready_runtime_namespaces(
                    graph_status, component, base, deployments
                )
                if namespaces != active:
                    snapshot = {
                        "block_size": block_size,
                        "endpoints": [f"{ns}.backend.generate" for ns in namespaces],
                    }
                    pending = membership.with_suffix(".tmp")
                    pending.write_text(json.dumps(snapshot))
                    pending.replace(membership)
                    print(f"LocalRouter {base} generations: {namespaces}", flush=True)
                    active = namespaces
                if child is None:
                    env = {**os.environ, "DYN_NAMESPACE": base}
                    env.pop("DYN_NAMESPACE_WORKER_SUFFIX", None)
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "dynamo.router",
                            "--endpoint",
                            f"{base}.backend.generate",
                            "--router-block-size",
                            str(block_size),
                            "--worker-generations-file",
                            str(membership),
                            "--router-replica-sync",
                            *router_args,
                        ],
                        env=env,
                    )
                if child and child.poll() is not None:
                    raise SystemExit(child.returncode or 1)
            except (OSError, ValueError) as error:
                print(
                    f"Waiting for worker discovery: {type(error).__name__}", flush=True
                )
            for _ in range(10):
                if stopping:
                    break
                time.sleep(1)
    finally:
        terminate()
        directory.cleanup()


if __name__ == "__main__":
    main()
