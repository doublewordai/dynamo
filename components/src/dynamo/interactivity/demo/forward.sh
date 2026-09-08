#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
forward_pid=
bind_addresses=${DEMO_BIND_ADDRESSES:-127.0.0.1}
cleanup() {
  if [[ -n "$forward_pid" ]]; then
    kill "$forward_pid" 2>/dev/null || true
    wait "$forward_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 0' INT TERM
# Some kubectl versions keep the listener open after its target pod disappears.
# Check HTTP as well as the process, and reattach after run.sh replaces the pod.
while true; do
  kubectl --context microk8s -n dynamo-interactivity-demo port-forward --address "$bind_addresses" service/frontend 18000:8000 &
  forward_pid=$!
  sleep 2
  while kill -0 "$forward_pid" 2>/dev/null && curl -fsS --max-time 2 http://127.0.0.1:18000/v1/models >/dev/null; do
    sleep 2
  done
  cleanup
  forward_pid=
  sleep 2
done
