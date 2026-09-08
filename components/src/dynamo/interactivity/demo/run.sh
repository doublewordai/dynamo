#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
K=(kubectl --context microk8s --namespace dynamo-interactivity-demo)
# Reset frontend-owned memberships. Run while this demo's workers are idle.
"${K[@]}" rollout restart deployment/frontend
"${K[@]}" rollout status deployment/frontend --timeout=180s
demo_module=$("${K[@]}" exec deployment/frontend -- python -c 'import json; c=json.load(open("/etc/pools/pools.json")); print("two_models" if isinstance(c,list) else "traffic")')
"${K[@]}" exec deployment/frontend -- python -m "dynamo.interactivity.demo.$demo_module"
