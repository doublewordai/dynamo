#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
DEMO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(git -C "$DEMO_DIR" rev-parse --show-toplevel)
BUILD_DIR=${DEMO_BUILD_DIR:-/tmp/dynamo-interactivity-image}
WHEEL_DIR=${DEMO_WHEEL_DIR:-/tmp/dynamo-interactivity-wheels}
IMAGE=${DEMO_IMAGE:-localhost:32000/dynamo-interactivity:local}
# Build the modified native bindings first using maturin; no stock Dynamo wheel.
mkdir -p "$BUILD_DIR/wheels" "$BUILD_DIR/components/src"
cp "$WHEEL_DIR"/*.whl "$BUILD_DIR/wheels/"
python3 - "$BUILD_DIR/components/src/dynamo" <<'PYBUILD'
import shutil
import sys
shutil.rmtree(sys.argv[1], ignore_errors=True)
PYBUILD
cp -a "$REPO_DIR/components/src/dynamo" "$BUILD_DIR/components/src/"
cp "$DEMO_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"
docker build -t "$IMAGE" "$BUILD_DIR"
docker push "$IMAGE"
