#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download tokenizer.json for the production model families into the proxy-core test
# fixtures. The files are large and gitignored; the retokenizer tests skip when absent.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/tests/fixtures/tokenizers"

# family/repo. If a repo is gated, swap in the nearest public repo of the same tokenizer.
repos=(
  "glm-5/zai-org/GLM-5.3"
  "deepseek-v4/deepseek-ai/DeepSeek-V4.1-Flash"
  "qwen3/Qwen/Qwen3-8B"
)

for entry in "${repos[@]}"; do
  family="${entry%%/*}"
  repo="${entry#*/}"
  out="$dest/$family/tokenizer.json"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    echo "already present: $family ($repo)"
    continue
  fi
  echo "downloading $repo -> $family/tokenizer.json"
  curl --fail --location --retry 3 --retry-delay 2 \
    "https://huggingface.co/$repo/resolve/main/tokenizer.json" -o "$out"
done

echo "done: $dest"
