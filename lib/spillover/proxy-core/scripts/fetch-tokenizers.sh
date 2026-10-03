#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download tokenizer.json for the production model families, and the tiktoken files of the
# tiktoken families, into the proxy-core test fixtures. The files are large and gitignored;
# the retokenizer tests skip when absent.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/tests/fixtures/tokenizers"

# family/repo@revision. The revision is the repo's `main` commit at the time the fixture
# was last validated, not a moving branch, so a rerun reproduces the same tokenizer even
# if upstream changes. If a repo is gated, swap in the nearest public repo of the same
# tokenizer. To refresh a revision: `curl -s "https://huggingface.co/api/models/$repo" | jq -r .sha`.
repos=(
  "glm-5/zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42"
  "deepseek-v4/deepseek-ai/DeepSeek-V4.1-Flash@2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
  "qwen3/Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218"
)
# Same format; these ship tiktoken.model instead of tokenizer.json, and Dynamo's tiktoken
# loader also reads config.json (model_type) and tokenizer_config.json (special tokens).
tiktoken_repos=(
  "kimi-k3/moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06"
)

download() {
  local repo="$1" revision="$2" file="$3" out="$4"

  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    echo "already present: $out ($repo)"
    return
  fi
  echo "downloading $repo@$revision -> $out"
  # Download to a sibling temp file and move it into place only after curl succeeds, so an
  # interrupted download never leaves a partial file that later runs would skip.
  local tmp="$out.tmp.$$"
  if curl --fail --location --retry 3 --retry-delay 2 \
      --connect-timeout 15 --max-time 300 \
      "https://huggingface.co/$repo/resolve/$revision/$file" -o "$tmp"; then
    mv -f "$tmp" "$out"
  else
    rm -f "$tmp"
    echo "failed to download $repo@$revision/$file" >&2
    exit 1
  fi
}

for entry in "${repos[@]}"; do
  family="${entry%%/*}"
  rest="${entry#*/}"
  repo="${rest%@*}"
  revision="${rest##*@}"
  download "$repo" "$revision" tokenizer.json "$dest/$family/tokenizer.json"
done

for entry in "${tiktoken_repos[@]}"; do
  family="${entry%%/*}"
  rest="${entry#*/}"
  repo="${rest%@*}"
  revision="${rest##*@}"
  # tiktoken.model last: the tests take its presence to mean the family is complete.
  for file in config.json tokenizer_config.json tiktoken.model; do
    download "$repo" "$revision" "$file" "$dest/$family/$file"
  done
done

echo "done: $dest"
