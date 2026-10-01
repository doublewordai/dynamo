# doublewordai/dynamo fork layout

Integration fork of ai-dynamo/dynamo. Production frontend, router, planner
and worker images are built from a commit of this repo's `main` by
doublewordai/dynamo-images, and the image tag carries that commit.

## Branches

- `upstream-base`: exactly the upstream `main` commit the fork is based on.
  Currently `2e4998a30ccd` (2026-09-22). Never carries our commits. Upstream
  cuts releases on branches off `main`, so the base is a pinned `main`
  commit, not a release tag.
- `fork-base`: `upstream-base` plus `vendor/fork-ci`, the one commit that
  makes upstream's CI runnable in this fork (hosted-runner Rust tests,
  upstream-only jobs skipped). Patch branches are based on it and pull
  requests target it, so every pull request gets real Rust CI. Nothing
  else ever lands on it.
- `main`: `fork-base` plus every patch branch below, merged with `--no-ff`
  in stack order. `git log --merges fork-base..main` is the patch list.
- `upstream-pr/<topic>`: a change we intend to land upstream. Based on
  `fork-base`. This repo is a public fork, so the branch heads the upstream
  PR directly once rebased onto upstream `main`
  (`git rebase --onto upstream/main fork-base`, which drops the CI commit).
- `vendor/<topic>`: a Doubleword-only change that will not go upstream.
  Based on `fork-base`.
- `archive/*` and other branches are history and are not part of any image.

## Rules

- No backports. Do not cherry-pick upstream commits onto `main`; a fix that
  is on upstream `main` arrives by moving `upstream-base`.
- One branch per patch, atomic, with the reason in the commit message.
- Moving the base: point `upstream-base` at the new upstream `main` commit,
  rebase `vendor/fork-ci` onto it and rebuild `fork-base`, rebase each patch
  branch that is still needed onto `fork-base`, drop the ones upstream now
  contains, rebuild `main` as `fork-base` plus merges, force-push `main`,
  build images. Update the stack list below. The base moves on a cadence,
  not only when a fix is needed.

## Current stack

- `vendor/fork-layout`: this section.
- `vendor/fork-ci`: the `fork-base` CI commit (see Branches).
- `upstream-pr/vllm-kv-cache-group-worker-extension`: KV cache group
  metadata through a vLLM worker extension; vLLM 0.30 has no engine-core
  utility for it. Fork PR #184.
- `upstream-pr/backend-admission-policies`: tracked copy of upstream's
  backend admission policies (upstream pull 14369): worker-side
  concurrency limit, bounded overflow queue, Controlled Delay, adaptive
  LIFO. Fork PR #185.
- `upstream-pr/admission-priority-queue`: order backend admission by
  `nvext.agent_hints.priority` and evict the newest lowest-priority waiter
  for a higher-priority arrival at a full queue. On top of the previous
  branch. Fork PR #188.

Until the planned stack below exists, `main` is still the pre-layout fork
history (kept at `archive/pre-upstream-main-20260922/main`) and is not
base plus merges.

## Planned stack (agreed 2026-09-22, in order)

1. Done: the vLLM worker extension branch above. The SGLang side of the
   engine bump was already upstream.
2. Done: the two admission branches above. Upstream's delay-bounded
   worker queue replaced the fork's frontend count margin.
3. `upstream-pr/worker-set-cost-placement`: cost-based placement across a
   model's worker sets as the advisory-query stage of the routing pipeline.
4. `vendor/mirror-worker-sets`: mirror sets shadowing one serving worker,
   on top of placement.
5. `upstream-pr/cross-set-migration-fallback`: continue a request in
   another worker set when its own is exhausted; replay usage correction.
6. `upstream-pr/worker-drain-before-exit`: SGLang and vLLM drain callbacks,
   including multinode non-leader coordination.
7. `upstream-pr/process-required-taints`: process-wide required taints on
   the router's taint filter.
8. Small residuals, each verified against upstream before it is written:
   planner GPU budget and capabilities, SGLang metadata and FPM
   attribution, GPT-OSS tool fixes, priority forwarding, per-worker success
   attribution, registered endpoints on health.
9. `vendor/ci-compliance`: fork CI workflows and SBOM baselines.
10. `vendor/frontend-crates-patch`: cargo patch pointing the parser crates
    at doublewordai/frontend-crates.

---

@AGENTS.md
