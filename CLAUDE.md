# doublewordai/dynamo fork layout

Integration fork of ai-dynamo/dynamo. Production frontend, router, planner
and worker images are built from a commit of this repo's `main` by
doublewordai/dynamo-images, and the image tag carries that commit.

## Branches

- `upstream-base`: exactly the upstream `main` commit the fork is based on.
  Currently `2e4998a30ccd` (2026-09-22). Never carries our commits. Upstream
  cuts releases on branches off `main`, so the base is a pinned `main`
  commit, not a release tag.
- `main`: `upstream-base` plus every patch branch below, merged with
  `--no-ff` in stack order. `git log --merges upstream-base..main` is the
  patch list.
- `upstream-pr/<topic>`: a change we intend to land upstream. Based on
  `upstream-base`. This repo is a public fork, so the branch heads the
  upstream PR directly once rebased onto upstream `main`.
- `vendor/<topic>`: a Doubleword-only change that will not go upstream.
  Based on `upstream-base`.
- `archive/*` and other branches are history and are not part of any image.

## Rules

- No backports. Do not cherry-pick upstream commits onto `main`; a fix that
  is on upstream `main` arrives by moving `upstream-base`.
- One branch per patch, atomic, with the reason in the commit message.
- Moving the base: point `upstream-base` at the new upstream `main` commit,
  rebase each patch branch that is still needed onto it, drop the ones
  upstream now contains, rebuild `main` as base plus merges, force-push
  `main`, build images. Update the stack list below. The base moves on a
  cadence, not only when a fix is needed.

## Current stack

- `vendor/fork-layout`: this section.

Until the planned stack below exists, `main` is still the pre-layout fork
history (kept at `archive/pre-upstream-main-20260922/main`) and is not
base plus merges.

## Planned stack (agreed 2026-09-22, in order)

1. `vendor/engines-vllm-0.30-sglang-0.5.20`: integration fixes for the
   engine versions the fleet runs ahead of upstream's pins.
2. `upstream-pr/worker-admission-queue-margin`: engine-queue margin with
   priority eviction at upstream's worker-side admission gate, fed the
   engine's live waiting count.
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
