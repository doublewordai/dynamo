# Deployment config generator

`lib/spillover/deploy/config/deployments.yaml` is the single source of truth for how every model
is deployed: the hosted settings the spillover policy needs, the model card the proxies mirror,
and the proxy tiers that fail over from the hosted workers. `spillover-deploy` turns it into the
two artifacts the runtime consumes, and validates them with the same Rust types that read them.

```
export PATH=/home/peter/.cargo/bin:$PATH
CARGO_TARGET_DIR=... cargo run -p dw-spillover-deploy -- \
    generate --input lib/spillover/deploy/config/deployments.yaml \
    --out lib/spillover/deploy/config/generated
cargo run -p dw-spillover-deploy -- check \
    --input lib/spillover/deploy/config/deployments.yaml
```

- `generate` writes `router-policy.yaml` (pass to Dynamo's frontend with
  `--router-policy-config`), one `router/<model>/hosted.args` per deployment (the SGLang
  worker `--router-*` flags), and one proxy config per `(deployment, tier, replica)`, under a
  directory named after the Dynamo model. Each proxy config is a complete
  `dw_proxy_core::config::ProxyConfig`. It records what it wrote in `.generated-files` and,
  on the next run, removes files it wrote before that the new input no longer describes, so a
  dropped tier does not leave a stale proxy config behind. Unrelated files in `--out` are
  never touched.
- `check` does exactly the same parsing, generation and validation without writing to the output
  directory. Use it in CI.
- Both parse the generated `parameters` with
  `dw_spillover_policy::SpilloverParameters` and call `validate()`, and load every proxy config
  with `dw_proxy_core::config::ProxyConfig::load`.
- `lib/spillover/deploy/config/generated/` is committed. `cargo test -p dw-spillover-deploy`
  fails if it is stale.

## Tier DP ranks

Tier `N` (0-based, in the order written in `lib/spillover/deploy/config/deployments.yaml`) of a
deployment is reserved the
inclusive DP rank range `[1000 * (N + 1), 1000 * (N + 1) + 999]`; the first tier starts at rank
1000 because rank 0 is a hosted worker. Replica `r` of that tier gets rank
`1000 * (N + 1) + r`, so a tier may have at most 1000 replicas. The policy's `dp_ranks` and the
proxy configs' `dp_rank`s are both derived from this rule and therefore always agree. Ranges are
per Dynamo deployment, so two deployments may reuse the same ranks.

## `deployments.yaml` schema

```yaml
deployments:
  "<Dynamo model name>":          # e.g. zai-org/GLM-5.3
    hosted:
      hosted_capacity_blocks: <float > 0>   # KV capacity of one hosted rank, in blocks
      occupancy_threshold: <float in (0, 1]>
      failover_penalty_blocks: <float >= 0> # cost added to a full hosted worker
      pending_weight_blocks: <float >= 0>   # cost per active request on any worker
      admission_queue_margin: <int >= 1, default 256> # engine-waiting requests before a hosted worker is excluded
    model:
      model_path: <HF repo id>              # same path as the SGLang workers
      served_model_names: [<name>, ...]     # must include the Dynamo model name above
      namespace: <Dynamo namespace>
      component: <Dynamo component>
      endpoint: <Dynamo endpoint>
      kv_block_size: <int > 0>              # must equal the SGLang workers'
      context_length: <int > 0>             # must equal the SGLang workers'
      parser_family: glm47 | deepseek_v41 | kimi_k3 | hermes
    vcache_ttl_secs: <int, default 300>
    vcache_max_blocks: <int, default 1000000>
    tiers:
      - name: <tier name, unique per deployment>
        provider:
          name: <name shown in logs and the served-by tag>
          base_url: <OpenAI-compatible base URL ending in /v1>
          api_key_env: <environment variable holding the API key>
          model: <provider-side model slug>
          provider_preferences: { ... }     # optional, merged into the request body as `provider`
        penalty_blocks: <float >= 0>        # fixed "always full" cost for the tier
        weight_blocks: <float >= 0>         # tier preference; smaller is preferred
        replicas: <int >= 1 and <= 1000>
```

`validate` rejects a deployment whose `served_model_names` does not contain its Dynamo model
name, duplicate tier names, two deployment or tier names that sanitize to the same output path,
a deployment with no tiers, `admission_queue_margin: 0`, and invalid hosted/tier values (the
same bounds the policy enforces). It also rejects names that sanitize to `.` or `..`, which
would write outside `--out`.

## Admission margin

The fork's engine-queue admission gate is a backend feature, not a router-policy one. It reads
`DYN_ADMISSION_QUEUE_MARGIN` from each **worker process** (`lib/runtime/src/admission_gate.rs`);
the frontend never reads it and there is no per-model override (no
`DYN_ADMISSION_QUEUE_MARGIN_OVERRIDES`). `generate` therefore writes two environment files per
deployment:

- `admission/<model>/hosted.env` — `export DYN_ADMISSION_QUEUE_MARGIN=<admission_queue_margin>`
  (`>= 1`), to be sourced by every hosted worker. The `export` means a plain `source` reaches
  the worker process even without `set -a`. The value bounds how many requests may sit in the
  engine's own waiting queue before the worker is excluded from selection; keeping it above the
  policy's failover point lets the policy decide to spill first. `0` is rejected because the
  runtime reads a present `0` as an always-firing gate, not as "off".
- `admission/<model>/proxy.env` — `unset DYN_ADMISSION_QUEUE_MARGIN`. Proxies never report
  `num_waiting_reqs`, so the margin is unenforceable on them, and clearing it stops a value
  leaking in from a shared launch environment.

The default comes from the `admission_queue_margin` sweep in `docs/spillover/tuning.md`:
steering away from hosted stops once the margin is above single digits for a normal worker, so
`256` is safely above the failover point.

## Active-block tracking

`dw-spillover` measures hosted occupancy as router-tracked decode blocks over
`hosted_capacity_blocks`, and the router only counts those blocks when
`router_track_active_blocks` is on. The fork's frontend default for that flag is on, but a
frontend started with `--no-router-track-active-blocks` reports zero occupancy, so `generate`
turns it on **per worker set**, not on the frontend:

- `router/<model>/hosted.args` — the SGLang worker `--router-*` flags
  (`--router-mode kv --router-track-active-blocks ...`) to append to every hosted worker's
  command line, so its model card carries the worker set's `router_config`.
- every proxy config gets the same `router_config`, so the proxy card matches and the two stay
  one worker set. The card checksum includes `router_config`, so a mismatch splits the set.
  `dw-proxy-worker` sets `shared_cache_multiplier` explicitly to the SGLang CLI default (0.5)
  because `KvRouterConfig::default()` is 0.0 and that field is serialized into the card.

`frontend.env` is a note recording that no frontend-wide flag is emitted. A frontend-wide
`DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true` (equivalently `--router-track-active-blocks`) also works
but changes tracking for every other model on the frontend, which is why this deployment does
not use it.

If tracking is off, the policy logs an error at construction naming the model and falls back
to Dynamo's default policy for it; failover then never fires, loudly rather than silently.

