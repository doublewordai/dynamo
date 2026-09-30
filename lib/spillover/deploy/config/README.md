# Deployment config generator

`lib/spillover/deploy/config/deployments.yaml` is the single source of truth for how every model
is deployed: the primary settings the spillover policy needs, the model card the proxies mirror,
and the proxy tiers that fail over from the primary workers. `spillover-deploy` turns it into the
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
  `--router-policy-config`), one `router/<model>/primary.args` per deployment (the SGLang
  worker `--router-*` flags), and one proxy config per `(deployment, tier, replica)`, under a
  directory named after the Dynamo model. Each proxy config is a complete
  `dw_proxy_core::config::ProxyConfig`. It records what it wrote in `.generated-files` and,
  on the next run, removes files it wrote before that the new input no longer describes, so a
  dropped tier does not leave a stale proxy config behind. Unrelated files in `--out` are
  never touched. It validates a staged copy of the whole tree before touching `--out`, and
  prunes stale files only after the new tree is written, so a failed run leaves the previous
  generated tree in place.
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
1000 because rank 0 is a primary worker. Replica `r` of that tier gets rank
`1000 * (N + 1) + r`, so a tier may have at most 1000 replicas. The policy's `dp_ranks` and the
proxy configs' `dp_rank`s are both derived from this rule and therefore always agree. Ranges are
per Dynamo deployment, so two deployments may reuse the same ranks.

## `deployments.yaml` schema

```yaml
deployments:
  "<Dynamo model name>":          # e.g. zai-org/GLM-5.3
    primary:
      primary_capacity_blocks: <float > 0, optional> # fallback KV capacity of one primary rank, in blocks
      occupancy_threshold: <float in (0, 4]>
      primary_max_requests: <int > 0, optional>      # fallback concurrency limit of one primary rank
      failover_penalty_blocks: <float >= 0> # cost added to a full primary worker
      pending_weight_blocks: <float >= 0>   # cost per active request on any worker
      admission_queue_margin: <int >= 1, default 256> # engine-waiting requests before a primary worker is excluded
    model:
      model_path: <absolute local model directory>  # same path as the SGLang workers; never a bare HF repo id
      served_model_names: [<primary name>, <alias>, ...]  # [0] must equal the Dynamo model name above
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
          thinking_dialect: reasoning_effort   # optional; see "Thinking controls" below
          thinking_strict: false               # optional
          cache_key: none                      # optional: none | prompt_cache_key | user
          cache_key_secret_env: <env var>      # required with cache_key
        penalty_blocks: <float >= 0>        # fixed "always full" cost for the tier
        weight_blocks: <float >= 0>         # tier preference; smaller is preferred
        replicas: <int >= 1 and <= 1000>
```

`validate` rejects a deployment whose `served_model_names[0]` is not its Dynamo model name
(the router keys the spillover policy by the primary served name, so a mismatch would silently
never spill), duplicate tier names, two deployment or tier names that sanitize to the same
output path, a deployment with no tiers, `admission_queue_margin: 0`, an `occupancy_threshold`
outside `(0, 4]`, a non-positive `primary_capacity_blocks` or `primary_max_requests` when set,
and invalid tier values (the same bounds the policy enforces). It also rejects names that
sanitize to `.` or `..`, which would write outside `--out`.

When `occupancy_threshold` is above `1.0`, `generate` prints a warning to stderr that the
frontend admission queue must be deep enough to hold the implied backlog; see
[admission margin](#admission-margin).

## Primary capacity

`primary_capacity_blocks` and `primary_max_requests` are optional fallbacks. The spillover
policy normally reads each primary worker's capacity from the `total_kv_blocks` and
`max_num_seqs` it advertises in its runtime config, so the policy always tracks the engine the
worker actually runs. Emit the fallbacks (set them in `primary:`) only when the primary workers
advertise no usable capacity; omit them and the generated `router-policy.yaml` leaves the keys
out entirely.

`occupancy_threshold` is the fraction of a primary worker's advertised capacity at which the
policy counts that worker as full. A value above `1.0` means a worker is only considered full
after it has already queued more work than its advertised capacity, which is only safe with a
large enough frontend admission queue.

A `dw_proxy_core::config::ProxyConfig` (one generated proxy YAML, or a hand-written config) can
advertise engine capacity of its own with `advertised_capacity`:

```yaml
advertised_capacity:
  kv_blocks: 4096     # -> ModelRuntimeConfig::total_kv_blocks
  max_requests: 32    # -> ModelRuntimeConfig::max_num_seqs
```

This is for a proxy that fronts a real primary engine (for example a simulated primary) so the
router sees the engine's true limits. A plain proxy that owns no KV cache leaves it unset and
advertises `None`, which the policy reads as "capacity not advertised" rather than mistaking a
placeholder for real capacity. Each field is validated to be greater than 0 when present.
`spillover-deploy` does not emit this for the provider proxy tiers it generates because those
proxy workers own no engine; set it by hand only for a primary-style proxy.

## Thinking controls

Every request reaches Dynamo as a chat completion, and the frontend normalizes the client's
thinking controls before a proxy sees them. The proxy reads one intent from that: thinking on,
off or adaptive, an effort grade, and a token budget. `thinking_dialect` names how the provider
expects it; each dialect is a fixed translation tested in `proxy-core/tests/thinking.rs`.

| dialect | on / off | effort | budget |
|---|---|---|---|
| `reasoning_effort` (default) | off as `reasoning_effort: none`; on only with a grade | `reasoning_effort` | not expressed |
| `reasoning_object` | `reasoning.enabled` | `reasoning.effort` | `reasoning.max_tokens` (instead of the effort) |
| `chat_template_kwargs` | `enable_thinking` and `thinking` | not expressed | not expressed |
| `none` | not expressed | not expressed | not expressed |

Adaptive sends nothing in every dialect: the model decides. When the dialect cannot express part
of a request's choice, the proxy sends what it can and counts
`proxy_thinking_total{event="unexpressed"}`; with `thinking_strict: true` it retries the
request on a primary worker instead. Leave strict off when the model has a deployment default
thinking mode, which marks every request as decided. A response that reasons after thinking
was turned off counts `proxy_thinking_total{event="ignored"}`, which is how a wrong dialect
shows up.

## What reaches the provider, and what comes back

The proxy forwards only an allow-list of chat fields (`proxy-core/src/chat_request.rs`). The
client's `user` and `prompt_cache_key` are never forwarded, because `user` identifies our
customer's end users. For providers that route prompt caching on a key, `cache_key` sends an
opaque key instead: a keyed BLAKE3 hash of the client's `prompt_cache_key` (or `user`), stable
for the same input and not reversible without the secret in `cache_key_secret_env`. Measure a
provider's cached prompt tokens with and without it before turning it on.

Nothing from the provider's response reaches the client directly: the proxy re-renders the
output as model tokens and the frontend builds the response. Usage reported to the client is our
own token count, as on a primary worker (the provider's counts go only to the proxy's metrics);
the served-by tag carries the tier name, not the provider's; and a provider `content_filter`
stop is retried on another worker. Tier names can reach a client that asks for
`nvext.engine_data`, so keep them neutral, and keep the Dynamo target in onwards strict or
sanitized so `nvext` is removed from responses.

## Admission margin

The fork's engine-queue admission gate is a backend feature, not a router-policy one. It reads
`DYN_ADMISSION_QUEUE_MARGIN` from each **worker process** (`lib/runtime/src/admission_gate.rs`);
the frontend never reads it and there is no per-model override (no
`DYN_ADMISSION_QUEUE_MARGIN_OVERRIDES`). `generate` therefore writes two environment files per
deployment:

- `admission/<model>/primary.env` — `export DYN_ADMISSION_QUEUE_MARGIN=<admission_queue_margin>`
  (`>= 1`), to be sourced by every primary worker. The `export` means a plain `source` reaches
  the worker process even without `set -a`. The value bounds how many requests may sit in the
  engine's own waiting queue before the worker is excluded from selection; keeping it above the
  policy's failover point lets the policy decide to spill first. `0` is rejected because the
  runtime reads a present `0` as an always-firing gate, not as "off".
- `admission/<model>/proxy.env` — `unset DYN_ADMISSION_QUEUE_MARGIN`. Proxies never report
  `num_waiting_reqs`, so the margin is unenforceable on them, and clearing it stops a value
  leaking in from a shared launch environment.

The default comes from the `admission_queue_margin` sweep in `docs/spillover/tuning.md`:
steering away from primary stops once the margin is above single digits for a normal worker, so
`256` is safely above the failover point.

## Active-block tracking

`dw-spillover` measures primary occupancy as router-tracked decode blocks over
each primary worker's advertised capacity (falling back to `primary_capacity_blocks` when the
worker advertises none), and the router only counts those blocks when
`router_track_active_blocks` is on. The fork's frontend default for that flag is on, but a
frontend started with `--no-router-track-active-blocks` reports zero occupancy, so `generate`
turns it on **per worker set**, not on the frontend:

- `router/<model>/primary.args` — the SGLang worker `--router-*` flags
  (`--router-mode kv --router-track-active-blocks ...`) to append to every primary worker's
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

