# Deployment config generator

`lib/spillover/deploy/config/deployments.yaml` is the single source of truth for how every model
is deployed: the primary settings the spillover policy needs, the model card the proxies mirror,
and the proxy tiers that fail over from the primary workers. `spillover-deploy` turns it into the
two artifacts the runtime consumes, and validates them with the same Rust types that read them.

```
export PATH="$HOME/.cargo/bin:$PATH"
CARGO_TARGET_DIR=... cargo run -p dw-spillover-deploy -- \
    generate --input lib/spillover/deploy/config/deployments.yaml \
    --out lib/spillover/deploy/config/generated
cargo run -p dw-spillover-deploy -- check \
    --input lib/spillover/deploy/config/deployments.yaml
```

- `generate` writes `router-policy.yaml` (pass to Dynamo's frontend with
  `--router-policy-config`), one `router/<model>/primary.args` per deployment (the primary
  worker `--router-*` flags, for every engine that parses them), and one proxy config per
  `(deployment, tier, replica)`, under a directory named after the Dynamo model. Each proxy
  config is a complete `dw_proxy_core::config::ProxyConfig`. It records what it wrote in
  `.generated-files` and, on the next run, removes files it wrote before that the new input no
  longer describes, so a dropped tier does not leave a stale proxy config behind. Unrelated
  files in `--out` are never touched. It validates a staged copy of the whole tree before
  touching `--out`, and prunes stale files only after the new tree is written, so a failed run
  leaves the previous generated tree in place.
- `check` does exactly the same parsing, generation and validation without writing to the output
  directory. Use it in CI.
- Both parse the generated `parameters` with
  `dw_spillover_policy::SpilloverParameters` and call `validate()`, and load every proxy config
  with `dw_proxy_core::config::ProxyConfig::load`.
- `lib/spillover/deploy/config/generated/` is committed. `cargo test -p dw-spillover-deploy`
  fails if it is stale.

## Primary engines

The spillover proxies register as ordinary Tokens chat workers, so spillover works with primary
workers on any Dynamo engine. `primary.engine` is required and is one of `sglang`, `vllm`,
`trtllm`, `mocker`, `tokenspeed`. It drives the few card fields a proxy can only mirror when the
primary registers them and the one deployment field, `admission_queue_margin`, that is only
enforceable when the engine reports its waiting queue.

In the table, **mirror** means the proxy must advertise the same value the primary engine does;
the value is part of the model card, and a mismatch splits the worker set. **optional** for
`context_length` means omitting it makes the card fall back to the model's architectural maximum.

| Engine | `kv_block_size` | `context_length` | `model_path` | Served names | `enable_eagle` | Chat template | Worker router flags | Admission margin | Limits |
|---|---|---|---|---|---|---|---|---|---|
| `sglang` | mirror | `--context-length`, or omit when unset | mirror | many aliases | supported (EAGLE/MTP) | mirror | `--router-*` | always enforceable | — |
| `vllm` | mirror | its resolved `max_model_len` (always published) | mirror | many aliases | must be false | mirror | `--router-*` | always enforceable | omitting `context_length` warns |
| `trtllm` | mirror | `--max-seq-len`, or omit when unset | mirror | one name | must be false | mirror | `--router-*` | only with `--publish-metrics` | no aliases |
| `mocker` | mirror | `--max-model-len` (set it: without it the mocker advertises 0, which a proxy cannot mirror) | mirror; a local path is not recorded (handled by the generator) | one name | must be false | mirror | `--router-*` | never | no aliases |
| `tokenspeed` | mirror | its scheduler `max_model_len` | mirror | one name | must be false | mirror | none | never | no aliases, no card `router_config` |

Field notes:

- `kv_block_size` and `context_length` must equal the primary engine's actual KV block size and
  resolved context length; they are part of the model card, and a mismatch splits the worker
  set. `context_length` is optional. Omitted, the card advertises no length and the router falls
  back to the model's architectural maximum, which matches SGLang started without
  `--context-length` and TRT-LLM started without `--max-seq-len`. Otherwise set it to the value
  the engine advertises (see the table). **vLLM always publishes its resolved `max_model_len`**,
  so a vLLM deployment that omits `context_length` advertises a different value and the
  generator warns to stderr.
- `model_path` must be the exact model string the primary was started with (a local directory
  or a Hugging Face id; the proxy fetches only config and tokenizer files). Most engines record it
  in the card as `source_path`, which feeds the checksum. The mocker's `make_engine` entrypoint
  records it only for a Hugging Face id, so for a mocker primary with a local path the generator
  sets `omit_source_path: true` on every proxy.
- `custom_jinja_template` is passed through to every proxy config. It must name the same chat
  template the primary engine uses, on a path the proxy image can read (see
  [Model files](../../../../docs/spillover/images.md#model-files)).
- `enable_eagle` mirrors whether the primary runs EAGLE/MTP. It is not part of the checksum, so a
  mismatch does not split the worker set; it changes how the router hashes request blocks for
  the set, so a mismatch silently breaks cache affinity between primary and proxies. **It must
  equal the primary's setting.** Only SGLang sets it (EAGLE, EAGLE3, MTP), so on another engine
  it must stay false and the generator warns if it is set.
- `served_model_names[0]` is always the primary's Dynamo model name. Additional aliases are
  accepted only for engines that register a model alias in their card: SGLang and vLLM. TRT-LLM,
  the mocker and TokenSpeed accept exactly one name, and `validate` rejects more.
- Worker router flags (the generated `router/<model>/primary.args`) are shared by SGLang, vLLM,
  TRT-LLM and the mocker through `parse_worker_router_config`. TokenSpeed has no such flags, so
  it gets no args file and its proxy configs carry no `router_config`; see
  [Active-block tracking](#active-block-tracking).
- Admission margin support decides whether `admission_queue_margin` is enforceable: only SGLang
  and vLLM always report their waiting queue, TRT-LLM reports it only with `--publish-metrics`,
  and the mocker and TokenSpeed never do. See [Admission margin](#admission-margin).

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
      engine: sglang | vllm | trtllm | mocker | tokenspeed  # required; see "Primary engines"
      primary_capacity_blocks: <float > 0, optional> # fallback KV capacity of one primary rank, in blocks
      occupancy_threshold: <float in (0, 4]>
      primary_max_requests: <int > 0, optional>      # fallback concurrency limit of one primary rank
      failover_penalty_blocks: <float >= 0> # cost added to a full primary worker; more than ceil(context_length / kv_block_size) + the costliest tier penalty + weight makes the threshold a hard cap under the default overlap weights with prefill_load_scale 1 (scale the context term if prefill_load_scale is raised, and add pending_weight_blocks times the concurrency a spill tier reaches; the generator warns below the base floor, and skips the warning when context_length is omitted)
      pending_weight_blocks: <float >= 0>   # cost per active request on any worker
      admission_queue_margin: <int >= 1, optional> # engine-waiting requests before a primary worker is excluded; default 256 for engines that report waiting, rejected otherwise
    model:
      model_path: <absolute local model directory>  # same path as the primary workers of any engine; never a bare HF repo id
      served_model_names: [<primary name>, <alias>, ...]  # [0] must equal the Dynamo model name above; aliases only for sglang/vllm
      namespace: <Dynamo namespace>
      component: <Dynamo component>
      endpoint: <Dynamo endpoint>
      kv_block_size: <int > 0>              # must equal the primary workers'
      context_length: <int > 0, optional>   # must equal the primary workers'; omit to fall back to the model max
      custom_jinja_template: <absolute path, optional> # must equal the primary workers' chat template
      enable_eagle: <bool, default false>   # must equal the primary workers'; only meaningful for sglang
      parser_family: glm47 | deepseek_v41 | kimi_k3 | hermes
      endpoint_types: chat,completions    # optional; non-empty subset of {chat, completions}, default chat,completions
    vcache_ttl_secs: <int, default 300>
    vcache_max_blocks: <int, default 1000000>
    tiers:
      - name: <tier name, unique per deployment>
        provider:
          name: <name shown in logs and metrics; never in the served-by tag>
          base_url: <OpenAI-compatible base URL ending in /v1>
          api_key_env: <environment variable holding the API key>
          model: <provider-side model slug>
          provider_preferences: { ... }     # optional, merged into the request body as `provider`
          thinking_dialect: reasoning_effort   # optional; see "Thinking controls" below
          thinking_strict: false               # optional
          cache_key: none                      # optional: none | prompt_cache_key | user
          cache_key_secret_env: <env var>      # required with cache_key
          circuit_breaker:                     # optional; enabled with defaults when omitted
            failure_threshold: 5               # consecutive provider failures that open the breaker
            cooldown_ms: 30000                 # how long it stays open before one probe
            max_cooldown_ms: 300000            # cap on the doubled cooldown after a failed probe
        penalty_blocks: <float >= 0>        # fixed "always full" cost for the tier
        weight_blocks: <float >= 0>         # tier preference; smaller is preferred
        replicas: <int >= 1 and <= 1000>
```

`validate` rejects a deployment whose `served_model_names[0]` is not its Dynamo model name
(the router keys the spillover policy by the primary served name, so a mismatch would silently
never spill), duplicate tier names, two deployment or tier names that sanitize to the same
output path, a deployment with no tiers, `admission_queue_margin: 0`, an explicit
`admission_queue_margin` on an engine that never reports a waiting queue, more than one
`served_model_names` entry on an engine that registers no aliases, `context_length: 0`, an
`occupancy_threshold` outside `(0, 4]`, a non-positive `primary_capacity_blocks` or
`primary_max_requests` when set, and invalid tier values (the same bounds the policy
enforces). It also rejects names that sanitize to `.` or `..`, which would write outside
`--out`.

When `occupancy_threshold` is above `1.0`, `generate` prints a warning to stderr that each
primary worker's engine-queue admission margin and admission gate must be able to hold the
implied backlog; see [admission margin](#admission-margin).

## Primary capacity

`primary_capacity_blocks` and `primary_max_requests` are optional fallbacks. The spillover
policy normally reads each primary worker's capacity from the `total_kv_blocks` and
`max_num_seqs` it advertises in its runtime config, so the policy always tracks the engine the
worker actually runs. Emit the fallbacks (set them in `primary:`) only when the primary workers
advertise no usable capacity; omit them and the generated `router-policy.yaml` leaves the keys
out entirely.

`occupancy_threshold` is the fraction of a primary worker's advertised capacity at which the
policy counts that worker as full. A value above `1.0` means a worker is only considered full
after it has already queued more work than its advertised capacity, which is only safe when
the primary workers' admission margin and gate can hold that queue.

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

## Endpoint types and the worker set

`model.endpoint_types` is the comma-separated endpoint advertisement every generated proxy
card carries. It defaults to `chat,completions`, the `WorkerConfig` default production
primary engines use, because `endpoint_types` feeds the card's `model_type`, and
`model_type` is part of `worker_set_key` (`lib/llm/src/discovery/watcher.rs`). A proxy that
advertised only `chat` would land in a different WorkerSet from a `chat,completions` primary
and the two would never route to each other, so spillover would never engage. Set it only when
the primary workers advertise something other than the default; it is validated as a non-empty
subset of `{chat, completions}`.

Because the default includes `completions`, the frontend may route a `/v1/completions` request
to a proxy. That request carries no chat request, so the proxy refuses it with the migratable
`no_chat_request` refusal and the router retries it on another worker. Each proxy tier
therefore costs one fast refused hop for a completions request; the frontend's migration limit
bounds how many hops a request can take.

## Provider circuit breaker

A proxy stays in the router's rotation while its provider is down, so every request routed to it
pays a provider attempt before the frontend migrates the request to another worker. The proxy's
per-provider circuit breaker caps that cost. After `failure_threshold` consecutive provider-side
failures (429, 5xx/stream errors, transport and dropped streams, and a rejected key) it opens and
refuses new requests immediately, before any provider call, with the same migratable
`WorkerOverloaded` error the proxy already returns, so the router retries on another worker. It
stays open for `cooldown_ms`, then lets exactly one request through (half-open); a concurrent
request is refused. A successful probe closes the breaker and resets the cooldown; a failed probe
reopens it with the cooldown doubled, up to `max_cooldown_ms`. A provider *answer*, even a 4xx
rejection or a content-filter stop, proves reachability and resets the streak without counting as
a failure. Proxy-side outcomes (cancellation, migration replay, a missing chat request, an
unsupported request) do not feed the breaker.

```yaml
provider:
  circuit_breaker:
    failure_threshold: 5     # > 0; default 5
    cooldown_ms: 30000       # > 0; default 30000 (30 s)
    max_cooldown_ms: 300000  # >= cooldown_ms; default 300000 (5 min)
```

The block is optional and, when present, each field falls back to its default. Omit it entirely
and the proxy still installs a breaker with those defaults; set `failure_threshold` high (or the
cost of a refused request is small) only if you would rather every request try the provider first.
The state is exposed as `dynamo_component_proxy_circuit_open` (1 while open or half-open) and
refusals are counted in `dynamo_component_proxy_requests_total{outcome="circuit_open"}`; the open
transition is logged once at `warn` and the close at `info`.

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

- `admission/<model>/primary.env` — exports `DYN_ADMISSION_QUEUE_MARGIN` for every primary
  worker of an engine that can enforce it, with `export` so a plain `source` reaches the worker
  process even without `set -a`. The value bounds how many requests may sit in the engine's own
  waiting queue before the worker is excluded from selection; keeping it above the policy's
  failover point lets the policy decide to spill first. `0` is rejected because the runtime
  reads a present `0` as an always-firing gate, not as "off". Which engines get it:
  - **SGLang and vLLM** always report their waiting queue, so the margin is always emitted.
  - **TRT-LLM** reports it only with `--publish-metrics`. The margin is emitted with a comment
    saying so, and `generate` prints a stderr warning, because a margin without the flag is
    unenforced and replaces the default concurrency limit.
  - **mocker and TokenSpeed** never report waiting. Their file `unset`s the variable, because a
    set margin would only remove the default concurrency limit. Setting `admission_queue_margin`
    explicitly for them is rejected.
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

- `router/<model>/primary.args` — the primary worker `--router-*` flags
  (`--router-mode kv --router-track-active-blocks ...`) to append to every primary worker's
  command line, so its model card carries the worker set's `router_config`. The flags are shared
  by SGLang, vLLM, TRT-LLM and the mocker; TokenSpeed has none and gets no file.
- every proxy config of an engine that advertises a card `router_config` gets the same
  `router_config`, so the proxy card matches and the two stay one worker set. The card checksum
  includes `router_config`, so a mismatch splits the set. `dw-proxy-worker` sets
  `shared_cache_multiplier` explicitly to the shared CLI default (0.5) because
  `KvRouterConfig::default()` is 0.0 and that field is serialized into the card.

`frontend.env` is a note recording that no frontend-wide flag is emitted for engines that
advertise a card `router_config`. A frontend-wide `DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true`
(equivalently `--router-track-active-blocks`) also works but changes tracking for every other
model on the frontend, which is why this deployment does not use it. **TokenSpeed is the
exception**: `lib/bindings/python/rust/backend.rs` hard-codes its card `router_config` to
`None`, so the generator writes no `router_config` for it (and no `primary.args`), and the note
says to start the frontend with `--router-track-active-blocks` globally for its worker set. The
per-request concurrency signal still works without it; only occupancy-based KV routing needs
the global flag.

If tracking is off, the policy logs an error at construction naming the model and falls back
to Dynamo's default policy for it; failover then never fires, loudly rather than silently.

