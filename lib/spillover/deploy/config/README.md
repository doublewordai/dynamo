# Deployment config generator

`deploy/deployments.yaml` is the single source of truth for how every model is deployed: the
hosted settings the spillover policy needs, the model card the proxies mirror, and the proxy
tiers that fail over from the hosted workers. `spillover-deploy` turns it into the two artifacts
the runtime consumes, and validates them with the same Rust types that read them.

```
export PATH=/home/peter/.cargo/bin:$PATH
CARGO_TARGET_DIR=... cargo run -p dw-spillover-deploy -- \
    generate --input deploy/deployments.yaml --out deploy/generated
cargo run -p dw-spillover-deploy -- check --input deploy/deployments.yaml
```

- `generate` writes `router-policy.yaml` (pass to Dynamo's frontend with
  `--router-policy-config`) and one proxy config per `(deployment, tier, replica)`, under a
  directory named after the Dynamo model. Each proxy config is a complete
  `dw_proxy_core::config::ProxyConfig`.
- `check` does exactly the same parsing, generation and validation without writing to the output
  directory. Use it in CI.
- Both parse the generated `parameters` with
  `dw_spillover_policy::SpilloverParameters` and call `validate()`, and load every proxy config
  with `dw_proxy_core::config::ProxyConfig::load`.
- `deploy/generated/` is committed. `cargo test -p dw-spillover-deploy` fails if it is stale.

## Tier DP ranks

Tier `N` (0-based, in the order written in `deployments.yaml`) of a deployment is reserved the
inclusive DP rank range `[1000 * (N + 1), 1000 * (N + 1) + 999]`; the first tier starts at rank
1000 because rank 0 is a hosted worker. Replica `r` of that tier gets rank
`1000 * (N + 1) + r`, so a tier may have at most 1000 replicas. The policy's `dp_ranks` and the
proxy configs' `dp_rank`s are both derived from this rule and therefore always agree. Ranges are
per Dynamo deployment, so two deployments may reuse the same ranks.

## `deployments.yaml` schema

```yaml
deployments:
  "<Dynamo model name>":          # e.g. zai-org/GLM-5.3@interactive
    hosted:
      hosted_capacity_blocks: <float > 0>   # KV capacity of one hosted rank, in blocks
      occupancy_threshold: <float in (0, 1]>
      failover_penalty_blocks: <float >= 0> # cost added to a full hosted worker
      pending_weight_blocks: <float >= 0>   # cost per active request on any worker
    model:
      model_path: <HF repo id>              # same path as the SGLang workers
      served_model_names: [<name>, ...]     # must include the Dynamo model name above
      namespace: <Dynamo namespace>
      component: <Dynamo component>
      endpoint: <Dynamo endpoint>
      kv_block_size: <int > 0>              # must equal the SGLang workers'
      context_length: <int > 0>             # must equal the SGLang workers'
      parser_family: glm47 | deepseek_v41 | hermes
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
name, duplicate tier names, a deployment with no tiers, and invalid hosted/tier values (the
same bounds the policy enforces).
