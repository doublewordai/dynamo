// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generator tests: end-to-end validation, rank assignment, input rejection, staleness.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use dw_proxy_core::config::{ProxyConfig, ProxyRouterMode};
use dw_spillover_deploy::{
    DeploymentsFile, build, check, generate, hard_cap_failover_penalty, replica_rank,
    tier_rank_base, validate_dir, write_files,
};

/// The crate's `config/` directory holds the example input and the committed output.
fn config_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config")
}

fn example_input() -> PathBuf {
    config_dir().join("deployments.yaml")
}

fn generated_dir() -> PathBuf {
    config_dir().join("generated")
}

/// Every regular file under `dir`, keyed by `/`-joined path relative to `dir`.
fn read_dir_files(dir: &Path) -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(&current).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let relative = path
                    .strip_prefix(dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                files.insert(relative, fs::read_to_string(&path).unwrap());
            }
        }
    }
    files
}

#[test]
fn example_generates_and_validates() {
    let files = build(&example_input()).unwrap();
    assert!(files.contains_key("router-policy.yaml"));
    // 1 policy + 1 frontend note + 1 manifest + for the one deployment (1 primary router args
    // file + 2 admission env files + 3 proxy configs).
    assert_eq!(files.len(), 1 + 1 + 1 + (1 + 2 + 3));

    let frontend_env = files.get("frontend.env").expect("frontend note");
    assert!(
        frontend_env
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .all(|line| !line.contains("DYN_ROUTER_TRACK_ACTIVE_BLOCKS")),
        "no frontend-wide tracking export: {frontend_env}"
    );
    assert!(
        frontend_env.contains("router_track_active_blocks")
            && frontend_env.contains("--router-track-active-blocks"),
        "{frontend_env}"
    );
    assert!(frontend_env.contains("zai-org/GLM-5.3"), "{frontend_env}");

    let args = files
        .get("router/zai-org_GLM-5.3/primary.args")
        .expect("router/zai-org_GLM-5.3/primary.args");
    let flags: Vec<&str> = args
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .collect();
    assert_eq!(
        flags,
        vec![
            "--router-mode kv --router-track-active-blocks --no-router-track-output-blocks \
             --shared-cache-multiplier 0.5"
        ],
        "{args}"
    );

    let primary_env = files
        .get("admission/zai-org_GLM-5.3/primary.env")
        .expect("primary admission env");
    assert!(
        primary_env.contains("export DYN_ADMISSION_QUEUE_MARGIN=256"),
        "{primary_env}"
    );
    let proxy_env = files
        .get("admission/zai-org_GLM-5.3/proxy.env")
        .expect("proxy admission env");
    assert!(
        proxy_env.contains("unset DYN_ADMISSION_QUEUE_MARGIN"),
        "{proxy_env}"
    );

    let temp = tempfile::tempdir().unwrap();
    write_files(temp.path(), &files).unwrap();
    validate_dir(temp.path()).unwrap();

    // `check` parses, generates and validates without touching the output directory.
    check(&example_input()).unwrap();
}

/// The shipped example must point `model_path` at a local mount, not a bare HF repo id: the
/// proxy is token-only, and `build_local_model` downloads full weights for a source that does
/// not exist on disk.
#[test]
fn shipped_example_fails_over_as_a_hard_cap() {
    let doc: DeploymentsFile =
        serde_yaml::from_str(&fs::read_to_string(example_input()).unwrap()).unwrap();
    for deployment in doc.deployments.values() {
        // ceil(131072 / 64) context blocks + the costliest tier's 200 + 40.
        assert_eq!(
            hard_cap_failover_penalty(deployment),
            Some(2048.0 + 240.0 + 1.0)
        );
        assert!(
            deployment.primary.failover_penalty_blocks
                >= hard_cap_failover_penalty(deployment).unwrap()
        );
        assert_eq!(deployment.primary.primary_capacity_blocks, None);
    }
}

#[test]
fn shipped_example_uses_a_local_model_path() {
    let raw = fs::read_to_string(example_input()).unwrap();
    let doc: DeploymentsFile = serde_yaml::from_str(&raw).unwrap();
    for (name, deployment) in &doc.deployments {
        let path = &deployment.model.model_path;
        assert!(
            Path::new(path).is_absolute(),
            "deployment {name:?}: model_path {path:?} must be an absolute local mount path, \
             not a bare HF repo id, or each proxy downloads the full model weights"
        );
    }
}

#[test]
fn model_endpoint_types_defaults_and_passes_through() {
    // The default must mirror the production primary `WorkerConfig` default so the
    // two cards share a `model_type` (and therefore a worker set).
    let temp = tempfile::tempdir().unwrap();
    let files = build(&write_input(
        temp.path(),
        &deployment_yaml("org/m", &["secondary"]),
    ))
    .unwrap();
    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("org_m/secondary-0.yaml").unwrap()).unwrap();
    assert_eq!(proxy.endpoint_types, "chat,completions");

    // An explicit advertisement reaches every generated proxy config.
    let temp = tempfile::tempdir().unwrap();
    let yaml = deployment_yaml("org/m", &["secondary"]).replace(
        "parser_family: glm47",
        "parser_family: glm47\n      endpoint_types: chat",
    );
    let files = build(&write_input(temp.path(), &yaml)).unwrap();
    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("org_m/secondary-0.yaml").unwrap()).unwrap();
    assert_eq!(proxy.endpoint_types, "chat");

    // An endpoint the proxy cannot serve is rejected at generation, naming the deployment.
    let temp = tempfile::tempdir().unwrap();
    let yaml = deployment_yaml("org/m", &["secondary"]).replace(
        "parser_family: glm47",
        "parser_family: glm47\n      endpoint_types: embedding",
    );
    let error = format!("{:#}", build(&write_input(temp.path(), &yaml)).unwrap_err());
    assert!(error.contains("endpoint_types"), "{error}");
    assert!(error.contains("org/m"), "{error}");
}

/// The router keys the spillover policy by `served_model_names[0]`, so the generator must
/// reject a deployment that lists the Dynamo model name as an alias instead.
#[test]
fn rejects_primary_served_name_not_equal_to_model_name() {
    let yaml = format!(
        "deployments:\n{}",
        deployment_block("org/m", &["secondary"]).replace(
            "served_model_names: [\"org/m\"]",
            "served_model_names: [\"alias/model\", \"org/m\"]"
        )
    );
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(temp.path(), &yaml);
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("served_model_names[0]"), "{error}");
    assert!(error.contains("alias/model"), "{error}");
}

/// Parse the emitted primary worker router flags into the fields the primary model card
/// carries.
///
/// This is deliberately independent of [`dw_spillover_deploy::ROUTER_ADVERTISEMENT`]: the
/// point of the test below is that the emitted flags and the proxy config agree, not that
/// both restate the same constant. It still does not run the engine Python CLI, so it cannot
/// prove the Python defaults match the Rust defaults; the cross-language check lives in
/// `proxy-worker`'s registration test.
fn parse_primary_args(args: &str) -> (String, bool, bool, Option<String>) {
    let mut mode = None;
    let mut track_active = false;
    let mut track_output = false;
    let mut shared_cache_multiplier = None;
    let mut words = args.split_whitespace();
    while let Some(flag) = words.next() {
        match flag {
            "--router-mode" => mode = words.next().map(str::to_string),
            "--router-track-active-blocks" => track_active = true,
            "--no-router-track-active-blocks" => track_active = false,
            "--router-track-output-blocks" => track_output = true,
            "--no-router-track-output-blocks" => track_output = false,
            "--shared-cache-multiplier" => {
                shared_cache_multiplier = words.next().map(str::to_string)
            }
            _ => {}
        }
    }
    (
        mode.expect("--router-mode"),
        track_active,
        track_output,
        shared_cache_multiplier,
    )
}

/// The generated proxy `router_config` and the emitted primary worker flags must describe
/// one advertisement, or the checksums differ and the worker set splits. This parses
/// the emitted flags rather than comparing them to the constant that produced them.
#[test]
fn primary_args_and_proxy_router_config_agree() {
    let files = build(&example_input()).unwrap();
    let args = files.get("router/zai-org_GLM-5.3/primary.args").unwrap();
    let flags: Vec<&str> = args
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .collect();
    let (mode, track_active, track_output, shared) = parse_primary_args(flags[0]);
    // Spillover requires KV routing; the card checksum pins the shared-cache multiplier to
    // the shared CLI default.
    assert_eq!(mode, "kv");
    assert_eq!(shared.as_deref(), Some("0.5"));

    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/secondary-0.yaml").unwrap()).unwrap();
    let router = proxy.router_config.expect("proxy config router_config");
    assert_eq!(router.mode, ProxyRouterMode::Kv);
    assert_eq!(router.track_active_blocks, track_active);
    assert_eq!(router.track_output_blocks, track_output);
}

#[test]
fn rank_assignment_matches_policy_and_proxies() {
    assert_eq!(tier_rank_base(0), 1000);
    assert_eq!(tier_rank_base(1), 2000);
    assert_eq!(tier_rank_base(2), 3000);
    assert_eq!(replica_rank(0, 0), 1000);
    assert_eq!(replica_rank(0, 1), 1001);
    assert_eq!(replica_rank(1, 0), 2000);

    let files = build(&example_input()).unwrap();
    let policy: serde_json::Value =
        serde_yaml::from_str(files.get("router-policy.yaml").unwrap()).unwrap();
    let models = &policy["worker_selection"]["instances"][0]["parameters"]["models"];
    let deployment = &models["zai-org/GLM-5.3"];
    assert_eq!(deployment["tiers"][0]["name"], "secondary");
    assert_eq!(
        deployment["tiers"][0]["dp_ranks"],
        serde_json::json!([1000, 1999])
    );
    assert_eq!(deployment["tiers"][1]["name"], "together");
    assert_eq!(
        deployment["tiers"][1]["dp_ranks"],
        serde_json::json!([2000, 2999])
    );

    let secondary_0: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/secondary-0.yaml").unwrap()).unwrap();
    let secondary_1: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/secondary-1.yaml").unwrap()).unwrap();
    let together_0: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/together-0.yaml").unwrap()).unwrap();
    assert_eq!(secondary_0.dp_rank, 1000);
    assert_eq!(secondary_1.dp_rank, 1001);
    assert_eq!(together_0.dp_rank, 2000);
    // Every proxy rank falls inside the range the policy reserves for its tier.
    assert_eq!(secondary_0.tier, "secondary");
    assert_eq!(together_0.tier, "together");
    assert!(secondary_0.router_config.is_some());
    assert!(together_0.router_config.is_some());
    assert_eq!(
        secondary_0.served_model_names,
        vec!["zai-org/GLM-5.3".to_string()]
    );
}

#[test]
fn rejects_deployment_missing_its_model_name() {
    let yaml = r#"
deployments:
  "zai-org/GLM-5.3":
    primary:
      engine: sglang
      primary_capacity_blocks: 1000
      occupancy_threshold: 0.9
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
    model:
      model_path: zai-org/GLM-5.3
      served_model_names: ["other/model"]
      namespace: dynamo
      component: backend
      endpoint: generate
      kv_block_size: 64
      context_length: 131072
      parser_family: glm47
    tiers:
      - name: secondary
        provider: {name: example-provider, base_url: https://x/v1, api_key_env: K, model: m}
        penalty_blocks: 200
        weight_blocks: 8
        replicas: 1
"#;
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("deployments.yaml");
    fs::write(&input, yaml).unwrap();
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("served_model_names"), "{error}");
    assert!(error.contains("zai-org/GLM-5.3"), "{error}");
}

#[test]
fn rejects_duplicate_tier_names() {
    let yaml = r#"
deployments:
  "org/m":
    primary:
      engine: sglang
      primary_capacity_blocks: 1000
      occupancy_threshold: 0.9
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
    model:
      model_path: m
      served_model_names: ["org/m"]
      namespace: dynamo
      component: backend
      endpoint: generate
      kv_block_size: 64
      context_length: 131072
      parser_family: glm47
    tiers:
      - name: secondary
        provider: {name: example-provider, base_url: https://x/v1, api_key_env: K, model: m}
        penalty_blocks: 200
        weight_blocks: 8
        replicas: 1
      - name: secondary
        provider: {name: other, base_url: https://y/v1, api_key_env: K2, model: m}
        penalty_blocks: 200
        weight_blocks: 40
        replicas: 1
"#;
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("deployments.yaml");
    fs::write(&input, yaml).unwrap();
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("duplicate tier name"), "{error}");
    assert!(error.contains("secondary"), "{error}");
}

/// A one-deployment YAML body with the given `primary.engine`, for the engine-matrix tests.
fn engine_yaml(engine: &str) -> String {
    let yaml = r#"
deployments:
  "org/m":
    primary:
      engine: __ENGINE__
      primary_capacity_blocks: 1000
      occupancy_threshold: 0.9
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
    model:
      model_path: org/model
      served_model_names: ["org/m"]
      namespace: dynamo
      component: backend
      endpoint: generate
      kv_block_size: 64
      context_length: 131072
      parser_family: glm47
    tiers:
      - name: secondary
        provider: {name: example-provider, base_url: https://x/v1, api_key_env: K, model: m}
        penalty_blocks: 200
        weight_blocks: 8
        replicas: 1
"#;
    yaml.replace("__ENGINE__", engine)
}

/// Build generated files from a YAML string written to a temporary input.
fn build_str(yaml: &str) -> BTreeMap<String, String> {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("deployments.yaml");
    fs::write(&input, yaml).unwrap();
    build(&input).unwrap()
}

#[test]
fn admission_margin_defaults_for_queue_reporting_engines() {
    for engine in ["sglang", "vllm"] {
        let files = build_str(&engine_yaml(engine));
        let env = files.get("admission/org_m/primary.env").unwrap();
        assert!(
            env.contains(&format!(
                "export DYN_ADMISSION_QUEUE_MARGIN={}",
                dw_spillover_deploy::DEFAULT_ADMISSION_QUEUE_MARGIN
            )),
            "{engine}: {env}"
        );
        assert!(env.contains(engine), "{engine}: {env}");
    }
}

#[test]
fn admission_margin_trtllm_is_emitted_with_publish_metrics_note() {
    let files = build_str(&engine_yaml("trtllm"));
    let env = files.get("admission/org_m/primary.env").unwrap();
    assert!(
        env.contains(&format!(
            "export DYN_ADMISSION_QUEUE_MARGIN={}",
            dw_spillover_deploy::DEFAULT_ADMISSION_QUEUE_MARGIN
        )),
        "{env}"
    );
    assert!(env.contains("--publish-metrics"), "{env}");
}

#[test]
fn admission_margin_is_unset_for_engines_that_never_report_waiting() {
    for engine in ["mocker", "tokenspeed"] {
        let files = build_str(&engine_yaml(engine));
        let env = files.get("admission/org_m/primary.env").unwrap();
        assert!(
            env.contains("unset DYN_ADMISSION_QUEUE_MARGIN"),
            "{engine}: {env}"
        );
        assert!(
            !env.contains("export DYN_ADMISSION_QUEUE_MARGIN"),
            "{engine}: {env}"
        );
    }
}

#[test]
fn explicit_admission_margin_is_rejected_for_engines_that_never_report_waiting() {
    for engine in ["mocker", "tokenspeed"] {
        let yaml = engine_yaml(engine).replace(
            "occupancy_threshold: 0.9",
            "occupancy_threshold: 0.9\n      admission_queue_margin: 64",
        );
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("deployments.yaml");
        fs::write(&input, yaml).unwrap();
        let error = format!("{:#}", build(&input).unwrap_err());
        assert!(
            error.contains("admission_queue_margin"),
            "{engine}: {error}"
        );
        assert!(error.contains(engine), "{engine}: {error}");
    }
}

#[test]
fn multiple_served_names_are_rejected_for_engines_without_aliases() {
    let with_alias = |engine: &str| {
        engine_yaml(engine).replace(
            "served_model_names: [\"org/m\"]",
            "served_model_names: [\"org/m\", \"org/m-alias\"]",
        )
    };
    for engine in ["trtllm", "mocker", "tokenspeed"] {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("deployments.yaml");
        fs::write(&input, with_alias(engine)).unwrap();
        let error = format!("{:#}", build(&input).unwrap_err());
        assert!(error.contains("served_model_names"), "{engine}: {error}");
        assert!(error.contains(engine), "{engine}: {error}");
    }
    // SGLang and vLLM register aliases, so the same input builds.
    for engine in ["sglang", "vllm"] {
        let files = build_str(&with_alias(engine));
        let proxy = first_proxy_config(&files);
        assert_eq!(proxy.served_model_names.len(), 2, "{engine}");
    }
}

/// Load the first generated proxy config from a `build` result.
fn first_proxy_config(files: &BTreeMap<String, String>) -> ProxyConfig {
    let relative = files
        .keys()
        .find(|path| path.ends_with(".yaml") && *path != "router-policy.yaml")
        .expect("a generated proxy config");
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("proxy.yaml");
    fs::write(&path, &files[relative]).unwrap();
    ProxyConfig::load(&path).unwrap()
}

#[test]
fn tokenspeed_advertises_no_card_router_config_and_gets_no_args_file() {
    let files = build_str(&engine_yaml("tokenspeed"));
    assert!(
        !files.contains_key("router/org_m/primary.args"),
        "tokenspeed must not emit a primary router args file"
    );
    assert!(
        first_proxy_config(&files).router_config.is_none(),
        "tokenspeed proxies must not advertise router_config"
    );
    let frontend_env = files.get("frontend.env").unwrap();
    assert!(
        frontend_env.contains("--router-track-active-blocks")
            && frontend_env.contains("tokenspeed"),
        "{frontend_env}"
    );
}

#[test]
fn proxy_engines_advertise_the_shared_router_config() {
    for engine in ["sglang", "vllm", "trtllm", "mocker"] {
        let files = build_str(&engine_yaml(engine));
        assert!(files.contains_key("router/org_m/primary.args"), "{engine}");
        let config = first_proxy_config(&files).router_config.unwrap();
        assert_eq!(config.mode, ProxyRouterMode::Kv, "{engine}");
        assert!(config.track_active_blocks, "{engine}");
        assert!(!config.track_output_blocks, "{engine}");
    }
}

#[test]
fn model_passthrough_fields_reach_every_proxy_config() {
    let yaml = engine_yaml("sglang")
        .replace(
            "      kv_block_size: 64\n",
            "      kv_block_size: 64\n      custom_jinja_template: /models/template.jinja\n      enable_eagle: true\n",
        )
        .replace("      context_length: 131072\n", "");
    let files = build_str(&yaml);
    let proxy = first_proxy_config(&files);
    assert_eq!(
        proxy.custom_jinja_template,
        Some(PathBuf::from("/models/template.jinja"))
    );
    assert!(proxy.enable_eagle);
    assert_eq!(proxy.context_length, None);
    let raw = files
        .iter()
        .find(|(path, _)| path.ends_with(".yaml") && *path != "router-policy.yaml")
        .map(|(_, contents)| contents)
        .unwrap();
    assert!(raw.contains("custom_jinja_template"));
    assert!(raw.contains("enable_eagle: true"));
    assert!(!raw.contains("context_length"), "{raw}");
}

#[test]
fn vllm_omitted_context_length_and_eagle_warn() {
    let yaml = engine_yaml("vllm")
        .replace(
            "      kv_block_size: 64\n",
            "      kv_block_size: 64\n      enable_eagle: true\n",
        )
        .replace("      context_length: 131072\n", "");
    let files = build_str(&yaml);
    let doc: DeploymentsFile = serde_yaml::from_str(&yaml).unwrap();
    let warnings = dw_spillover_deploy::engine_field_warnings(&doc);
    assert!(
        warnings.iter().any(|w| w.contains("max_model_len")),
        "{warnings:?}"
    );
    assert!(warnings.iter().any(|w| w.contains("EAGLE")), "{warnings:?}");
    assert_eq!(first_proxy_config(&files).context_length, None);
}

#[test]
fn sglang_context_length_is_optional_and_omitted_from_yaml_when_unset() {
    let yaml = engine_yaml("sglang").replace("      context_length: 131072\n", "");
    let files = build_str(&yaml);
    assert_eq!(first_proxy_config(&files).context_length, None);
    // With no `context_length` the soft-cap warning is skipped, not guessed.
    let doc: DeploymentsFile = serde_yaml::from_str(&yaml).unwrap();
    let deployment = doc.deployments.values().next().unwrap();
    assert_eq!(hard_cap_failover_penalty(deployment), None);
}

/// A deployment mapping body (no `deployments:` key) with a primary worker and one proxy
/// replica per given tier name. Used by the validation and pruning tests.
fn deployment_block(name: &str, tier_names: &[&str]) -> String {
    let tiers: String = tier_names
        .iter()
        .enumerate()
        .map(|(index, tier)| {
            format!(
                "      - name: {tier}\n        provider: {{name: p{index}, base_url: https://x/v1, api_key_env: K, model: m}}\n        penalty_blocks: 200\n        weight_blocks: 8\n        replicas: 1\n"
            )
        })
        .collect();
    format!(
        "  \"{name}\":\n    primary:\n      engine: sglang\n      primary_capacity_blocks: 1000\n      occupancy_threshold: 0.9\n      failover_penalty_blocks: 200\n      pending_weight_blocks: 4\n    model:\n      model_path: m\n      served_model_names: [\"{name}\"]\n      namespace: dynamo\n      component: backend\n      endpoint: generate\n      kv_block_size: 64\n      context_length: 131072\n      parser_family: glm47\n    tiers:\n{tiers}"
    )
}

fn deployment_yaml(name: &str, tier_names: &[&str]) -> String {
    format!("deployments:\n{}", deployment_block(name, tier_names))
}

fn write_input(dir: &Path, yaml: &str) -> PathBuf {
    let input = dir.join("deployments.yaml");
    fs::write(&input, yaml).unwrap();
    input
}

#[test]
fn sanitized_name_collisions_are_rejected() {
    // Tier names `a/b` and `a_b` are distinct raw names but map to the same file stem.
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(temp.path(), &deployment_yaml("org/m", &["a/b", "a_b"]));
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("sanitizing"), "{error}");
    assert!(error.contains("a/b") && error.contains("a_b"), "{error}");

    // Deployment names `org/m` and `org:m` map to the same directory.
    let temp = tempfile::tempdir().unwrap();
    let yaml = format!(
        "deployments:\n{}{}",
        deployment_block("org/m", &["secondary"]),
        deployment_block("org:m", &["secondary"])
    );
    let input = write_input(temp.path(), &yaml);
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("sanitizing"), "{error}");
    assert!(error.contains("org:m"), "{error}");
}

#[test]
fn path_traversal_names_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(temp.path(), &deployment_yaml("..", &["secondary"]));
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("unsafe path component"), "{error}");

    let temp = tempfile::tempdir().unwrap();
    let input = write_input(temp.path(), &deployment_yaml("org/m", &[".."]));
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("unsafe path component"), "{error}");
}

#[test]
fn rejects_zero_admission_margin() {
    let temp = tempfile::tempdir().unwrap();
    let yaml = deployment_yaml("org/m", &["secondary"]).replace(
        "pending_weight_blocks: 4",
        "pending_weight_blocks: 4\n      admission_queue_margin: 0",
    );
    let input = write_input(temp.path(), &yaml);
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("admission_queue_margin"), "{error}");
}

/// `primary_capacity_blocks` is now a fallback: it may be omitted so the policy
/// reads the advertised `total_kv_blocks`, and `primary_max_requests` passes
/// through when set.
#[test]
fn primary_capacity_is_optional_and_max_requests_passes_through() {
    let temp = tempfile::tempdir().unwrap();
    let yaml = r#"
deployments:
  "org/m":
    primary:
      engine: sglang
      occupancy_threshold: 0.9
      primary_max_requests: 64
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
    model:
      model_path: m
      served_model_names: ["org/m"]
      namespace: dynamo
      component: backend
      endpoint: generate
      kv_block_size: 64
      context_length: 131072
      parser_family: glm47
    tiers:
      - name: secondary
        provider: {name: example-provider, base_url: https://x/v1, api_key_env: K, model: m}
        penalty_blocks: 200
        weight_blocks: 8
        replicas: 1
"#;
    let input = write_input(temp.path(), yaml);
    let files = build(&input).unwrap();
    let policy: serde_json::Value =
        serde_yaml::from_str(files.get("router-policy.yaml").unwrap()).unwrap();
    let model = &policy["worker_selection"]["instances"][0]["parameters"]["models"]["org/m"];
    assert!(
        model.get("primary_capacity_blocks").is_none(),
        "omitted fallback must not be emitted: {model:?}"
    );
    assert_eq!(model["primary_max_requests"], serde_json::json!(64));
}

/// Values above 1.0 are allowed up to 4; the generator accepts them (and warns on
/// stderr) so a deliberate over-subscription only has to satisfy the policy bounds.
#[test]
fn occupancy_threshold_allows_up_to_four_and_rejects_more() {
    let temp = tempfile::tempdir().unwrap();
    let over = deployment_yaml("org/m", &["secondary"])
        .replace("occupancy_threshold: 0.9", "occupancy_threshold: 1.5");
    build(&write_input(temp.path(), &over)).unwrap();

    let too_big = deployment_yaml("org/m", &["secondary"])
        .replace("occupancy_threshold: 0.9", "occupancy_threshold: 4.5");
    let error = format!(
        "{:#}",
        build(&write_input(temp.path(), &too_big)).unwrap_err()
    );
    assert!(error.contains("occupancy_threshold"), "{error}");
}

#[test]
fn rejects_nonpositive_primary_capacity_and_max_requests() {
    let temp = tempfile::tempdir().unwrap();
    let zero_capacity = deployment_yaml("org/m", &["secondary"]).replace(
        "primary_capacity_blocks: 1000",
        "primary_capacity_blocks: 0",
    );
    let error = format!(
        "{:#}",
        build(&write_input(temp.path(), &zero_capacity)).unwrap_err()
    );
    assert!(error.contains("primary_capacity_blocks"), "{error}");

    let zero_requests = deployment_yaml("org/m", &["secondary"]).replace(
        "pending_weight_blocks: 4",
        "pending_weight_blocks: 4\n      primary_max_requests: 0",
    );
    let error = format!(
        "{:#}",
        build(&write_input(temp.path(), &zero_requests)).unwrap_err()
    );
    assert!(error.contains("primary_max_requests"), "{error}");
}

/// Finding 5: `build()` must reject every numeric field the policy would reject later, so it
/// can never return a tree that `validate_dir` then fails. Covers tier costs, primary costs,
/// and the zero model-card divisors.
#[test]
fn rejects_costs_outside_policy_bounds_and_zero_model_card_fields() {
    let cases: [(&str, &str, &str); 7] = [
        (
            "penalty_blocks: 200",
            "penalty_blocks: -1",
            "penalty_blocks",
        ),
        (
            "weight_blocks: 8",
            "weight_blocks: 100000000000",
            "weight_blocks",
        ),
        (
            "failover_penalty_blocks: 200",
            "failover_penalty_blocks: -1",
            "failover_penalty_blocks",
        ),
        (
            "pending_weight_blocks: 4",
            "pending_weight_blocks: 100000000000",
            "pending_weight_blocks",
        ),
        ("kv_block_size: 64", "kv_block_size: 0", "kv_block_size"),
        (
            "context_length: 131072",
            "context_length: 0",
            "context_length",
        ),
        (
            "failover_penalty_blocks: 200",
            "failover_penalty_blocks: 100000000000",
            "failover_penalty_blocks",
        ),
    ];
    for (from, to, field) in cases {
        let temp = tempfile::tempdir().unwrap();
        let yaml = deployment_yaml("org/m", &["secondary"]).replace(from, to);
        assert_ne!(yaml, deployment_yaml("org/m", &["secondary"]), "{field}");
        let error = format!("{:#}", build(&write_input(temp.path(), &yaml)).unwrap_err());
        assert!(
            error.contains(field),
            "expected {field} in error, got: {error}"
        );
    }
}

#[test]
fn generate_prunes_stale_files() {
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(
        temp.path(),
        &deployment_yaml("org/m", &["secondary", "together"]),
    );
    let out = temp.path().join("out");
    generate(&input, &out).unwrap();
    assert!(out.join("org_m/together-0.yaml").is_file());

    fs::write(&input, deployment_yaml("org/m", &["secondary"])).unwrap();
    generate(&input, &out).unwrap();
    assert!(
        !out.join("org_m/together-0.yaml").exists(),
        "a dropped tier's proxy config is pruned"
    );
    assert!(out.join("org_m/secondary-0.yaml").is_file());

    // Only files the previous run recorded are pruned; unrelated files survive.
    let notes = out.join("org_m/operator-notes.yaml");
    fs::write(&notes, "notes: true\n").unwrap();
    generate(&input, &out).unwrap();
    assert!(notes.is_file(), "unrelated file must not be pruned");
}

/// `generate` validates a staged tree before touching `--out`, and prunes stale files only
/// after the new tree is written, so a failed write cannot delete the last-good tree.
#[test]
fn failed_generate_does_not_prune_the_previous_tree() {
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(
        temp.path(),
        &deployment_yaml("org/m", &["secondary", "together"]),
    );
    let out = temp.path().join("out");
    generate(&input, &out).unwrap();
    let together = out.join("org_m/together-0.yaml");
    assert!(together.is_file());

    // Make writing the new tree fail: a generated file path is now a directory.
    let blocked = out.join("org_m/secondary-0.yaml");
    fs::remove_file(&blocked).unwrap();
    fs::create_dir(&blocked).unwrap();

    fs::write(&input, deployment_yaml("org/m", &["secondary"])).unwrap();
    assert!(
        generate(&input, &out).is_err(),
        "writing over a directory must fail"
    );
    assert!(
        together.is_file(),
        "a failed generate must not prune the previous tree first"
    );
}

#[test]
fn validate_dir_ignores_unrelated_yaml() {
    let temp = tempfile::tempdir().unwrap();
    let files = build(&example_input()).unwrap();
    write_files(temp.path(), &files).unwrap();
    fs::write(
        temp.path().join("k8s-manifest.yaml"),
        "kind: Deployment\nmetadata: {name: x}\n",
    )
    .unwrap();
    validate_dir(temp.path()).unwrap();
}

#[test]
fn validate_dir_rejects_rank_outside_tiers() {
    let temp = tempfile::tempdir().unwrap();
    let files = build(&example_input()).unwrap();
    write_files(temp.path(), &files).unwrap();
    let path = temp.path().join("zai-org_GLM-5.3/secondary-0.yaml");
    let raw = fs::read_to_string(&path).unwrap();
    assert!(raw.contains("dp_rank: 1000"), "{raw}");
    fs::write(&path, raw.replace("dp_rank: 1000", "dp_rank: 9000")).unwrap();
    let error = format!("{:#}", validate_dir(temp.path()).unwrap_err());
    assert!(error.contains("9000"), "{error}");
}

#[test]
fn committed_output_is_not_stale() {
    let generated = build(&example_input()).unwrap();
    let committed = read_dir_files(&generated_dir());
    assert_eq!(
        generated.keys().collect::<Vec<_>>(),
        committed.keys().collect::<Vec<_>>(),
        "generated/ file set differs; run `spillover-deploy generate`"
    );
    for (path, expected) in &generated {
        assert_eq!(
            committed.get(path),
            Some(expected),
            "generated/{path} is stale; run `spillover-deploy generate`"
        );
    }
}

#[test]
fn a_tier_thinking_dialect_reaches_its_proxy_configs() {
    let temp = tempfile::tempdir().unwrap();
    let yaml = deployment_yaml("org/m", &["secondary"]).replace(
        "api_key_env: K, model: m}",
        "api_key_env: K, model: m, thinking_dialect: reasoning_object, thinking_strict: true}",
    );
    let input = write_input(temp.path(), &yaml);
    let files = build(&input).unwrap();
    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("org_m/secondary-0.yaml").unwrap()).unwrap();
    assert_eq!(
        proxy.provider.thinking_dialect,
        dw_proxy_core::thinking::ThinkingDialect::ReasoningObject
    );
    assert!(proxy.provider.thinking_strict);

    // An unknown dialect fails generation, not the proxy at startup.
    let bad = deployment_yaml("org/m", &["secondary"]).replace(
        "api_key_env: K, model: m}",
        "api_key_env: K, model: m, thinking_dialect: reasoning}",
    );
    let bad_input = write_input(temp.path(), &bad);
    assert!(build(&bad_input).is_err());
}

#[test]
fn a_tier_cache_key_needs_its_secret_and_reaches_the_proxy() {
    let temp = tempfile::tempdir().unwrap();
    let missing = deployment_yaml("org/m", &["secondary"]).replace(
        "api_key_env: K, model: m}",
        "api_key_env: K, model: m, cache_key: prompt_cache_key}",
    );
    let error = format!(
        "{:#}",
        build(&write_input(temp.path(), &missing)).unwrap_err()
    );
    assert!(error.contains("cache_key_secret_env"), "{error}");

    let complete = deployment_yaml("org/m", &["secondary"]).replace(
        "api_key_env: K, model: m}",
        "api_key_env: K, model: m, cache_key: prompt_cache_key, cache_key_secret_env: CK}",
    );
    let files = build(&write_input(temp.path(), &complete)).unwrap();
    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("org_m/secondary-0.yaml").unwrap()).unwrap();
    assert_eq!(
        proxy.provider.cache_key,
        dw_proxy_core::cache_key::CacheKeyField::PromptCacheKey
    );
    assert_eq!(proxy.provider.cache_key_secret_env.as_deref(), Some("CK"));
}

#[test]
fn a_tier_circuit_breaker_reaches_its_proxy_configs_and_is_validated() {
    let temp = tempfile::tempdir().unwrap();
    let yaml = deployment_yaml("org/m", &["secondary"]).replace(
        "api_key_env: K, model: m}",
        "api_key_env: K, model: m, circuit_breaker: {failure_threshold: 3, cooldown_ms: 500, \
         max_cooldown_ms: 2000}}",
    );
    let files = build(&write_input(temp.path(), &yaml)).unwrap();
    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("org_m/secondary-0.yaml").unwrap()).unwrap();
    let breaker = proxy
        .provider
        .circuit_breaker
        .expect("circuit breaker passed through");
    assert_eq!(breaker.failure_threshold, 3);
    assert_eq!(breaker.cooldown_ms, 500);
    assert_eq!(breaker.max_cooldown_ms, 2_000);

    // Bad values pass the generator's own input checks but are caught when the
    // generated tree is validated, exactly as a hand-written proxy config is.
    let bad = deployment_yaml("org/m", &["secondary"]).replace(
        "api_key_env: K, model: m}",
        "api_key_env: K, model: m, circuit_breaker: {max_cooldown_ms: 100}}",
    );
    let error = format!("{:#}", check(&write_input(temp.path(), &bad)).unwrap_err());
    assert!(error.contains("max_cooldown_ms"), "{error}");
}

#[test]
fn proxies_omit_the_source_path_only_where_the_primary_records_none() {
    let proxy_yaml = |engine: &str, model_path: &str| {
        let yaml = engine_yaml(engine).replace(
            "      model_path: org/model\n",
            &format!("      model_path: {model_path}\n"),
        );
        let files = build_str(&yaml);
        files
            .iter()
            .find(|(path, _)| path.ends_with("secondary-0.yaml"))
            .map(|(_, body)| body.clone())
            .expect("proxy config")
    };
    // The mocker's make_engine entrypoint records no source path for a local model directory.
    assert!(proxy_yaml("mocker", "/models/m").contains("omit_source_path: true"));
    // It records a Hugging Face id, as every other engine records any model string.
    assert!(!proxy_yaml("mocker", "org/m").contains("omit_source_path"));
    for engine in ["sglang", "vllm", "trtllm", "tokenspeed"] {
        assert!(
            !proxy_yaml(engine, "/models/m").contains("omit_source_path"),
            "{engine}"
        );
    }
}

#[test]
fn generate_rejects_a_previous_manifest_that_escapes_out() {
    // Finding 1: `validate_dir` only inspected the tree being staged, so a corrupt or
    // hand-edited previous `.generated-files` naming an absolute path or `..` could make
    // `remove_files` delete outside `--out`. Every manifest entry is now validated against
    // the same "relative, normal components only" rule as `write_files`.
    for entry in ["../escape", "/tmp/escape"] {
        let temp = tempfile::tempdir().unwrap();
        let input = write_input(temp.path(), &deployment_yaml("org/m", &["secondary"]));
        let out = temp.path().join("out");
        generate(&input, &out).unwrap();

        let outside = temp.path().join("escape");
        if entry.starts_with('/') {
            // Point the absolute entry at a real file we own so the test can check it.
            fs::write(&outside, "keep me\n").unwrap();
            let manifest = format!("{}\n", outside.display());
            fs::write(out.join(".generated-files"), manifest).unwrap();
        } else {
            fs::write(&outside, "keep me\n").unwrap();
            fs::write(out.join(".generated-files"), "../escape\n").unwrap();
        }

        let error = format!("{:#}", generate(&input, &out).unwrap_err());
        assert!(error.contains("outside"), "entry {entry:?}: {error}");
        assert!(
            outside.is_file(),
            "entry {entry:?}: a file outside --out must not be pruned"
        );
    }
}

#[test]
fn mocker_rejects_an_ambiguous_relative_model_path() {
    // Finding 2: the mocker reads `Path(model_path).exists()` to choose local vs hub, but
    // the generator cannot stat at generate time. A relative path with a leading `./` or
    // `../`, or a single segment, is therefore ambiguous and rejected; an `org/name` hub id
    // or an absolute filesystem path is accepted.
    for model_path in ["./models/m", "../models/m", "m"] {
        let yaml = engine_yaml("mocker").replace(
            "      model_path: org/model\n",
            &format!("      model_path: {model_path}\n"),
        );
        let temp = tempfile::tempdir().unwrap();
        let error = format!("{:#}", build(&write_input(temp.path(), &yaml)).unwrap_err());
        assert!(error.contains("model_path"), "{model_path}: {error}");
    }
    for model_path in ["org/model", "/models/m"] {
        let yaml = engine_yaml("mocker").replace(
            "      model_path: org/model\n",
            &format!("      model_path: {model_path}\n"),
        );
        build_str(&yaml);
    }
}

#[test]
fn router_passthrough_fields_reach_primary_args_and_proxy_configs() {
    // Finding 3: the busy-worker thresholds and session-affinity settings flow into the
    // primary's `--active-*`/`--router-session-affinity-*` flags and into every proxy's card
    // `router_config`, so the two cards hash equal.
    let yaml = engine_yaml("sglang").replace(
        "      parser_family: glm47\n",
        concat!(
            "      parser_family: glm47\n",
            "      load_threshold_config:\n",
            "        active_decode_blocks_threshold: 0.75\n",
            "        active_prefill_tokens_threshold: 4096\n",
            "        active_prefill_tokens_threshold_frac: 0.5\n",
            "      session_affinity_ttl_secs: 60\n",
            "      session_affinity_mode: soft\n",
        ),
    );
    let files = build_str(&yaml);
    let args = files
        .get("router/org_m/primary.args")
        .expect("primary router args");
    for flag in [
        "--active-decode-blocks-threshold 0.75",
        "--active-prefill-tokens-threshold 4096",
        "--active-prefill-tokens-threshold-frac 0.5",
        "--router-session-affinity-ttl-secs 60",
        "--router-session-affinity-mode soft",
    ] {
        assert!(args.contains(flag), "{flag} missing from:\n{args}");
    }

    let proxy = files.get("org_m/secondary-0.yaml").expect("proxy config");
    for key in [
        "load_threshold_config:",
        "active_decode_blocks_threshold: 0.75",
        "active_prefill_tokens_threshold: 4096",
        "active_prefill_tokens_threshold_frac: 0.5",
        "session_affinity_ttl_secs: 60",
        "session_affinity_mode: soft",
    ] {
        assert!(proxy.contains(key), "{key} missing from:\n{proxy}");
    }
}

#[test]
fn router_passthrough_is_absent_when_unset() {
    // Finding 3: a deployment that does not configure the new fields must generate exactly
    // the previous YAML and args, so no card checksum changes for existing deployments.
    let files = build_str(&engine_yaml("sglang"));
    let proxy = files.get("org_m/secondary-0.yaml").unwrap();
    assert!(!proxy.contains("load_threshold_config"), "{proxy}");
    assert!(!proxy.contains("session_affinity"), "{proxy}");
    let args = files.get("router/org_m/primary.args").unwrap();
    assert!(!args.contains("--active-"), "{args}");
    assert!(!args.contains("--router-session-affinity"), "{args}");
}
