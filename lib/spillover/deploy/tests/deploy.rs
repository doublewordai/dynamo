// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generator tests: end-to-end validation, rank assignment, input rejection, staleness.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use dw_proxy_core::config::{ProxyConfig, ProxyRouterMode};
use dw_spillover_deploy::{
    build, check, generate, replica_rank, tier_rank_base, validate_dir, write_files,
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
    // 1 policy + 1 frontend note + 1 manifest + for the one deployment (1 hosted router args
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
        .get("router/zai-org_GLM-5.3/hosted.args")
        .expect("router/zai-org_GLM-5.3/hosted.args");
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

    let hosted_env = files
        .get("admission/zai-org_GLM-5.3/hosted.env")
        .expect("hosted admission env");
    assert!(
        hosted_env.contains("export DYN_ADMISSION_QUEUE_MARGIN=256"),
        "{hosted_env}"
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

/// Parse the emitted SGLang router flags into the fields the hosted model card carries.
///
/// This is deliberately independent of [`dw_spillover_deploy::ROUTER_ADVERTISEMENT`]: the
/// point of the test below is that the emitted flags and the proxy config agree, not that
/// both restate the same constant. It still does not run the SGLang Python CLI, so it cannot
/// prove the Python defaults match the Rust defaults; the cross-language check lives in
/// `proxy-worker`'s registration test.
fn parse_hosted_args(args: &str) -> (String, bool, bool, Option<String>) {
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

/// The generated proxy `router_config` and the emitted SGLang flags must describe
/// one advertisement, or the checksums differ and the worker set splits. This parses
/// the emitted flags rather than comparing them to the constant that produced them.
#[test]
fn hosted_args_and_proxy_router_config_agree() {
    let files = build(&example_input()).unwrap();
    let args = files.get("router/zai-org_GLM-5.3/hosted.args").unwrap();
    let flags: Vec<&str> = args
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .collect();
    let (mode, track_active, track_output, shared) = parse_hosted_args(flags[0]);
    // Spillover requires KV routing; the card checksum pins the shared-cache multiplier to
    // the SGLang CLI default.
    assert_eq!(mode, "kv");
    assert_eq!(shared.as_deref(), Some("0.5"));

    let proxy: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/openrouter-0.yaml").unwrap()).unwrap();
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
    assert_eq!(deployment["tiers"][0]["name"], "openrouter");
    assert_eq!(
        deployment["tiers"][0]["dp_ranks"],
        serde_json::json!([1000, 1999])
    );
    assert_eq!(deployment["tiers"][1]["name"], "together");
    assert_eq!(
        deployment["tiers"][1]["dp_ranks"],
        serde_json::json!([2000, 2999])
    );

    let openrouter_0: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/openrouter-0.yaml").unwrap()).unwrap();
    let openrouter_1: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/openrouter-1.yaml").unwrap()).unwrap();
    let together_0: ProxyConfig =
        serde_yaml::from_str(files.get("zai-org_GLM-5.3/together-0.yaml").unwrap()).unwrap();
    assert_eq!(openrouter_0.dp_rank, 1000);
    assert_eq!(openrouter_1.dp_rank, 1001);
    assert_eq!(together_0.dp_rank, 2000);
    // Every proxy rank falls inside the range the policy reserves for its tier.
    assert_eq!(openrouter_0.tier, "openrouter");
    assert_eq!(together_0.tier, "together");
    assert!(openrouter_0.router_config.is_some());
    assert!(together_0.router_config.is_some());
    assert_eq!(
        openrouter_0.served_model_names,
        vec!["zai-org/GLM-5.3".to_string()]
    );
}

#[test]
fn rejects_deployment_missing_its_model_name() {
    let yaml = r#"
deployments:
  "zai-org/GLM-5.3":
    hosted:
      hosted_capacity_blocks: 1000
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
      - name: openrouter
        provider: {name: openrouter, base_url: https://x/v1, api_key_env: K, model: m}
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
    hosted:
      hosted_capacity_blocks: 1000
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
      - name: openrouter
        provider: {name: openrouter, base_url: https://x/v1, api_key_env: K, model: m}
        penalty_blocks: 200
        weight_blocks: 8
        replicas: 1
      - name: openrouter
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
    assert!(error.contains("openrouter"), "{error}");
}

#[test]
fn admission_margin_defaults_and_overrides() {
    let yaml = r#"
deployments:
  "org/m":
    hosted:
      hosted_capacity_blocks: 1000
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
      - name: openrouter
        provider: {name: openrouter, base_url: https://x/v1, api_key_env: K, model: m}
        penalty_blocks: 200
        weight_blocks: 8
        replicas: 1
"#;
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("deployments.yaml");
    fs::write(&input, yaml).unwrap();
    let files = build(&input).unwrap();
    let env = files.get("admission/org_m/hosted.env").unwrap();
    assert!(
        env.contains(&format!(
            "DYN_ADMISSION_QUEUE_MARGIN={}",
            dw_spillover_deploy::DEFAULT_ADMISSION_QUEUE_MARGIN
        )),
        "{env}"
    );
}

/// A deployment mapping body (no `deployments:` key) with a hosted worker and one proxy
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
        "  \"{name}\":\n    hosted:\n      hosted_capacity_blocks: 1000\n      occupancy_threshold: 0.9\n      failover_penalty_blocks: 200\n      pending_weight_blocks: 4\n    model:\n      model_path: m\n      served_model_names: [\"{name}\"]\n      namespace: dynamo\n      component: backend\n      endpoint: generate\n      kv_block_size: 64\n      context_length: 131072\n      parser_family: glm47\n    tiers:\n{tiers}"
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
        deployment_block("org/m", &["openrouter"]),
        deployment_block("org:m", &["openrouter"])
    );
    let input = write_input(temp.path(), &yaml);
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("sanitizing"), "{error}");
    assert!(error.contains("org:m"), "{error}");
}

#[test]
fn path_traversal_names_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(temp.path(), &deployment_yaml("..", &["openrouter"]));
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
    let yaml = deployment_yaml("org/m", &["openrouter"]).replace(
        "pending_weight_blocks: 4",
        "pending_weight_blocks: 4\n      admission_queue_margin: 0",
    );
    let input = write_input(temp.path(), &yaml);
    let error = format!("{:#}", build(&input).unwrap_err());
    assert!(error.contains("admission_queue_margin"), "{error}");
}

#[test]
fn generate_prunes_stale_files() {
    let temp = tempfile::tempdir().unwrap();
    let input = write_input(
        temp.path(),
        &deployment_yaml("org/m", &["openrouter", "together"]),
    );
    let out = temp.path().join("out");
    generate(&input, &out).unwrap();
    assert!(out.join("org_m/together-0.yaml").is_file());

    fs::write(&input, deployment_yaml("org/m", &["openrouter"])).unwrap();
    generate(&input, &out).unwrap();
    assert!(
        !out.join("org_m/together-0.yaml").exists(),
        "a dropped tier's proxy config is pruned"
    );
    assert!(out.join("org_m/openrouter-0.yaml").is_file());

    // Only files the previous run recorded are pruned; unrelated files survive.
    let notes = out.join("org_m/operator-notes.yaml");
    fs::write(&notes, "notes: true\n").unwrap();
    generate(&input, &out).unwrap();
    assert!(notes.is_file(), "unrelated file must not be pruned");
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
    let path = temp.path().join("zai-org_GLM-5.3/openrouter-0.yaml");
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
