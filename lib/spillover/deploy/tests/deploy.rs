//! Generator tests: end-to-end validation, rank assignment, input rejection, staleness.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use dw_proxy_core::config::ProxyConfig;
use dw_spillover_deploy::{build, check, replica_rank, tier_rank_base, validate_dir, write_files};

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
    assert_eq!(files.len(), 1 + 3 + 3);

    let temp = tempfile::tempdir().unwrap();
    write_files(temp.path(), &files).unwrap();
    validate_dir(temp.path()).unwrap();

    // `check` parses, generates and validates without touching the output directory.
    check(&example_input()).unwrap();
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
    let interactive = &models["zai-org/GLM-5.3@interactive"];
    assert_eq!(interactive["tiers"][0]["name"], "openrouter");
    assert_eq!(
        interactive["tiers"][0]["dp_ranks"],
        serde_json::json!([1000, 1999])
    );
    assert_eq!(interactive["tiers"][1]["name"], "together");
    assert_eq!(
        interactive["tiers"][1]["dp_ranks"],
        serde_json::json!([2000, 2999])
    );

    let openrouter_0: ProxyConfig = serde_yaml::from_str(
        files
            .get("zai-org_GLM-5.3_interactive/openrouter-0.yaml")
            .unwrap(),
    )
    .unwrap();
    let openrouter_1: ProxyConfig = serde_yaml::from_str(
        files
            .get("zai-org_GLM-5.3_interactive/openrouter-1.yaml")
            .unwrap(),
    )
    .unwrap();
    let together_0: ProxyConfig = serde_yaml::from_str(
        files
            .get("zai-org_GLM-5.3_interactive/together-0.yaml")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(openrouter_0.dp_rank, 1000);
    assert_eq!(openrouter_1.dp_rank, 1001);
    assert_eq!(together_0.dp_rank, 2000);
    // Every proxy rank falls inside the range the policy reserves for its tier.
    assert_eq!(openrouter_0.tier, "openrouter");
    assert_eq!(together_0.tier, "together");
    assert_eq!(
        openrouter_0.served_model_names,
        vec![
            "zai-org/GLM-5.3@interactive".to_string(),
            "zai-org/GLM-5.3".to_string()
        ]
    );
}

#[test]
fn rejects_deployment_missing_its_model_name() {
    let yaml = r#"
deployments:
  "zai-org/GLM-5.3@interactive":
    hosted:
      hosted_capacity_blocks: 1000
      occupancy_threshold: 0.9
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
    model:
      model_path: zai-org/GLM-5.3
      served_model_names: ["zai-org/GLM-5.3@throughput"]
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
    assert!(error.contains("zai-org/GLM-5.3@interactive"), "{error}");
}

#[test]
fn rejects_duplicate_tier_names() {
    let yaml = r#"
deployments:
  "m@interactive":
    hosted:
      hosted_capacity_blocks: 1000
      occupancy_threshold: 0.9
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
    model:
      model_path: m
      served_model_names: ["m@interactive"]
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
