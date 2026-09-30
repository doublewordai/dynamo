// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ProxyConfig` load and validation.

use std::path::{Path, PathBuf};

use dw_proxy_core::config::ProxyConfig;
use dw_proxy_core::render::ParserFamily;
use dw_proxy_core::upstream::ProviderConfig;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn valid() -> ProxyConfig {
    ProxyConfig::load(&fixture("proxy.example.yaml")).expect("fixture must load")
}

/// Write `contents` to a fresh temporary file and return its path.
fn temp_yaml(name: &str, contents: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "dw-proxy-core-config-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.yaml");
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn loads_full_example() {
    let config = valid();

    assert_eq!(config.model_path, "zai-org/GLM-5.3");
    assert_eq!(
        config.served_model_names,
        vec!["zai-org/GLM-5.3".to_string(), "glm-5.3".to_string()]
    );
    assert_eq!(config.namespace, "dynamo");
    assert_eq!(config.component, "backend");
    assert_eq!(config.endpoint, "generate");
    assert_eq!(config.kv_block_size, 64);
    assert_eq!(config.context_length, 131_072);
    assert_eq!(config.dp_rank, 7);
    assert_eq!(config.tier, "proxy");
    assert_eq!(config.parser_family, ParserFamily::Glm47);
    assert_eq!(
        config.provider,
        ProviderConfig {
            name: "openrouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model: "z-ai/glm-5.3".to_string(),
            provider_preferences: None,
            body_overrides: None,
            extra_headers: Default::default(),
            connect_timeout_ms: 10_000,
            read_timeout_ms: 300_000,
        }
    );
    assert_eq!(config.vcache_ttl_secs, 600);
    assert_eq!(config.vcache_max_blocks, 2048);

    config.validate().expect("example must be valid");
}

#[test]
fn vcache_defaults_apply_when_omitted() {
    let yaml = r#"
model_path: m
served_model_names: [m]
namespace: dynamo
component: backend
endpoint: generate
kv_block_size: 16
context_length: 4096
dp_rank: 1
tier: proxy
parser_family: hermes
provider:
  name: p
  base_url: http://127.0.0.1:8080/v1
  api_key_env: KEY
  model: upstream-model
"#;
    let config = ProxyConfig::load(&temp_yaml("defaults", yaml)).unwrap();
    assert_eq!(config.vcache_ttl_secs, 300);
    assert_eq!(config.vcache_max_blocks, 1_000_000);
    assert_eq!(config.provider.connect_timeout_ms, 10_000);
    assert_eq!(config.provider.read_timeout_ms, 120_000);
    assert_eq!(config.provider.extra_headers.len(), 0);
}

#[test]
fn unknown_field_is_rejected() {
    let yaml = std::fs::read_to_string(fixture("proxy.example.yaml")).unwrap();
    let with_extra = format!("{yaml}\nnot_a_real_field: true\n");
    let err = ProxyConfig::load(&temp_yaml("unknown", &with_extra)).unwrap_err();
    let text = format!("{err:#}");
    assert!(
        text.contains("not_a_real_field"),
        "unexpected error: {text}"
    );
}

#[test]
fn unknown_provider_field_is_rejected() {
    let yaml = std::fs::read_to_string(fixture("proxy.example.yaml"))
        .unwrap()
        .replace(
            "  connect_timeout_ms: 10000",
            "  connect_timeout_ms: 10000\n  surprise: 1",
        );
    let err = ProxyConfig::load(&temp_yaml("unknown-provider", &yaml)).unwrap_err();
    let text = format!("{err:#}");
    assert!(text.contains("surprise"), "unexpected error: {text}");
}

#[test]
fn empty_model_path_is_rejected() {
    let mut config = valid();
    config.model_path = "  ".to_string();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("model_path")
    );
}

#[test]
fn empty_served_model_names_is_rejected() {
    let mut config = valid();
    config.served_model_names = Vec::new();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("served_model_names")
    );

    config.served_model_names = vec!["zai-org/GLM-5.3".to_string(), " ".to_string()];
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("served_model_names")
    );
}

#[test]
fn empty_namespace_component_endpoint_are_rejected() {
    let mut config = valid();
    config.namespace = String::new();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("namespace")
    );

    let mut config = valid();
    config.component = " ".to_string();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("component")
    );

    let mut config = valid();
    config.endpoint = String::new();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("endpoint")
    );
}

#[test]
fn zero_block_size_context_length_and_dp_rank_are_rejected() {
    let mut config = valid();
    config.kv_block_size = 0;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("kv_block_size")
    );

    let mut config = valid();
    config.context_length = 0;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("context_length")
    );

    let mut config = valid();
    config.dp_rank = 0;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("dp_rank")
    );
}

#[test]
fn max_dp_rank_is_rejected() {
    let mut config = valid();
    config.dp_rank = u32::MAX;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("dp_rank"),
    );
}

#[test]
fn empty_tier_is_rejected() {
    let mut config = valid();
    config.tier = String::new();
    assert!(config.validate().unwrap_err().to_string().contains("tier"));
}

#[test]
fn empty_provider_fields_are_rejected() {
    let mut config = valid();
    config.provider.name = String::new();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("provider.name")
    );

    let mut config = valid();
    config.provider.base_url = " ".to_string();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("provider.base_url")
    );

    let mut config = valid();
    config.provider.api_key_env = String::new();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("provider.api_key_env")
    );

    let mut config = valid();
    config.provider.model = String::new();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("provider.model")
    );
}

#[test]
fn load_validates() {
    let yaml = std::fs::read_to_string(fixture("proxy.example.yaml"))
        .unwrap()
        .replace("dp_rank: 7", "dp_rank: 0");
    let err = ProxyConfig::load(&temp_yaml("invalid", &yaml)).unwrap_err();
    assert!(
        err.to_string().contains("dp_rank"),
        "unexpected error: {err}"
    );
}

#[test]
fn non_https_base_url_is_rejected() {
    let mut config = valid();
    config.provider.base_url = "http://openrouter.ai/api/v1".to_string();
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("provider.base_url"), "{error}");
}

#[test]
fn malformed_base_url_is_rejected() {
    let mut config = valid();
    config.provider.base_url = "not a url".to_string();
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("provider.base_url"), "{error}");
}

#[test]
fn base_url_with_query_or_fragment_is_rejected() {
    let mut config = valid();
    config.provider.base_url = "https://openrouter.ai/api/v1?key=secret".to_string();
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("query"), "{error}");

    let mut config = valid();
    config.provider.base_url = "https://openrouter.ai/api/v1#frag".to_string();
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("query"), "{error}");
}

#[test]
fn loopback_http_base_url_is_allowed() {
    let mut config = valid();
    config.provider.base_url = "http://127.0.0.1:8080/v1".to_string();
    config.validate().unwrap();

    let mut config = valid();
    config.provider.base_url = "http://localhost:8080/v1".to_string();
    config.validate().unwrap();
}

#[test]
fn zero_provider_timeouts_are_rejected() {
    let mut config = valid();
    config.provider.connect_timeout_ms = 0;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("connect_timeout_ms"), "{error}");

    let mut config = valid();
    config.provider.read_timeout_ms = 0;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("read_timeout_ms"), "{error}");
}

#[test]
fn zero_vcache_settings_are_rejected() {
    let mut config = valid();
    config.vcache_ttl_secs = 0;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("vcache_ttl_secs"), "{error}");

    let mut config = valid();
    config.vcache_max_blocks = 0;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("vcache_max_blocks"), "{error}");
}
