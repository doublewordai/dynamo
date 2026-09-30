// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading the proxy worker's YAML config. Errors fail at process start, before the Dynamo
//! runtime is built.

use std::path::Path;

use dw_proxy_core::config::ProxyConfig;

/// Read, parse and validate `path`.
pub fn load(path: &Path) -> anyhow::Result<ProxyConfig> {
    ProxyConfig::load(path)
}

/// Parse a YAML document into [`ProxyConfig`] without validating it.
#[cfg(test)]
fn from_yaml(text: &str) -> anyhow::Result<ProxyConfig> {
    use anyhow::Context;
    serde_yaml::from_str(text).context("invalid proxy config YAML")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dw_proxy_core::render::ParserFamily;

    const SAMPLE: &str = r#"
model_path: /models/glm-5.3
served_model_names:
  - zai-org/GLM-5.3
  - glm-5.3
namespace: dynamo
component: backend
endpoint: generate
kv_block_size: 64
context_length: 202752
dp_rank: 7
tier: spillover
parser_family: glm47
provider:
  name: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  model: z-ai/glm-5.3
"#;

    #[test]
    fn parses_required_fields_and_defaults() {
        let cfg = from_yaml(SAMPLE).expect("sample parses");
        assert_eq!(cfg.model_path, "/models/glm-5.3");
        assert_eq!(cfg.served_model_names.len(), 2);
        assert_eq!(cfg.kv_block_size, 64);
        assert_eq!(cfg.context_length, 202752);
        assert_eq!(cfg.dp_rank, 7);
        assert_eq!(cfg.tier, "spillover");
        assert_eq!(cfg.parser_family, ParserFamily::Glm47);
        assert_eq!(cfg.provider.name, "openrouter");
        assert_eq!(cfg.provider.model, "z-ai/glm-5.3");
        // Optional cache settings fall back to the documented defaults.
        assert_eq!(cfg.vcache_ttl_secs, 300);
        assert_eq!(cfg.vcache_max_blocks, 1_000_000);
    }

    #[test]
    fn rejects_unknown_fields() {
        let bad = format!("{SAMPLE}\nunexpected: true\n");
        let err = from_yaml(&bad).expect_err("unknown field must be rejected");
        assert!(err.to_string().contains("parse proxy config") || err.to_string().contains("YAML"));
    }

    #[test]
    fn rejects_missing_provider() {
        let without_provider: String = SAMPLE
            .lines()
            .take_while(|l| !l.starts_with("provider:"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(from_yaml(&without_provider).is_err());
    }
}
