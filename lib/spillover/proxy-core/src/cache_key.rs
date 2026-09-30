// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An opaque, stable cache key for provider-side prompt caching.
//!
//! Some providers route requests that share a key to the same cache. The client's own
//! `prompt_cache_key` or `user` would do, but `user` identifies our customer's end users to a third
//! party. The proxy therefore sends a keyed hash of it instead: stable for the same input, so
//! requests group exactly as they would with the raw value, but not reversible or linkable
//! without the deployment's secret. The raw values are never forwarded.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Context string for deriving the hashing key from the configured secret.
const KEY_CONTEXT: &str = "dw-proxy provider cache key v1";
/// Hex characters of the hash sent to the provider (128 bits).
const KEY_HEX_CHARS: usize = 32;

/// Which request field the provider reads its cache routing key from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKeyField {
    /// Send no key.
    #[default]
    None,
    /// OpenAI's `prompt_cache_key`.
    PromptCacheKey,
    /// The `user` field, for providers that route on it.
    User,
}

impl CacheKeyField {
    fn body_field(self) -> Option<&'static str> {
        match self {
            CacheKeyField::None => None,
            CacheKeyField::PromptCacheKey => Some("prompt_cache_key"),
            CacheKeyField::User => Some("user"),
        }
    }
}

/// Derives and inserts the opaque key. Built once per process; each request costs one keyed
/// BLAKE3 hash of a short string.
#[derive(Clone)]
pub struct CacheKeyer {
    field: &'static str,
    key: [u8; 32],
}

impl std::fmt::Debug for CacheKeyer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheKeyer")
            .field("field", &self.field)
            .finish_non_exhaustive()
    }
}

impl CacheKeyer {
    /// `None` when `field` is [`CacheKeyField::None`].
    pub fn new(field: CacheKeyField, secret: &[u8]) -> Option<Self> {
        Some(Self {
            field: field.body_field()?,
            key: blake3::derive_key(KEY_CONTEXT, secret),
        })
    }

    /// The opaque key for a chat request: from its `prompt_cache_key`, else its `user`. `None`
    /// when the request carries neither, so unrelated requests are not grouped together.
    pub fn key_for(&self, request: &Value) -> Option<String> {
        let source = ["prompt_cache_key", "user"].iter().find_map(|name| {
            request
                .get(*name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
        })?;
        let hash = blake3::keyed_hash(&self.key, source.as_bytes());
        Some(hash.to_hex()[..KEY_HEX_CHARS].to_string())
    }

    /// Insert the opaque key into the provider body, if the request carries a source value.
    pub fn apply(&self, request: &Value, body: &mut Map<String, Value>) {
        if let Some(key) = self.key_for(request) {
            body.insert(self.field.to_string(), Value::String(key));
        }
    }
}
