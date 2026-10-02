// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The opaque provider cache key.

use dw_proxy_core::cache_key::{CacheKeyField, CacheKeyer};
use serde_json::{Map, Value, json};

fn keyer(field: CacheKeyField, secret: &str) -> CacheKeyer {
    CacheKeyer::new(field, secret.as_bytes()).expect("a field was chosen")
}

#[test]
fn none_builds_no_keyer() {
    assert!(CacheKeyer::new(CacheKeyField::None, b"secret").is_none());
}

#[test]
fn the_key_is_stable_opaque_and_secret_dependent() {
    let a = keyer(CacheKeyField::PromptCacheKey, "secret-a");
    let request = json!({"user": "customer-end-user-42"});
    let key = a.key_for(&request).unwrap();
    // Stable: the same input groups the same way on every request.
    assert_eq!(a.key_for(&request).unwrap(), key);
    // Opaque: fixed-length hex that does not contain the input.
    assert_eq!(key.len(), 32);
    assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(!key.contains("customer"));
    // Different inputs and different secrets give different keys.
    assert_ne!(a.key_for(&json!({"user": "someone-else"})).unwrap(), key);
    let b = keyer(CacheKeyField::PromptCacheKey, "secret-b");
    assert_ne!(b.key_for(&request).unwrap(), key);
}

#[test]
fn prompt_cache_key_is_preferred_and_absent_input_sends_nothing() {
    let k = keyer(CacheKeyField::PromptCacheKey, "s");
    let both = json!({"prompt_cache_key": "conversation-1", "user": "u"});
    assert_eq!(
        k.key_for(&both),
        k.key_for(&json!({"prompt_cache_key": "conversation-1"}))
    );
    assert_eq!(k.key_for(&json!({"messages": []})), None);
    assert_eq!(k.key_for(&json!({"user": ""})), None);
}

#[test]
fn apply_writes_only_the_chosen_field() {
    let request = json!({"user": "customer-end-user-42"});
    for (field, name) in [
        (CacheKeyField::PromptCacheKey, "prompt_cache_key"),
        (CacheKeyField::User, "user"),
    ] {
        let k = keyer(field, "s");
        let mut body = Map::new();
        k.apply(&request, &mut body);
        assert_eq!(body.len(), 1);
        let sent = body.get(name).and_then(Value::as_str).unwrap();
        assert_ne!(sent, "customer-end-user-42", "the raw value is never sent");
    }
}

#[test]
fn field_names_deserialize() {
    for (name, field) in [
        ("none", CacheKeyField::None),
        ("prompt_cache_key", CacheKeyField::PromptCacheKey),
        ("user", CacheKeyField::User),
    ] {
        assert_eq!(serde_yaml::from_str::<CacheKeyField>(name).unwrap(), field);
    }
    assert!(serde_yaml::from_str::<CacheKeyField>("session").is_err());
}
