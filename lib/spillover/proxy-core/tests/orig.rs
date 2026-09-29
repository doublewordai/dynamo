//! Tests for the `dw.orig.v1` request carrier.

use dw_proxy_core::orig::{self, OrigError};
use serde_json::{Value, json};

fn request() -> Value {
    json!({
        "model": "zai-org/GLM-5.3",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "nvext": {"extra_fields": ["something"]},
        "stream": true
    })
}

#[test]
fn select_keeps_only_carried_fields() {
    let selected = orig::select_carried_fields(&request());
    assert_eq!(
        selected,
        json!({
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7
        })
    );
}

#[test]
fn round_trips_through_encode_decode() {
    let selected = orig::select_carried_fields(&request());
    let entry = orig::encode(&selected);
    assert!(entry.starts_with(orig::PREFIX));
    let decoded = orig::decode(&entry).unwrap().unwrap();
    assert_eq!(decoded, selected);
}

#[test]
fn round_trips_unicode_and_nested_values() {
    let selected = json!({
        "messages": [{"role": "user", "content": "héllo 🌍 — 日本語"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"x": 1}}}],
        "stop": ["\n", "END"],
    });
    let decoded = orig::decode(&orig::encode(&selected)).unwrap().unwrap();
    assert_eq!(decoded, selected);
}

#[test]
fn decode_ignores_entries_without_prefix() {
    assert_eq!(orig::decode("nvext.other:xyz").unwrap(), None);
    assert_eq!(orig::decode("").unwrap(), None);
}

#[test]
fn decode_rejects_garbled_entries() {
    let bad_base64 = orig::decode("dw.orig.v1:not!base64!").unwrap_err();
    assert!(matches!(bad_base64, OrigError::Base64(_)));

    // `bm90IGpzb24` is `not json` in base64url-no-pad.
    let bad_json = orig::decode("dw.orig.v1:bm90IGpzb24").unwrap_err();
    assert!(matches!(bad_json, OrigError::Json(_)));

    // `MTIz` is `123`, valid JSON but not an object.
    let not_object = orig::decode("dw.orig.v1:MTIz").unwrap_err();
    assert!(matches!(not_object, OrigError::NotAnObject));
}

#[test]
fn from_extra_args_takes_the_first_orig_entry() {
    let first = json!({"messages": [{"role": "user", "content": "first"}]});
    let second = json!({"messages": [{"role": "user", "content": "second"}]});
    let extra_args = json!({
        "nvext": {"extra_fields": [
            "unrelated:value",
            orig::encode(&first),
            orig::encode(&second)
        ]}
    });
    assert_eq!(
        orig::from_extra_args(Some(&extra_args)).unwrap(),
        Some(first)
    );
}

#[test]
fn from_extra_args_skips_non_string_entries() {
    let orig_entry = json!({"messages": [{"role": "user", "content": "x"}]});
    let extra_args = json!({
        "nvext": {"extra_fields": [7, null, orig::encode(&orig_entry)]}
    });
    assert_eq!(
        orig::from_extra_args(Some(&extra_args)).unwrap(),
        Some(orig_entry)
    );
}

#[test]
fn from_extra_args_falls_back_to_messages() {
    let extra_args = json!({
        "nvext": {"extra_fields": ["unrelated:value"]},
        "messages": [{"role": "user", "content": "from messages"}]
    });
    assert_eq!(
        orig::from_extra_args(Some(&extra_args)).unwrap(),
        Some(json!({"messages": [{"role": "user", "content": "from messages"}]}))
    );
}

#[test]
fn from_extra_args_returns_none_when_absent() {
    assert_eq!(orig::from_extra_args(None).unwrap(), None);
    assert_eq!(
        orig::from_extra_args(Some(&json!({"other": true}))).unwrap(),
        None
    );
}

#[test]
fn from_extra_args_propagates_garbled_orig() {
    let extra_args = json!({
        "nvext": {"extra_fields": ["dw.orig.v1:not!base64!"]}
    });
    assert!(matches!(
        orig::from_extra_args(Some(&extra_args)).unwrap_err(),
        OrigError::Base64(_)
    ));
}
