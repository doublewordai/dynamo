// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The discovery-store channel through which a rollout controller sets the pool
//! role of a worker that booted with one.
//!
//! A worker started with [`MirrorTarget::ENV`] publishes a [`PoolMember`] record
//! that ties its discovery instance to its Pod, bound to its discovery lease.
//! It then follows its own [`PoolRole`] record: its base model card carries the
//! taints the worker registered with plus the role's pool taints, set through
//! the same update the `update/model_taints` route makes. Both records live in
//! the discovery key-value store, so the controller needs no network path to
//! the worker itself, and needs to know only the pool taints it sets.
//!
//! Removing a worker's role record returns it to the role it booted with, never
//! to an ordinary serving worker: a parked or mirroring worker whose record is
//! deleted stays out of client traffic. Promotion is always an explicit role
//! with no pool taints.

use std::collections::HashSet;

use anyhow::Context as _;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::storage::kv;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use serde::{Deserialize, Serialize};

use crate::local_model::update_model_taints;

/// Bucket of [`PoolMember`] records, keyed `<dynamo namespace>/<instance id in hex>`.
pub const MEMBERS_BUCKET: &str = "v1/pool_members";
/// Prefix of the per-worker buckets `<dynamo namespace>/<instance id in hex>`,
/// each holding one [`PoolRole`] record under [`ROLE_KEY`].
pub const ROLES_BUCKET: &str = "v1/pool_roles";
/// The key of a worker's [`PoolRole`] record in its bucket.
pub const ROLE_KEY: &str = "role";

const ROLE_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// The Pod a worker instance runs in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolMember {
    pub pod_namespace: String,
    pub pod_name: String,
}

/// The pool taints a worker's base model card carries beside its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRole {
    pub taints: Vec<String>,
}

/// Publish this worker's [`PoolMember`] record and apply every [`PoolRole`]
/// written for it, beside `own_taints`, until the runtime shuts down. When the
/// role record is removed, the worker returns to `boot_role`, the pool taints
/// it registered with.
///
/// Call after the base model card is registered on `endpoint`. Returns without
/// doing anything when discovery has no key-value store or the Pod identity
/// (`POD_NAMESPACE`, `POD_NAME`) is not in the environment.
pub async fn follow(
    endpoint: Endpoint,
    own_taints: Vec<String>,
    boot_role: PoolRole,
) -> anyhow::Result<()> {
    let drt = endpoint.drt();
    let Some(store) = drt.discovery().kv_store() else {
        tracing::warn!(
            "Pool role channel needs a key-value discovery backend; roles will not be followed"
        );
        return Ok(());
    };
    let (Ok(pod_namespace), Ok(pod_name)) =
        (std::env::var("POD_NAMESPACE"), std::env::var("POD_NAME"))
    else {
        tracing::warn!("POD_NAMESPACE and POD_NAME are unset; pool roles will not be followed");
        return Ok(());
    };

    let namespace = endpoint.id().namespace;
    let instance = format!("{:x}", drt.connection_id());
    let cancel = drt.primary_token();

    // Follow roles before announcing membership, so a role written as soon as
    // the member record appears is in the watch's initial snapshot.
    let roles_bucket = format!("{ROLES_BUCKET}/{namespace}/{instance}");
    let first_watch = store
        .clone()
        .watch(&roles_bucket, None, cancel.clone())
        .await
        .with_context(|| format!("watch {roles_bucket}"))?;

    let member = serde_json::to_vec(&PoolMember {
        pod_namespace,
        pod_name,
    })?;
    store
        .get_or_create_bucket(MEMBERS_BUCKET, None)
        .await?
        .insert(
            &kv::Key::new(format!("{namespace}/{instance}")),
            member.into(),
            0,
        )
        .await
        .context("publish pool member record")?;

    let boot_role = serde_json::to_vec(&boot_role)?;
    tokio::spawn(async move {
        // The latest role that failed to apply; it is retried until it applies
        // or a newer role replaces it.
        let mut pending: Option<Vec<u8>> = None;
        // Whether a written role is in force, so a removal restores the boot
        // role once rather than on every role-less resync.
        let mut assigned = false;
        let mut watch = Some(first_watch);
        // A watch that ends before the runtime shuts down is re-established,
        // so the worker keeps following roles while its member record lives.
        // The new watch starts with a resync of the current record.
        while !cancel.is_cancelled() {
            let (_task, mut events) = match watch.take() {
                Some(watch) => watch,
                None => match store
                    .clone()
                    .watch(&roles_bucket, None, cancel.clone())
                    .await
                {
                    Ok(watch) => {
                        // A watch replays existing records as puts, so a role
                        // removed while it was down shows up as nothing at
                        // all. Read the record once to see a removal.
                        match current_role(&store, &roles_bucket).await {
                            Ok(Some(value)) => {
                                assigned = true;
                                pending = apply_or_keep(&endpoint, &own_taints, value).await;
                            }
                            Ok(None) => {
                                if let Some(value) = restore_boot_role(&mut assigned, &boot_role) {
                                    pending = apply_or_keep(&endpoint, &own_taints, value).await;
                                }
                            }
                            Err(error) => {
                                tracing::warn!(%error, "Failed to read pool role after re-watching");
                            }
                        }
                        watch
                    }
                    Err(error) => {
                        tracing::warn!(%error, "Failed to watch pool role; retrying");
                        tokio::time::sleep(ROLE_RETRY_INTERVAL).await;
                        continue;
                    }
                },
            };
            loop {
                let event = match &pending {
                    Some(_) => tokio::select! {
                        event = events.recv() => event,
                        () = tokio::time::sleep(ROLE_RETRY_INTERVAL) => {
                            if let Some(value) = pending.take() {
                                pending = apply_or_keep(&endpoint, &own_taints, value).await;
                            }
                            continue;
                        }
                    },
                    None => events.recv().await,
                };
                let Some(event) = event else { break };
                let value = match event {
                    kv::WatchEvent::Put(entry) if is_role_key(&entry.key()) => {
                        assigned = true;
                        Some(entry.value().to_vec())
                    }
                    kv::WatchEvent::Resync(entries) => {
                        let role = entries
                            .into_iter()
                            .find(|(key, _)| is_role_key(key.as_ref()))
                            .map(|(_, value)| value.to_vec());
                        match role {
                            Some(role) => {
                                assigned = true;
                                Some(role)
                            }
                            // No role record: back to the boot role if one was in force.
                            None => restore_boot_role(&mut assigned, &boot_role),
                        }
                    }
                    kv::WatchEvent::Delete(key) if is_role_key(key.as_ref()) => {
                        restore_boot_role(&mut assigned, &boot_role)
                    }
                    _ => None,
                };
                if let Some(value) = value {
                    pending = apply_or_keep(&endpoint, &own_taints, value).await;
                }
            }
            if !cancel.is_cancelled() {
                tracing::warn!("Pool role watch ended; re-establishing it");
                tokio::time::sleep(ROLE_RETRY_INTERVAL).await;
            }
        }
    });
    Ok(())
}

/// The worker's role record, if one exists.
async fn current_role(
    store: &std::sync::Arc<kv::Manager>,
    bucket: &str,
) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(bucket) = store.get_bucket(bucket).await? else {
        return Ok(None);
    };
    Ok(bucket
        .get(&kv::Key::new(ROLE_KEY.to_string()))
        .await?
        .map(|value| value.to_vec()))
}

/// The boot role to apply when a written role is removed, once per removal.
fn restore_boot_role(assigned: &mut bool, boot_role: &[u8]) -> Option<Vec<u8>> {
    std::mem::take(assigned).then(|| boot_role.to_vec())
}

/// Apply a role, returning it when it failed in a way a retry can fix, so
/// the caller retries it. A role that cannot be decoded or that sets a
/// runtime-derived taint never applies; it is logged once and the worker
/// keeps its current role until a new record replaces it.
async fn apply_or_keep(
    endpoint: &Endpoint,
    own_taints: &[String],
    value: Vec<u8>,
) -> Option<Vec<u8>> {
    let role = match decode(&value) {
        Ok(role) => role,
        Err(error) => {
            tracing::error!(%error, "Rejected pool role; keeping the current role");
            return None;
        }
    };
    match apply(endpoint, own_taints, role).await {
        Ok(()) => None,
        Err(error) => {
            tracing::warn!(%error, "Failed to apply pool role; retrying");
            Some(value)
        }
    }
}

fn is_role_key(key: &str) -> bool {
    key.rsplit('/').next() == Some(ROLE_KEY)
}

/// Decode a role record, rejecting one that sets a runtime-derived taint.
fn decode(value: &[u8]) -> anyhow::Result<PoolRole> {
    let role: PoolRole = serde_json::from_slice(value).context("decode pool role")?;
    if let Some(taint) = role
        .taints
        .iter()
        .find(|taint| taint.starts_with(crate::local_model::runtime_config::TOPOLOGY_TAINT_PREFIX))
    {
        anyhow::bail!("pool role sets the runtime-derived taint {taint:?}");
    }
    Ok(role)
}

async fn apply(endpoint: &Endpoint, own_taints: &[String], role: PoolRole) -> anyhow::Result<()> {
    tracing::info!(taints = ?role.taints, "Applying pool role");
    update_model_taints(endpoint, card_taints(own_taints, role)).await
}

/// The caller-managed taints of a card: the worker's own and the role's.
fn card_taints(own_taints: &[String], role: PoolRole) -> HashSet<String> {
    own_taints.iter().cloned().chain(role.taints).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_role_keeps_the_workers_own_taints() {
        let own = vec!["region=eu".to_string(), "country=fi".to_string()];
        let role = PoolRole {
            taints: vec!["dynamo.pool/mirror-of=ns/11".to_string()],
        };
        let expected: HashSet<String> = ["region=eu", "country=fi", "dynamo.pool/mirror-of=ns/11"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(card_taints(&own, role), expected);
        let promoted = PoolRole { taints: vec![] };
        assert_eq!(card_taints(&own, promoted).len(), 2);
    }

    #[test]
    fn removing_a_role_restores_the_boot_role_once() {
        let boot = serde_json::to_vec(&PoolRole {
            taints: vec!["dynamo.pool/mirror-of=dynamo-parked/0".to_string()],
        })
        .unwrap();
        let mut assigned = false;
        assert_eq!(restore_boot_role(&mut assigned, &boot), None);
        assigned = true;
        assert_eq!(restore_boot_role(&mut assigned, &boot), Some(boot.clone()));
        assert!(!assigned);
        assert_eq!(restore_boot_role(&mut assigned, &boot), None);
    }

    #[test]
    fn a_malformed_or_topology_role_is_rejected_not_retried() {
        assert!(decode(b"not json").is_err());
        assert!(decode(br#"{"taints":["dynamo.topology/zone=a"]}"#).is_err());
        assert_eq!(
            decode(br#"{"taints":["dynamo.pool/mirror-of=ns/1"]}"#).unwrap(),
            PoolRole {
                taints: vec!["dynamo.pool/mirror-of=ns/1".to_string()]
            }
        );
    }

    #[test]
    fn role_key_matches_only_the_final_segment() {
        assert!(is_role_key("v1/pool_roles/ns/1a2b/role"));
        assert!(is_role_key("role"));
        assert!(!is_role_key("v1/pool_roles/ns/1a2b/roles"));
        assert!(!is_role_key("v1/pool_roles/ns/role/x"));
    }
}
