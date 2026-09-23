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

/// Taints the runtime derives itself; neither a worker's own taints nor a role
/// sets them.
pub const TOPOLOGY_TAINT_PREFIX: &str = "dynamo.topology/";

/// Publish this worker's [`PoolMember`] record and apply every [`PoolRole`]
/// written for it, beside `own_taints`, until the runtime shuts down.
///
/// Call after the base model card is registered on `endpoint`. Returns without
/// doing anything when discovery has no key-value store or the Pod identity
/// (`POD_NAMESPACE`, `POD_NAME`) is not in the environment.
pub async fn follow(endpoint: Endpoint, own_taints: Vec<String>) -> anyhow::Result<()> {
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
    let (_task, mut events) = store
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

    tokio::spawn(async move {
        // The latest role that failed to apply; it is retried until it applies
        // or a newer role replaces it.
        let mut pending: Option<Vec<u8>> = None;
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
                    Some(entry.value().to_vec())
                }
                kv::WatchEvent::Resync(entries) => {
                    let role = entries
                        .into_iter()
                        .find(|(key, _)| is_role_key(key.as_ref()))
                        .map(|(_, value)| value.to_vec());
                    // No role record: nothing is pending any more.
                    if role.is_none() {
                        pending = None;
                    }
                    role
                }
                kv::WatchEvent::Delete(key) if is_role_key(key.as_ref()) => {
                    pending = None;
                    None
                }
                _ => None,
            };
            if let Some(value) = value {
                pending = apply_or_keep(&endpoint, &own_taints, value).await;
            }
        }
    });
    Ok(())
}

/// Apply a role, returning it when it failed so the caller retries it.
async fn apply_or_keep(
    endpoint: &Endpoint,
    own_taints: &[String],
    value: Vec<u8>,
) -> Option<Vec<u8>> {
    match apply(endpoint, own_taints, &value).await {
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

async fn apply(endpoint: &Endpoint, own_taints: &[String], value: &[u8]) -> anyhow::Result<()> {
    let role: PoolRole = serde_json::from_slice(value).context("decode pool role")?;
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
    fn role_key_matches_only_the_final_segment() {
        assert!(is_role_key("v1/pool_roles/ns/1a2b/role"));
        assert!(is_role_key("role"));
        assert!(!is_role_key("v1/pool_roles/ns/1a2b/roles"));
        assert!(!is_role_key("v1/pool_roles/ns/role/x"));
    }
}
