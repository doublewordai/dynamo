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
//!
//! The role record is the only writer of a following worker's card taints:
//! [`update_model_taints`](crate::local_model::update_model_taints), which the
//! `update/model_taints` route calls, refuses such a worker, so a direct update
//! can neither promote a parked worker behind the record's back nor be
//! overwritten by the next role.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use anyhow::Context as _;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::storage::kv;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use serde::{Deserialize, Serialize};

use crate::local_model::set_model_taints;

/// Bucket of [`PoolMember`] records, keyed `<dynamo namespace>/<instance id in hex>`.
pub const MEMBERS_BUCKET: &str = "v1/pool_members";
/// Prefix of the per-worker buckets `<dynamo namespace>/<instance id in hex>`,
/// each holding one [`PoolRole`] record under [`ROLE_KEY`].
pub const ROLES_BUCKET: &str = "v1/pool_roles";
/// The key of a worker's [`PoolRole`] record in its bucket.
pub const ROLE_KEY: &str = "role";

const ROLE_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// The base model cards, by [`card_key`], whose taints a role follower owns.
static FOLLOWED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

fn card_key(endpoint: &Endpoint) -> String {
    let id = endpoint.id();
    format!(
        "{}/{}/{}/{:x}",
        id.namespace,
        id.component,
        id.name,
        endpoint.drt().connection_id()
    )
}

/// Whether this worker's base model card on `endpoint` takes its taints from
/// its pool role record, so nothing else may set them.
pub fn follows_role(endpoint: &Endpoint) -> bool {
    FOLLOWED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains(&card_key(endpoint))
}

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

    // From here the role record is the card's only taint writer: before the
    // member record lets a controller see the worker.
    FOLLOWED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(card_key(&endpoint));

    let member = serde_json::to_vec(&PoolMember {
        pod_namespace,
        pod_name,
    })?;
    let published: anyhow::Result<()> = async {
        store
            .get_or_create_bucket(MEMBERS_BUCKET, None)
            .await?
            .insert(
                &kv::Key::new(format!("{namespace}/{instance}")),
                member.into(),
                0,
            )
            .await?;
        Ok(())
    }
    .await;
    if published.is_err() {
        // No role loop runs, so taint updates must not stay refused.
        FOLLOWED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&card_key(&endpoint));
    }
    published.context("publish pool member record")?;

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
                None => {
                    let watch = match store
                        .clone()
                        .watch(&roles_bucket, None, cancel.clone())
                        .await
                    {
                        Ok(watch) => watch,
                        Err(error) => {
                            tracing::warn!(%error, "Failed to watch pool role; retrying");
                            tokio::time::sleep(ROLE_RETRY_INTERVAL).await;
                            continue;
                        }
                    };
                    // The new watch replays the records that exist as puts,
                    // then every later change, so a role that exists, or is
                    // written later, arrives through it in order. A role
                    // removed while no watch ran shows up as nothing at all,
                    // so read the record once the watch is established, as
                    // `Manager::watch` requires, and take only its absence
                    // from the read: the worker returns to its boot role. A
                    // put the new watch still has queued from its snapshot
                    // is then fenced by `role_to_apply`, so the removed role
                    // is not applied again behind the read.
                    match current_role(&store, &roles_bucket).await {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            if let Some(value) = restore_boot_role(&mut assigned, &boot_role) {
                                pending = apply_or_keep(&endpoint, &own_taints, value).await;
                            }
                        }
                        Err(error) => {
                            // Without the read a role removed while no watch
                            // ran would stay in force; re-watch and read again.
                            tracing::warn!(%error, "Failed to read pool role after re-watching; retrying");
                            drop(watch);
                            tokio::time::sleep(ROLE_RETRY_INTERVAL).await;
                            continue;
                        }
                    }
                    watch
                }
            };
            loop {
                let event = match &pending {
                    // A queued role supersedes the pending one, so it is read
                    // before the pending role is retried.
                    Some(_) => tokio::select! {
                        biased;
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
                let action = role_to_apply(event, &mut assigned, &boot_role, || {
                    current_role(&store, &roles_bucket)
                })
                .await;
                match action {
                    RoleAction::Apply(value) => {
                        pending = apply_or_keep(&endpoint, &own_taints, value).await;
                    }
                    // The change that superseded this event follows it, so
                    // an older written role still waiting for a retry is not
                    // worth applying either. A pending boot role is kept: only
                    // a delete restores it, and once it is pending the next
                    // delete finds no written role in force and does nothing.
                    RoleAction::Superseded => {
                        if assigned {
                            pending = None;
                        }
                    }
                    RoleAction::None => {}
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

/// The worker's role record as stored now, if it has one.
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

/// What a watch event asks of the worker.
#[derive(Debug, PartialEq, Eq)]
enum RoleAction {
    /// Apply this role record.
    Apply(Vec<u8>),
    /// The store has changed since this event; the change follows as its own
    /// event, so neither this one nor an older pending role applies.
    Superseded,
    /// Nothing to do.
    None,
}

/// The role a watch event asks the worker to apply, if any.
///
/// A put or a delete applies only while it still describes the stored record,
/// read through `read_current` once the event arrives. A watch re-established
/// after its predecessor ended replays the records of its snapshot as puts,
/// and the record may have changed or gone since; every such change follows
/// as its own event, so a superseded event is skipped rather than briefly
/// applied. A failed read acts on the event, as the watch alone would. A
/// resync is the store's full state at one revision and applies as it is.
async fn role_to_apply<F, Fut>(
    event: kv::WatchEvent,
    assigned: &mut bool,
    boot_role: &[u8],
    read_current: F,
) -> RoleAction
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Option<Vec<u8>>>>,
{
    match event {
        kv::WatchEvent::Put(entry) if is_role_key(&entry.key()) => {
            let value = entry.value().to_vec();
            match read_current().await {
                Ok(Some(current)) if current == value => {}
                Ok(_) => {
                    tracing::debug!("Skipping a pool role the store has since replaced or removed");
                    return RoleAction::Superseded;
                }
                Err(error) => {
                    tracing::warn!(%error, "Failed to confirm pool role; applying it as watched");
                }
            }
            *assigned = true;
            RoleAction::Apply(value)
        }
        kv::WatchEvent::Resync(entries) => {
            let role = entries
                .into_iter()
                .find(|(key, _)| is_role_key(key.as_ref()))
                .map(|(_, value)| value.to_vec());
            match role {
                Some(role) => {
                    *assigned = true;
                    RoleAction::Apply(role)
                }
                // No role record: back to the boot role if one was in force.
                None => restore_boot_role(assigned, boot_role)
                    .map_or(RoleAction::None, RoleAction::Apply),
            }
        }
        kv::WatchEvent::Delete(key) if is_role_key(key.as_ref()) => {
            match read_current().await {
                // Written again since: the put that follows applies it.
                Ok(Some(_)) => return RoleAction::Superseded,
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(%error, "Failed to confirm pool role removal; acting on it as watched");
                }
            }
            restore_boot_role(assigned, boot_role).map_or(RoleAction::None, RoleAction::Apply)
        }
        _ => RoleAction::None,
    }
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
    set_model_taints(endpoint, card_taints(own_taints, role)).await
}

/// The caller-managed taints of a card: the worker's own and the role's.
fn card_taints(own_taints: &[String], role: PoolRole) -> HashSet<String> {
    own_taints.iter().cloned().chain(role.taints).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn role_put(value: &[u8]) -> kv::WatchEvent {
        kv::WatchEvent::Put(kv::KeyValue::new(
            kv::Key::new(format!("v1/pool_roles/ns/1/{ROLE_KEY}")),
            bytes::Bytes::copy_from_slice(value),
        ))
    }

    fn role_delete() -> kv::WatchEvent {
        kv::WatchEvent::Delete(kv::Key::new(format!("v1/pool_roles/ns/1/{ROLE_KEY}")))
    }

    /// Replays a re-established watch: its snapshot still holds a role that
    /// was removed before the post-watch read, then the removal follows.
    #[tokio::test]
    async fn a_removed_role_queued_in_a_new_watch_is_not_applied_again() {
        let boot = br#"{"taints":["dynamo.pool/mirror-of=parked/0"]}"#.to_vec();
        let stale = br#"{"taints":[]}"#.to_vec();
        let newer = br#"{"taints":["dynamo.pool/mirror-of=ns/7"]}"#.to_vec();
        // The post-watch read found no record and restored the boot role.
        let mut assigned = false;

        // The snapshot's put of the removed role is skipped...
        let read_none = || async { Ok(None) };
        assert_eq!(
            role_to_apply(role_put(&stale), &mut assigned, &boot, read_none).await,
            RoleAction::Superseded
        );
        assert!(!assigned);
        // ...and the delete that follows changes nothing: the boot role is in force.
        let read_none = || async { Ok(None) };
        assert_eq!(
            role_to_apply(role_delete(), &mut assigned, &boot, read_none).await,
            RoleAction::None
        );

        // A role replaced before its put is read is skipped for the newer one.
        let read_newer = || async { Ok(Some(newer.clone())) };
        assert_eq!(
            role_to_apply(role_put(&stale), &mut assigned, &boot, read_newer).await,
            RoleAction::Superseded
        );
        let read_newer = || async { Ok(Some(newer.clone())) };
        assert_eq!(
            role_to_apply(role_put(&newer), &mut assigned, &boot, read_newer).await,
            RoleAction::Apply(newer.clone())
        );
        assert!(assigned);

        // Its removal restores the boot role once.
        let read_none = || async { Ok(None) };
        assert_eq!(
            role_to_apply(role_delete(), &mut assigned, &boot, read_none).await,
            RoleAction::Apply(boot.clone())
        );
    }

    /// A role deleted and written again before the delete is read: the boot
    /// role is not applied in between, and the put that follows applies.
    #[tokio::test]
    async fn a_delete_followed_by_a_rewrite_does_not_restore_the_boot_role() {
        let boot = br#"{"taints":["dynamo.pool/mirror-of=parked/0"]}"#.to_vec();
        let role = br#"{"taints":[]}"#.to_vec();
        let mut assigned = true;
        let read_role = || async { Ok(Some(role.clone())) };
        assert_eq!(
            role_to_apply(role_delete(), &mut assigned, &boot, read_role).await,
            RoleAction::Superseded
        );
        assert!(assigned);
        let read_role = || async { Ok(Some(role.clone())) };
        assert_eq!(
            role_to_apply(role_put(&role), &mut assigned, &boot, read_role).await,
            RoleAction::Apply(role.clone())
        );
    }

    #[tokio::test]
    async fn an_event_is_acted_on_when_the_confirming_read_fails() {
        let role = br#"{"taints":[]}"#.to_vec();
        let mut assigned = false;
        let failing = || async { Err(anyhow::anyhow!("store unavailable")) };
        assert_eq!(
            role_to_apply(role_put(&role), &mut assigned, b"boot", failing).await,
            RoleAction::Apply(role)
        );
        assert!(assigned);
        let failing = || async { Err(anyhow::anyhow!("store unavailable")) };
        assert_eq!(
            role_to_apply(role_delete(), &mut assigned, b"boot", failing).await,
            RoleAction::Apply(b"boot".to_vec())
        );
    }

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
