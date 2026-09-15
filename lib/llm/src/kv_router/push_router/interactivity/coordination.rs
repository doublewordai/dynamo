// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use dynamo_runtime::transports::etcd;
use etcd_client::{Compare, CompareOp, Txn, TxnOp};

use super::*;

#[cfg(all(test, feature = "testing-etcd"))]
mod tests;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Assignment {
    is_present: bool,
    stable_id: String,
    pool: String,
    target: Option<String>,
    epoch: u64,
    changed_ms: u64,
    rank_start: u32,
    rank_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Document {
    config: PoolConfig,
    epoch: u64,
    assignments: BTreeMap<u64, Assignment>,
    // Registrations and acknowledgements deliberately survive lease expiry. A
    // disconnected frontend may still reach workers. Only that frontend can leave.
    frontends: BTreeMap<String, BTreeMap<u64, u64>>,
}

impl Document {
    fn new(config: PoolConfig) -> Self {
        Self {
            config,
            epoch: 0,
            assignments: BTreeMap::new(),
            frontends: BTreeMap::new(),
        }
    }

    fn drain_ready(&self, worker: u64) -> bool {
        let Some(assignment) = self.assignments.get(&worker) else {
            return false;
        };
        assignment.target.is_some()
            && !self.frontends.is_empty()
            && self
                .frontends
                .values()
                .all(|acks| acks.get(&worker) == Some(&assignment.epoch))
    }

    fn apply(&self, state: &mut State, configs: &HashMap<u64, ModelRuntimeConfig>, now: Instant) {
        // Retain outstanding accounting if discovery temporarily loses a worker.
        // It must not turn a dropped discovery record into a drain acknowledgement.
        state.members.retain(|id, member| {
            self.assignments.contains_key(id)
                || !member.local.is_empty()
                || member.has_uncertain_requests
        });
        let wall_ms = wall_ms();
        for (&id, assignment) in &self.assignments {
            let Some(config) = configs.get(&id).filter(|c| {
                worker_identity(id, c) == assignment.stable_id
                    && c.data_parallel_start_rank == assignment.rank_start
                    && c.data_parallel_size.max(1) == assignment.rank_count
            }) else {
                if let Some(member) = state.members.get_mut(&id) {
                    member.is_present = false;
                    member.reports.clear();
                }
                continue;
            };
            let member = state.members.entry(id).or_insert_with(|| Member {
                is_present: assignment.is_present,
                stable_id: assignment.stable_id.clone(),
                pool: assignment.pool.clone(),
                target: assignment.target.clone(),
                changed: now,
                rank_start: assignment.rank_start,
                rank_count: assignment.rank_count,
                decode_blocks: HashMap::new(),
                kv_total: config.total_kv_blocks,
                reports: HashMap::new(),
                local: HashMap::new(),
                has_uncertain_requests: false,
                probe: None,
            });
            if member.stable_id != assignment.stable_id
                || member.rank_start != assignment.rank_start
                || member.rank_count != assignment.rank_count
            {
                member.stable_id.clone_from(&assignment.stable_id);
                member.rank_start = assignment.rank_start;
                member.rank_count = assignment.rank_count;
                member.reports.clear();
                member.decode_blocks.clear();
                member.probe = None;
            }
            member.pool.clone_from(&assignment.pool);
            member.is_present = assignment.is_present;
            member.target.clone_from(&assignment.target);
            member.changed = now
                .checked_sub(Duration::from_millis(
                    wall_ms.saturating_sub(assignment.changed_ms),
                ))
                .unwrap_or(now);
            member.kv_total = config.total_kv_blocks;
        }
        state.is_ready = true;
        state.drain_ready = state
            .members
            .iter()
            .find(|(_, m)| m.is_present && m.target.is_some())
            .is_some_and(|(id, _)| self.drain_ready(*id));
    }

    fn capture(&mut self, state: &State, now: Instant) -> anyhow::Result<()> {
        let wall_ms = wall_ms();
        let epoch = self.epoch + 1;
        // Keep assignments for absent workers. Rediscovery of the same incarnation
        // must preserve its pool; a discovery gap is not a drain barrier.
        for (&id, member) in &state.members {
            let previous = self.assignments.get(&id);
            if let Some(previous) = previous
                && previous.pool != member.pool
            {
                anyhow::ensure!(
                    previous.target.as_ref() == Some(&member.pool) && self.drain_ready(id),
                    "pool reclassification requires the matching frontend barrier"
                );
            }
            if previous.is_some_and(|a| {
                a.is_present
                    && a.stable_id == member.stable_id
                    && a.pool == member.pool
                    && a.target == member.target
                    && a.rank_start == member.rank_start
                    && a.rank_count == member.rank_count
            }) {
                continue;
            }
            self.assignments.insert(
                id,
                Assignment {
                    is_present: true,
                    stable_id: member.stable_id.clone(),
                    pool: member.pool.clone(),
                    target: member.target.clone(),
                    epoch,
                    changed_ms: wall_ms.saturating_sub(
                        now.saturating_duration_since(member.changed).as_millis() as u64,
                    ),
                    rank_start: member.rank_start,
                    rank_count: member.rank_count,
                },
            );
            self.epoch = epoch;
        }
        for (id, assignment) in &mut self.assignments {
            if assignment.is_present && !state.members.contains_key(id) {
                assignment.is_present = false;
                assignment.target = None;
                assignment.epoch = epoch;
                assignment.changed_ms = wall_ms;
                self.epoch = epoch;
            }
        }
        Ok(())
    }
}

fn wall_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(super) struct Coordinator {
    client: etcd::Client,
    frontend: String,
    document_key: String,
    leader_key: String,
    revision: i64,
    // A fresh worker observation is required after this frontend becomes idle.
    barriers: HashMap<(u64, u64), HashMap<u32, u64>>,
    was_leader: bool,
}

impl Coordinator {
    pub(super) fn new(client: etcd::Client, endpoint: &str) -> Self {
        let prefix = format!("/dynamo/interactivity/{endpoint}");
        Self {
            client,
            frontend: uuid::Uuid::new_v4().to_string(),
            document_key: format!("{prefix}/assignments"),
            leader_key: format!("{prefix}/leader"),
            revision: 0,
            barriers: HashMap::new(),
            was_leader: false,
        }
    }

    pub(super) fn frontend(&self) -> &str {
        &self.frontend
    }
    pub(super) fn revision(&self) -> i64 {
        self.revision
    }

    async fn read(&self, config: &PoolConfig) -> anyhow::Result<(Document, i64)> {
        let values = self.client.kv_get(self.document_key.clone(), None).await?;
        let Some(value) = values.first() else {
            return Ok((Document::new(config.clone()), 0));
        };
        let document: Document = serde_json::from_slice(value.value())?;
        anyhow::ensure!(
            &document.config == config,
            "frontends have different interactivity pool configurations"
        );
        Ok((document, value.mod_revision()))
    }

    async fn write(
        &self,
        document: &Document,
        revision: i64,
        as_leader: bool,
    ) -> anyhow::Result<bool> {
        let mut conditions = vec![Compare::mod_revision(
            self.document_key.clone(),
            CompareOp::Equal,
            revision,
        )];
        if as_leader {
            conditions.push(Compare::value(
                self.leader_key.clone(),
                CompareOp::Equal,
                self.frontend.clone(),
            ));
        }
        let txn = Txn::new().when(conditions).and_then(vec![TxnOp::put(
            self.document_key.clone(),
            serde_json::to_vec(document)?,
            None,
        )]);
        Ok(self.client.kv_txn(txn).await?.succeeded())
    }

    async fn is_leader(&self) -> anyhow::Result<bool> {
        anyhow::ensure!(
            self.client.lease_id() != 0,
            "pool leader election requires an etcd lease"
        );
        self.client
            .kv_create(&self.leader_key, self.frontend.as_bytes().to_vec(), None)
            .await?;
        Ok(self
            .client
            .kv_get(self.leader_key.clone(), None)
            .await?
            .first()
            .is_some_and(|kv| kv.value() == self.frontend.as_bytes()))
    }

    fn acknowledge(&mut self, document: &mut Document, state: &State, now: Instant) {
        let Some(acks) = document.frontends.get_mut(&self.frontend) else {
            return;
        };
        let draining: BTreeSet<_> = document
            .assignments
            .iter()
            .filter_map(|(&id, a)| a.target.as_ref().map(|_| (id, a.epoch)))
            .collect();
        acks.retain(|id, epoch| draining.contains(&(*id, *epoch)));
        self.barriers.retain(|key, _| draining.contains(key));
        for (id, epoch) in draining {
            if acks.get(&id) == Some(&epoch) {
                continue;
            }
            let Some(member) = state.members.get(&id) else {
                continue;
            };
            if !member.local.is_empty() || member.has_uncertain_requests {
                self.barriers.remove(&(id, epoch));
                continue;
            }
            let view = member.view(id, &state.config, now);
            if !view.healthy {
                continue;
            }
            let baseline = self.barriers.entry((id, epoch)).or_insert_with(|| {
                member
                    .reports
                    .iter()
                    .map(|(&rank, r)| (rank, r.revision))
                    .collect()
            });
            if view.idle
                && member.decode_blocks.len() == member.rank_count as usize
                && view.occupied == 0
                && view.waiting == 0
                && member.reports.iter().all(|(rank, r)| {
                    baseline
                        .get(rank)
                        .is_some_and(|revision| r.revision > *revision)
                })
            {
                acks.insert(id, epoch);
            }
        }
    }

    pub(super) async fn step(
        &mut self,
        manager: &PoolManager,
        configs: &HashMap<u64, ModelRuntimeConfig>,
    ) -> anyhow::Result<()> {
        let config = manager.state.lock().config.clone();
        let (mut document, revision) = self.read(&config).await?;
        self.revision = revision;
        if !document.frontends.contains_key(&self.frontend) {
            document
                .frontends
                .insert(self.frontend.clone(), BTreeMap::new());
            self.write(&document, revision, false).await?;
            return Ok(());
        }
        let previous = document.clone();
        let now = Instant::now();
        {
            let mut state = manager.state.lock();
            document.apply(&mut state, configs, now);
            self.acknowledge(&mut document, &state, now);
        }
        // Acknowledgements and registration race through the same CAS as assignments.
        // A frontend joining during a move must therefore participate in its barrier.
        if document != previous {
            self.write(&document, revision, false).await?;
            return Ok(());
        }
        let is_leader = self.is_leader().await?;
        if !is_leader {
            self.was_leader = false;
            return Ok(());
        }
        let mut proposed = manager.state.lock().clone();
        if !self.was_leader {
            proposed.pressure_since.clear();
        }
        self.was_leader = true;
        proposed.reconcile(configs, now);
        for (id, member) in &mut proposed.members {
            if let Some(assignment) = document.assignments.get(id) {
                member.pool.clone_from(&assignment.pool);
            }
        }
        let aborted = document.assignments.iter().any(|(id, assignment)| {
            assignment.target.is_some()
                && proposed
                    .members
                    .get(id)
                    .is_none_or(|member| member.target.is_none())
        });
        // Publish an abort before considering another move, so old acknowledgements
        // cannot be reused for a newly started drain on the same worker.
        if !aborted {
            proposed.rebalance(now);
        }
        document.capture(&proposed, now)?;
        if document != previous {
            if self.write(&document, revision, true).await? {
                tracing::info!(endpoint=%config.endpoint, epoch=document.epoch, "Pool assignments committed by leader");
            }
        } else {
            // Only local policy history changes here. Roles are applied exclusively
            // from committed records on the next pass, including on the leader.
            let mut state = manager.state.lock();
            state.pressure_since = proposed.pressure_since;
            state.membership_changed = proposed.membership_changed;
            state.shortage = proposed.shortage;
        }
        Ok(())
    }

    pub(super) async fn leave(&self, manager: &PoolManager) -> anyhow::Result<()> {
        let config = {
            let mut state = manager.state.lock();
            state.is_ready = false;
            if state
                .members
                .values()
                .any(|m| !m.local.is_empty() || m.has_uncertain_requests)
            {
                return Ok(());
            }
            state.config.clone()
        };
        loop {
            let (mut document, revision) = self.read(&config).await?;
            document.frontends.remove(&self.frontend);
            if self.write(&document, revision, false).await? {
                break;
            }
            tokio::task::yield_now().await;
        }
        // Compare ownership so a late shutdown cannot remove a successor's lease.
        self.client
            .kv_txn(
                Txn::new()
                    .when(vec![Compare::value(
                        self.leader_key.clone(),
                        CompareOp::Equal,
                        self.frontend.clone(),
                    )])
                    .and_then(vec![TxnOp::delete(self.leader_key.clone(), None)]),
            )
            .await?;
        Ok(())
    }
}
