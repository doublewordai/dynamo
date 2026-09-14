// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_runtime::Runtime;

struct Fleet {
    frontends: Vec<(Coordinator, Arc<PoolManager>)>,
    runtimes: Vec<Runtime>,
    configs: HashMap<u64, ModelRuntimeConfig>,
    config: PoolConfig,
}

impl Fleet {
    async fn new(count: usize) -> anyhow::Result<Self> {
        let config: PoolConfig = serde_json::from_value(serde_json::json!({
            "endpoint": format!("test-{}.worker.generate", uuid::Uuid::new_v4()),
            "pools": {"interactive": {"min_decode_tps_per_user": 50.0, "minimum": 1}, "throughput": {"min_decode_tps_per_user": 10.0, "minimum": 1}},
            "default_pool": "throughput"
        }))?;
        let configs = (1..=4)
            .map(|id| {
                (
                    id,
                    ModelRuntimeConfig {
                        data_parallel_size: 2,
                        total_kv_blocks: Some(100),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let mut fleet = Self {
            frontends: Vec::new(),
            runtimes: Vec::new(),
            configs,
            config,
        };
        for _ in 0..count {
            fleet.join().await?;
        }
        assert!(fleet.frontends[0].0.is_leader().await?);
        for _ in 0..count + 4 {
            fleet.round().await?;
        }
        Ok(fleet)
    }

    async fn join(&mut self) -> anyhow::Result<()> {
        let runtime = Runtime::from_current()?;
        let client = etcd::Client::new(etcd::ClientOptions::default(), runtime.clone()).await?;
        let coordinator = Coordinator::new(client, &self.config.endpoint);
        let manager = Arc::new(PoolManager {
            state: Mutex::new(State::new(self.config.clone())),
        });
        self.frontends.push((coordinator, manager));
        self.runtimes.push(runtime);
        Ok(())
    }

    fn observe(manager: &PoolManager) {
        let mut state = manager.state.lock();
        for member in state.members.values_mut() {
            for rank in member.rank_start..member.rank_start + member.rank_count {
                member.decode_blocks.insert(rank, 0);
                let revision = member.reports.get(&rank).map_or(1, |r| r.revision + 1);
                member.reports.insert(
                    rank,
                    RankReport {
                        decode: super::super::tests::metric(None, 0, revision),
                        waiting: 0,
                        kv_used: Some(0),
                        revision,
                        received: Instant::now(),
                    },
                );
            }
        }
    }

    async fn step(&mut self, frontend: usize) -> anyhow::Result<()> {
        let (coordinator, manager) = &mut self.frontends[frontend];
        Self::observe(manager);
        coordinator.step(manager, &self.configs).await
    }

    async fn round(&mut self) -> anyhow::Result<()> {
        for (_, manager) in &self.frontends {
            Self::observe(manager);
        }
        futures::future::try_join_all(
            self.frontends
                .iter_mut()
                .map(|(coordinator, manager)| coordinator.step(manager, &self.configs)),
        )
        .await?;
        Ok(())
    }

    async fn document(&self) -> anyhow::Result<Document> {
        Ok(self.frontends[0].0.read(&self.config).await?.0)
    }

    async fn start_drain(&self) -> anyhow::Result<u64> {
        let (coordinator, manager) = &self.frontends[0];
        assert!(coordinator.is_leader().await?);
        let (mut document, revision) = coordinator.read(&self.config).await?;
        let worker = *document
            .assignments
            .iter()
            .find(|(_, a)| a.pool == "throughput")
            .unwrap()
            .0;
        let mut proposed = manager.state.lock().clone();
        let member = proposed.members.get_mut(&worker).unwrap();
        member.target = Some("interactive".into());
        member.changed = Instant::now();
        document.capture(&proposed, Instant::now())?;
        assert!(coordinator.write(&document, revision, true).await?);
        Ok(worker)
    }

    async fn close(self) -> anyhow::Result<()> {
        for (coordinator, manager) in &self.frontends {
            coordinator.leave(manager).await?;
        }
        let coordinator = &self.frontends[0].0;
        coordinator
            .client
            .kv_delete(coordinator.document_key.clone(), None)
            .await?;
        for runtime in &self.runtimes {
            runtime.shutdown();
        }
        tokio::task::spawn_blocking(move || drop(self)).await?;
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interactivity_three_frontends_wait_for_delayed_updates_and_dispatches()
-> anyhow::Result<()> {
    let mut fleet = Fleet::new(3).await?;
    let original = fleet.document().await?;
    let worker = *original
        .assignments
        .iter()
        .find(|(_, a)| a.pool == "throughput")
        .unwrap()
        .0;
    let key = WorkerWithDpRank::new(worker, 0);
    // Existing traffic supplies measured capacity; this case isolates the drain barrier.
    {
        let mut state = fleet.frontends[2].1.state.lock();
        for rank in 0..2 {
            state.report(
                ActiveLoad {
                    worker_id: worker,
                    dp_rank: rank,
                    decode_metrics: Some(super::super::tests::metric(Some(100.0), 1, 100)),
                    ..Default::default()
                },
                Instant::now(),
            );
        }
    }
    let mut held = fleet.frontends[2].1.admit(key, "throughput").unwrap();
    // Selected but not yet dispatched: even a worker reporting zero is not drained.
    assert_eq!(fleet.start_drain().await?, worker);
    for _ in 0..4 {
        fleet.step(0).await?;
        fleet.step(1).await?;
    }
    assert_eq!(
        fleet.document().await?.assignments[&worker].pool,
        "throughput"
    );
    assert!(fleet.frontends[0].1.admit(key, "throughput").is_none());
    assert!(fleet.frontends[1].1.admit(key, "interactive").is_none());
    // The delayed frontend still has the old assignment. Its acknowledgement is essential.
    assert!(fleet.frontends[2].1.admit(key, "throughput").is_some());
    fleet.step(2).await?;
    assert!(fleet.frontends[2].1.admit(key, "throughput").is_none());
    held.start_dispatch();
    for _ in 0..3 {
        fleet.round().await?;
    }
    assert_eq!(
        fleet.document().await?.assignments[&worker].pool,
        "throughput"
    );
    held.complete();
    drop(held);
    {
        let (coordinator, manager) = &mut fleet.frontends[2];
        coordinator.step(manager, &fleet.configs).await?;
        // Replaying an unchanged observation, or refreshing only one DP rank,
        // must not satisfy the backend observation barrier.
        for _ in 0..3 {
            coordinator.step(manager, &fleet.configs).await?;
        }
        assert!(
            !coordinator.read(&fleet.config).await?.0.frontends[coordinator.frontend()]
                .contains_key(&worker)
        );
        manager
            .state
            .lock()
            .members
            .get_mut(&worker)
            .unwrap()
            .reports
            .get_mut(&0)
            .unwrap()
            .revision += 1;
        coordinator.step(manager, &fleet.configs).await?;
        assert!(
            !coordinator.read(&fleet.config).await?.0.frontends[coordinator.frontend()]
                .contains_key(&worker)
        );
    }
    for _ in 0..8 {
        fleet.round().await?;
    }
    let document = fleet.document().await?;
    assert_eq!(document.assignments[&worker].pool, "interactive");
    for (_, manager) in &fleet.frontends {
        assert_eq!(manager.state.lock().members[&worker].pool, "interactive");
        assert!(manager.admit(key, "interactive").is_some());
    }
    fleet.close().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interactivity_failover_fences_old_leader_and_preserves_join_barrier() -> anyhow::Result<()>
{
    let mut fleet = Fleet::new(3).await?;
    let worker = fleet.start_drain().await?;
    for _ in 0..3 {
        fleet.step(0).await?;
        fleet.step(1).await?;
    }
    let old_epoch = fleet.document().await?.assignments[&worker].epoch;
    // Expire the leader's real lease; its durable registration must survive.
    let mut client =
        etcd_client::Client::connect(etcd::ClientOptions::default().etcd_url, None).await?;
    client
        .lease_revoke(fleet.frontends[0].0.client.lease_id() as i64)
        .await?;
    assert!(fleet.frontends[1].0.is_leader().await?);
    let (document, revision) = fleet.frontends[0].0.read(&fleet.config).await?;
    assert!(
        !fleet.frontends[0]
            .0
            .write(&document, revision, true)
            .await?
    );
    assert_eq!(document.assignments[&worker].epoch, old_epoch);
    assert!(
        document
            .frontends
            .contains_key(fleet.frontends[0].0.frontend())
    );
    // Joining after the drain started must be registered before it can serve.
    fleet.join().await?;
    assert!(
        fleet.frontends[3]
            .1
            .admit(WorkerWithDpRank::new(worker, 0), "throughput")
            .is_none()
    );
    fleet.step(3).await?;
    for _ in 0..3 {
        fleet.step(1).await?;
        fleet.step(2).await?;
    }
    assert_eq!(
        fleet.document().await?.assignments[&worker].pool,
        "throughput"
    );
    for _ in 0..8 {
        fleet.step(1).await?;
        fleet.step(2).await?;
        fleet.step(3).await?;
    }
    assert_eq!(
        fleet.document().await?.assignments[&worker].pool,
        "interactive"
    );
    fleet.close().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interactivity_uncertain_dispatch_aborts_move_without_forgetting_frontend()
-> anyhow::Result<()> {
    let mut fleet = Fleet::new(3).await?;
    let document = fleet.document().await?;
    let worker = *document
        .assignments
        .iter()
        .find(|(_, a)| a.pool == "throughput")
        .unwrap()
        .0;
    let mut request = fleet.frontends[2]
        .1
        .admit(WorkerWithDpRank::new(worker, 0), "throughput")
        .unwrap();
    request.start_dispatch();
    drop(request);
    fleet.start_drain().await?;
    for _ in 0..4 {
        fleet.round().await?;
    }
    let (mut document, revision) = fleet.frontends[0].0.read(&fleet.config).await?;
    assert!(!document.drain_ready(worker));
    let draining = document.assignments.get_mut(&worker).unwrap();
    let drain_epoch = draining.epoch;
    draining.changed_ms = wall_ms() - 301_000;
    assert!(
        fleet.frontends[0]
            .0
            .write(&document, revision, true)
            .await?
    );
    for _ in 0..3 {
        fleet.round().await?;
    }
    let document = fleet.document().await?;
    let assignment = &document.assignments[&worker];
    assert_eq!(assignment.pool, "throughput");
    assert_eq!(assignment.target, None);
    assert!(assignment.epoch > drain_epoch);
    fleet.frontends[2].0.leave(&fleet.frontends[2].1).await?;
    assert!(
        fleet
            .document()
            .await?
            .frontends
            .contains_key(fleet.frontends[2].0.frontend())
    );
    // A worker disappearing and returning with the same incarnation retains its pool.
    let config = fleet.configs.remove(&worker).unwrap();
    fleet.step(0).await?;
    fleet.configs.insert(worker, config);
    for _ in 0..3 {
        fleet.round().await?;
    }
    assert_eq!(
        fleet.document().await?.assignments[&worker].pool,
        "throughput"
    );
    fleet.close().await
}
