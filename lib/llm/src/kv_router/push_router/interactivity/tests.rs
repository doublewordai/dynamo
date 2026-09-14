// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn metric(rate: Option<f64>, running: u64, revision: u64) -> DecodeMetrics {
    DecodeMetrics {
        tokens_per_user_second: rate,
        num_running_reqs: running,
        num_waiting_reqs: 0,
        observation_revision: revision,
        observed_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    }
}

fn state() -> (State, Instant) {
    let config: PoolConfig = serde_json::from_value(serde_json::json!({
        "endpoint":"speed.worker.generate", "default_pool":"throughput",
        "pools":{"interactive":{"min_decode_tps_per_user":50,"minimum":1},
                 "throughput":{"min_decode_tps_per_user":10,"minimum":1}},
        "sustained_seconds": 1, "cooldown_seconds": 2
    }))
    .unwrap();
    config.validate().unwrap();
    let mut state = State::new(config);
    let now = Instant::now();
    state.is_ready = true;
    state.drain_ready = true;
    state.reconcile(
        &(1..=3)
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
            .collect(),
        now - Duration::from_secs(100),
    );
    state.members.get_mut(&1).unwrap().pool = "interactive".into();
    state.members.get_mut(&2).unwrap().pool = "throughput".into();
    state.members.get_mut(&3).unwrap().pool = "throughput".into();
    state.membership_changed = false;
    for id in 1..=3 {
        state.members.get_mut(&id).unwrap().decode_blocks = HashMap::from([(0, 0), (1, 0)]);
        observe(&mut state, id, Some(100.0), 1, now);
    }
    (state, now)
}

fn observe(state: &mut State, id: u64, rate: Option<f64>, running: u64, now: Instant) {
    for rank in 0..2 {
        let revision = state.members[&id]
            .reports
            .get(&rank)
            .map_or(1, |r| r.revision + 1);
        state.report(
            ActiveLoad {
                worker_id: id,
                dp_rank: rank,
                decode_metrics: Some(metric(rate, running, revision)),
                ..Default::default()
            },
            now,
        );
    }
}

fn manager() -> Arc<PoolManager> {
    Arc::new(PoolManager {
        state: Mutex::new(state().0),
    })
}

#[test]
fn speed_gates_every_rank_strictly_and_ignores_kv_budget() {
    let (mut state, now) = state();
    state
        .members
        .get_mut(&1)
        .unwrap()
        .decode_blocks
        .insert(0, 10_000);
    assert!(state.views(now)[0].eligible("interactive"));
    for speed in [50.0, 49.0, 0.0] {
        observe(&mut state, 1, Some(speed), 1, now);
        assert!(!state.views(now)[0].eligible("interactive"));
    }
    observe(&mut state, 1, Some(60.0), 1, now);
    state
        .members
        .get_mut(&1)
        .unwrap()
        .reports
        .get_mut(&1)
        .unwrap()
        .decode
        .tokens_per_user_second = Some(40.0);
    assert!(!state.views(now)[0].eligible("interactive"));
}

#[test]
fn borrowing_retains_stricter_target_until_cleanup() {
    let manager = manager();
    let borrower = manager
        .admit(WorkerWithDpRank::new(2, 0), "interactive")
        .unwrap();
    {
        let mut state = manager.state.lock();
        observe(&mut state, 2, Some(40.0), 1, Instant::now());
        let view = &state.views(Instant::now())[1];
        assert_eq!(view.effective_target, 50.0);
        assert!(!view.available_by_pool["throughput"]);
        assert!(!view.available_by_pool["interactive"]);
    }
    drop(borrower);
    assert!(
        manager
            .admit(WorkerWithDpRank::new(2, 0), "throughput")
            .is_some()
    );
}

#[test]
fn reclaim_is_bounded_across_dp_ranks_and_blocks_borrowing() {
    let manager = manager();
    let borrower = manager
        .admit(WorkerWithDpRank::new(1, 0), "throughput")
        .unwrap();
    observe(&mut manager.state.lock(), 1, Some(40.0), 1, Instant::now());
    let home = manager
        .admit(WorkerWithDpRank::new(1, 1), "interactive")
        .unwrap();
    assert!(home.priority > borrower.priority);
    for rank in 0..2 {
        assert!(
            manager
                .admit(WorkerWithDpRank::new(1, rank), "interactive")
                .is_none()
        );
        assert!(
            manager
                .admit(WorkerWithDpRank::new(1, rank), "throughput")
                .is_none()
        );
    }
    observe(&mut manager.state.lock(), 1, Some(100.0), 1, Instant::now());
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_some()
    );
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "throughput")
            .is_none()
    );
    drop(home);
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "throughput")
            .is_some()
    );
    drop(borrower);
}

#[test]
fn reclaim_cannot_bypass_missing_speed_stale_reports_or_draining() {
    let manager = manager();
    let _borrower = manager
        .admit(WorkerWithDpRank::new(1, 0), "throughput")
        .unwrap();
    let now = Instant::now();
    observe(&mut manager.state.lock(), 1, None, 1, now);
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_none()
    );
    observe(&mut manager.state.lock(), 1, Some(1.0), 1, now);
    assert!(!manager.state.lock().views(now + Duration::from_secs(6))[0].eligible("interactive"));
    manager.state.lock().members.get_mut(&1).unwrap().target = Some("throughput".into());
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_none()
    );
}

#[test]
fn idle_probe_waits_for_measurement_or_completion_then_a_new_idle_observation() {
    let manager = manager();
    observe(&mut manager.state.lock(), 1, None, 0, Instant::now());
    let mut probe = manager
        .admit(WorkerWithDpRank::new(1, 0), "interactive")
        .unwrap();
    probe.start_dispatch();
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 1), "interactive")
            .is_none()
    );
    observe(&mut manager.state.lock(), 1, None, 0, Instant::now());
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 1), "interactive")
            .is_none()
    );
    probe.complete();
    drop(probe);
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 1), "interactive")
            .is_none()
    );
    observe(&mut manager.state.lock(), 1, None, 0, Instant::now());
    let mut probe = manager
        .admit(WorkerWithDpRank::new(1, 1), "interactive")
        .unwrap();
    probe.start_dispatch();
    observe(&mut manager.state.lock(), 1, Some(100.0), 1, Instant::now());
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_some()
    );
    probe.complete();
}

#[test]
fn failed_selection_releases_probe_but_uncertain_dispatch_does_not() {
    let manager = manager();
    observe(&mut manager.state.lock(), 1, None, 0, Instant::now());
    drop(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .unwrap(),
    );
    let mut probe = manager
        .admit(WorkerWithDpRank::new(1, 0), "interactive")
        .unwrap();
    probe.start_dispatch();
    drop(probe);
    observe(&mut manager.state.lock(), 1, None, 0, Instant::now());
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_none()
    );
}

#[test]
fn kv_updates_and_heartbeat_replays_do_not_refresh_decode_freshness() {
    let (mut state, now) = state();
    let original = state.members[&1].reports[&0].decode.clone();
    let later = now + Duration::from_secs(6);
    state.report(
        ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            decode_metrics: Some(original),
            ..Default::default()
        },
        later,
    );
    state.report(
        ActiveLoad {
            worker_id: 1,
            dp_rank: 1,
            num_waiting_reqs: Some(0),
            load_report_revision: Some(999),
            ..Default::default()
        },
        later,
    );
    assert!(!state.views(later)[0].healthy);
    observe(&mut state, 1, None, 0, later);
    assert!(state.views(later)[0].healthy);
    let mut old = metric(Some(100.0), 1, 999);
    old.observed_at_unix_ms -= 60_000;
    state.report(
        ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            decode_metrics: Some(old),
            ..Default::default()
        },
        later,
    );
    assert_ne!(state.members[&1].reports[&0].revision, 999);
}

#[test]
fn pressure_moves_only_idle_donors_with_headroom_and_minimum() {
    let (mut state, now) = state();
    observe(&mut state, 1, Some(60.0), 1, now); // above target, below headroom boundary
    state.rebalance(now);
    state.rebalance(now + Duration::from_secs(1));
    assert!(state.members.values().all(|m| m.target.is_none())); // all donors busy
    observe(&mut state, 2, None, 0, now);
    state.rebalance(now + Duration::from_secs(1));
    assert_eq!(state.members[&2].target.as_deref(), Some("interactive"));
    state.drain_ready = false;
    state.rebalance(now + Duration::from_secs(1));
    assert_eq!(state.members[&2].pool, "throughput");
    state.drain_ready = true;
    state.rebalance(now + Duration::from_secs(1));
    assert_eq!(state.members[&2].pool, "interactive");
    observe(&mut state, 2, Some(60.0), 1, now);
    observe(&mut state, 3, None, 0, now);
    state.rebalance(now + Duration::from_secs(3));
    state.rebalance(now + Duration::from_secs(4));
    assert!(state.members.values().all(|m| m.target.is_none())); // donor minimum
}

#[test]
fn remaining_donor_speed_and_endpoint_independence() {
    let (mut state, now) = state();
    let (other, _) = self::state();
    observe(&mut state, 1, Some(30.0), 1, now);
    observe(&mut state, 2, None, 0, now);
    observe(&mut state, 3, Some(11.0), 1, now);
    state.rebalance(now);
    state.rebalance(now + Duration::from_secs(1));
    assert!(state.members.values().all(|m| m.target.is_none()));
    observe(&mut state, 3, Some(20.0), 1, now);
    state.rebalance(now + Duration::from_secs(1));
    assert!(state.members[&2].target.is_some());
    assert!(other.members[&2].target.is_none());
}

#[test]
fn single_worker_gating_priority_and_invalid_rank() {
    let manager = manager();
    manager.state.lock().members.retain(|id, _| *id == 1);
    for (pool, priority) in [("interactive", 100), ("throughput", 0)] {
        let lease = manager.admit(WorkerWithDpRank::new(1, 0), pool).unwrap();
        let mut request = PreprocessedRequest::builder()
            .model("test".into())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.routing_mut().priority = Some(i32::MAX);
        lease.apply_priority(&mut request);
        assert_eq!(
            serde_json::to_value(request).unwrap()["routing"]["priority"],
            priority
        );
    }
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 2), "interactive")
            .is_none()
    );
    observe(&mut manager.state.lock(), 1, Some(40.0), 1, Instant::now());
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_none()
    );
}

#[test]
fn configuration_requires_positive_explicit_speed_targets() {
    let (state, _) = state();
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut config = state.config.clone();
        config
            .pools
            .get_mut("interactive")
            .unwrap()
            .min_decode_tps_per_user = speed;
        assert!(config.validate().is_err());
    }
    let mut config = serde_json::to_value(&state.config).unwrap();
    config["pools"]["interactive"]
        .as_object_mut()
        .unwrap()
        .remove("min_decode_tps_per_user");
    config["pools"]["interactive"]["kv_fraction"] = 0.1.into();
    assert!(parse_configs(&serde_json::to_vec(&config).unwrap()).is_err());
    assert!(
        parse_configs(&serde_json::to_vec(&vec![&state.config, &state.config]).unwrap()).is_err()
    );
}

#[test]
fn cooldown_delays_speed_pressure_drain() {
    let (mut state, now) = state();
    observe(&mut state, 1, Some(30.0), 1, now);
    observe(&mut state, 2, None, 0, now);
    state.members.get_mut(&1).unwrap().changed = now;
    state.rebalance(now);
    state.rebalance(now + Duration::from_secs(1));
    assert!(state.members[&2].target.is_none());
    state.rebalance(now + Duration::from_secs(2));
    assert_eq!(state.members[&2].target.as_deref(), Some("interactive"));
}

#[test]
fn global_dp_attribution_and_kv_reports_do_not_replace_speed() {
    let (mut state, now) = state();
    let member = state.members.get_mut(&1).unwrap();
    member.rank_start = 4;
    member.reports.clear();
    state.report(
        ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            decode_metrics: Some(metric(Some(100.0), 1, 1)),
            ..Default::default()
        },
        now,
    );
    assert!(state.members[&1].reports.is_empty());
    for rank in [4, 5] {
        state.report(
            ActiveLoad {
                worker_id: 1,
                dp_rank: rank,
                decode_metrics: Some(metric(Some(100.0), 1, 1)),
                ..Default::default()
            },
            now,
        );
    }
    assert!(state.views(now)[0].eligible("interactive"));
    state.report(
        ActiveLoad {
            worker_id: 1,
            dp_rank: 4,
            kv_used_blocks: Some(91),
            load_report_revision: Some(99),
            ..Default::default()
        },
        now + Duration::from_secs(6),
    );
    let view = &state.views(now + Duration::from_secs(6))[0];
    assert!(!view.healthy);
    assert_eq!(view.kv_used_blocks, 91);
    assert_eq!(view.ranks[&4].revision, 1);
}

#[test]
fn pending_idle_probe_does_not_block_a_measured_active_rank() {
    let manager = manager();
    manager.state.lock().report(
        ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            decode_metrics: Some(metric(None, 0, 2)),
            ..Default::default()
        },
        Instant::now(),
    );
    let _probe = manager
        .admit(WorkerWithDpRank::new(1, 0), "interactive")
        .unwrap();
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 0), "interactive")
            .is_none()
    );
    assert!(
        manager
            .admit(WorkerWithDpRank::new(1, 1), "interactive")
            .is_some()
    );
}
