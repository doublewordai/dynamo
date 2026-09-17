// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::RouterConfigOverride;
use super::filter::RoutingEligibility;
use super::overlap::{OverlapSignals, SelectedWorkerTierSnapshot};
use super::prefill_load::effective_prefill_tokens;
pub use crate::protocols::PotentialLoad;
use crate::protocols::{
    LocalBlockHash, RoutingConstraints, SharedCacheHits, WorkerConfigLike, WorkerId,
    WorkerWithDpRank,
};
use crate::scheduling::policy_queue::QueueRejection;
use crate::scheduling::queue_admission::RequestProgressUpdater;
use crate::sequences::WorkerLoadProjection;

/// Router-side view of which workers should not receive new requests.
///
/// Overload is soft: selection prefers other workers and falls back to an
/// overloaded one when nothing else is eligible. Inhibition is hard: a worker
/// the request plane has reported down is never selected, so a request whose
/// stream just failed on it is re-dispatched elsewhere.
pub trait WorkerAvailability: Send + Sync + 'static {
    fn overloaded_worker_ids(&self) -> Option<HashSet<WorkerId>>;

    fn inhibited_worker_ids(&self) -> Option<HashSet<WorkerId>> {
        None
    }

    /// The latest load each worker rank reported about itself, keyed by rank.
    /// `None` when the provider has no worker reports. Providers hand out a
    /// shared snapshot rebuilt only when a report changes.
    fn reported_loads(&self) -> Option<Arc<FxHashMap<WorkerWithDpRank, ReportedRankLoad>>> {
        None
    }
}

/// What a worker rank last reported about its own load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReportedRankLoad {
    /// Requests waiting in the rank's engine scheduler queue.
    pub waiting_requests: u64,
    /// KV blocks in use on the rank, when reported.
    pub kv_used_blocks: Option<u64>,
    /// Total KV blocks on the rank, when known.
    pub kv_total_blocks: Option<u64>,
    /// Revision of the load report this snapshot came from, when the worker
    /// versions its reports. A change means the rank has observed everything
    /// dispatched before the previous report.
    pub report_revision: Option<u64>,
}

impl ReportedRankLoad {
    /// Identity of the report, so a consumer can tell a fresh report from a
    /// repeat of the last one. Versioned reports use their revision; legacy
    /// reports fall back to the reported values themselves.
    pub fn identity(&self) -> ReportIdentity {
        match self.report_revision {
            Some(revision) => ReportIdentity::Versioned(revision),
            None => ReportIdentity::Legacy {
                kv_used_blocks: self.kv_used_blocks,
                waiting_requests: self.waiting_requests,
            },
        }
    }
}

/// Identity of a worker load report, see [`ReportedRankLoad::identity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportIdentity {
    Versioned(u64),
    Legacy {
        kv_used_blocks: Option<u64>,
        waiting_requests: u64,
    },
}

impl<F> WorkerAvailability for F
where
    F: Fn() -> Option<HashSet<WorkerId>> + Send + Sync + 'static,
{
    fn overloaded_worker_ids(&self) -> Option<HashSet<WorkerId>> {
        (self)()
    }
}

pub type WorkerAvailabilityProvider = Arc<dyn WorkerAvailability>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierOverlapBlocks {
    #[serde(default)]
    pub device: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub host_pinned: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub disk: FxHashMap<WorkerWithDpRank, usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error(transparent)]
    QueueRejected(#[from] QueueRejection),

    #[error("all eligible workers are overloaded")]
    AllEligibleWorkersOverloaded,

    #[error("pinned worker {worker_id} is overloaded")]
    PinnedWorkerOverloaded { worker_id: WorkerId },

    #[error("pinned worker {worker_id} is not in allowed worker set")]
    PinnedWorkerNotAllowed { worker_id: WorkerId },

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to book scheduler state: {0}")]
    BookingFailed(String),

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),
}

impl KvSchedulerError {
    pub fn is_overload(&self) -> bool {
        matches!(
            self,
            Self::AllEligibleWorkersOverloaded | Self::PinnedWorkerOverloaded { .. }
        )
    }
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub effective_overlap_blocks: f64,
    pub cached_tokens: usize,
    pub selected_worker_tiers: SelectedWorkerTierSnapshot,
    pub request_progress: Option<RequestProgressUpdater>,
    pub lifecycle_lease: Option<super::queue::RequestLifecycleLease>,
    pub potential_decode_blocks: usize,
}

#[derive(Debug, Clone)]
pub enum ScheduleMode {
    QueryOnly {
        request_id: Option<String>,
    },
    /// Tracks worker state; the caller releases it with `free`.
    Tracked {
        request_id: String,
    },
    /// Tracks worker and request lifecycle state; the caller reports dispatch and a terminal outcome.
    TrackedWithLifecycle {
        request_id: String,
    },
}

impl ScheduleMode {
    pub fn from_legacy(
        request_id: Option<String>,
        update_states: bool,
    ) -> Result<Self, KvSchedulerError> {
        if !update_states {
            return Ok(Self::QueryOnly { request_id });
        }

        let Some(request_id) = request_id else {
            return Err(KvSchedulerError::BookingFailed(
                "tracked scheduling request requires a request_id".to_string(),
            ));
        };
        Ok(Self::Tracked { request_id })
    }

    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::QueryOnly { request_id } => request_id.as_deref(),
            Self::Tracked { request_id } | Self::TrackedWithLifecycle { request_id } => {
                Some(request_id)
            }
        }
    }

    pub fn is_tracked(&self) -> bool {
        matches!(
            self,
            Self::Tracked { .. } | Self::TrackedWithLifecycle { .. }
        )
    }

    pub(crate) fn lifecycle_request_id(&self) -> Option<&str> {
        match self {
            Self::TrackedWithLifecycle { request_id } => Some(request_id),
            Self::QueryOnly { .. } | Self::Tracked { .. } => None,
        }
    }

    pub fn tracked_request_id(&self) -> Option<&str> {
        match self {
            Self::QueryOnly { .. } => None,
            Self::Tracked { request_id } | Self::TrackedWithLifecycle { request_id } => {
                Some(request_id)
            }
        }
    }
}

/// Validated request accepted by [`LocalScheduler`](super::LocalScheduler).
pub struct ScheduleRequest {
    pub mode: ScheduleMode,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub block_hashes: Option<Vec<LocalBlockHash>>,
    pub isl_tokens: usize,
    pub lora_name: Option<String>,
    pub expected_output_tokens: Option<u32>,
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub excluded_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_id: Option<String>,
    pub overlap: OverlapSignals,
    pub shared_cache_hits: Option<SharedCacheHits>,
}

/// Actor-owned admission request.
///
/// After enqueue, the caller retains only the response receiver while the
/// scheduler owns this request and its sender. Dropping the caller's selection
/// future closes that receiver, but cannot retract the request from the actor.
pub struct SchedulingRequest {
    // Request identity and payload.
    pub mode: ScheduleMode,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub lora_name: Option<String>,
    pub expected_output_tokens: Option<u32>,

    // Routing constraints and request-level config.
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub excluded_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub track_prefill_tokens: bool,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_id: Option<String>,

    // Overlap and cache signals.
    pub overlap: OverlapSignals,
    pub shared_cache_hits: Option<SharedCacheHits>,

    // Load state computed during admission.
    pub worker_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,

    /// Sender half of the admission ownership handoff. For tracked requests,
    /// the actor must book before sending and undo the booking if delivery fails.
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

#[derive(Clone, Copy)]
pub struct SchedulingContext<'a, C> {
    request: &'a SchedulingRequest,
    eligibility: RoutingEligibility<'a>,
    workers: &'a HashMap<WorkerId, C>,
}

impl<'a, C: WorkerConfigLike> SchedulingContext<'a, C> {
    pub fn new(request: &'a SchedulingRequest, workers: &'a HashMap<WorkerId, C>) -> Self {
        Self {
            request,
            eligibility: request.eligibility(),
            workers,
        }
    }

    pub fn request(&self) -> &'a SchedulingRequest {
        self.request
    }

    pub fn best_effective_prefill_tokens(&self) -> usize {
        effective_prefill_tokens(self.request.isl_tokens, self.best_cached_tokens())
    }

    pub fn best_cached_tokens(&self) -> usize {
        match self.eligibility.pinned_worker() {
            Some(worker) => self.request.effective_cached_tokens_for(worker),
            None => self
                .request
                .overlap
                .effective_cached_tokens
                .iter()
                .filter(|(worker, _)| {
                    self.workers.get(&worker.worker_id).is_some_and(|config| {
                        self.eligibility.allows_worker(worker.worker_id, config)
                    })
                })
                .map(|(_, cached_tokens)| *cached_tokens)
                .max()
                .unwrap_or(0),
        }
    }
}

impl SchedulingRequest {
    #[inline]
    pub fn eligibility(&self) -> RoutingEligibility<'_> {
        self.eligibility_with_overloaded(None)
    }

    #[inline]
    pub fn eligibility_with_overloaded<'a>(
        &'a self,
        overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    ) -> RoutingEligibility<'a> {
        self.eligibility_with_availability(overloaded_worker_ids, None)
    }

    #[inline]
    pub fn eligibility_with_availability<'a>(
        &'a self,
        overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
        inhibited_worker_ids: Option<&'a HashSet<WorkerId>>,
    ) -> RoutingEligibility<'a> {
        RoutingEligibility::new(
            self.allowed_worker_ids.as_ref(),
            overloaded_worker_ids,
            self.pinned_worker,
            &self.routing_constraints,
        )
        .with_excluded_worker_ids(self.excluded_worker_ids.as_ref())
        .with_inhibited_worker_ids(inhibited_worker_ids)
    }

    pub(crate) fn effective_cached_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        self.overlap
            .effective_cached_tokens
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn effective_overlap_blocks_for(&self, worker: WorkerWithDpRank) -> f64 {
        self.overlap
            .effective_overlap_blocks
            .get(&worker)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn worker_load_for(&self, worker: WorkerWithDpRank) -> WorkerLoadProjection {
        self.worker_loads.get(&worker).copied().unwrap_or_default()
    }

    /// Prompt blocks resident on `worker`'s device, credited the way the
    /// selector's logit does: tier data wins when any tier map is present, and
    /// the untiered effective overlap is only a fallback for callers that never
    /// populate tier maps.
    pub(crate) fn device_overlap_blocks_for(&self, worker: WorkerWithDpRank) -> f64 {
        let tiers = &self.overlap.tier_overlap_blocks;
        let has_tier_overlap_blocks =
            !tiers.device.is_empty() || !tiers.host_pinned.is_empty() || !tiers.disk.is_empty();
        match tiers.device.get(&worker) {
            Some(blocks) => *blocks as f64,
            None if has_tier_overlap_blocks => 0.0,
            None => self.effective_overlap_blocks_for(worker),
        }
    }

    /// Prompt blocks `worker` would have to allocate for this request beyond
    /// what its device already holds.
    pub(crate) fn uncached_request_blocks_for(
        &self,
        worker: WorkerWithDpRank,
        block_size: u32,
    ) -> u64 {
        self.request_blocks(block_size)
            .saturating_sub(self.device_overlap_blocks_for(worker).floor() as u64)
    }

    pub(crate) fn request_blocks(&self, block_size: u32) -> u64 {
        self.isl_tokens.div_ceil(block_size as usize) as u64
    }

    pub(crate) fn potential_decode_blocks_after_admission(
        &self,
        worker: WorkerWithDpRank,
        block_size: u32,
    ) -> usize {
        self.worker_load_for(worker)
            .potential_decode_blocks()
            .saturating_add(self.request_blocks(block_size) as usize)
    }

    pub(crate) fn response_is_closed(&self) -> bool {
        self.resp_tx.as_ref().is_none_or(|tx| tx.is_closed())
    }

    pub fn respond(&mut self, result: Result<SchedulingResponse, KvSchedulerError>) -> bool {
        let Some(tx) = self.resp_tx.take() else {
            tracing::error!("respond called multiple times on same request");
            return false;
        };
        if tx.send(result).is_err() {
            tracing::debug!("requestor dropped scheduling response");
            return false;
        }
        true
    }
}
