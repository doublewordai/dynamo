// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-queue margin: an alternative admission policy for the backend
//! admission gate, enabled by `DYN_ADMISSION_QUEUE_MARGIN`.
//!
//! The default gate bounds how many requests are *in* the engine and queues
//! the rest in Dynamo. This policy instead bounds the engine's own **waiting
//! queue** and never holds a request back in Dynamo: an admitted request goes
//! straight to the engine, whose scheduler decides what runs and keeps its own
//! priority scheduling, including preempting running requests for
//! higher-priority ones.
//!
//! The queue estimate is the engine's last reported waiting count, summed over
//! its data-parallel ranks, plus every request admitted since that report that
//! has not yet left the engine queue. A counted admission is given back as
//! soon as the request is known to have left the queue: at its first response
//! item (the engine is running it), when its stream ends, or when dispatch
//! failed before a stream existed. A complete report (every rank fresh since
//! the previous complete one) also resets the count; a partial one updates
//! the depth and keeps the count, since the stale ranks have not yet seen
//! those admissions. Between those, the estimate errs high. A process whose
//! engine has never reported is unenforced.
//!
//! A request is admitted iff the estimate is below the margin, or an admitted
//! request of strictly lower priority — anywhere in the process's in-flight
//! set, running or waiting — can be evicted to make room. The victim is the
//! lowest priority, tie-break most recently admitted, which naturally picks
//! engine-queued, zero-progress work. The victim's response stream ends with a
//! [`ErrorType::ResourceExhausted`] overload error and its request is then
//! killed so the engine aborts it. With no victim the arrival
//! is refused as [`ErrorType::WorkerOverloaded`], so the router can place it on
//! another worker.
//!
//! Only requests routed by a frontend carry an admission priority; requests
//! without one (control and management calls) stay under the gate's default
//! engine request limit and queue.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use futures::Stream;
use parking_lot::Mutex;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

use crate::engine::{AsyncEngineContextProvider, AsyncEngineStream, Data, EngineStream};
use crate::error::{DynamoError, ErrorType};
use crate::metrics::backend_admission::BackendAdmissionMetrics;

/// Per-worker engine waiting-queue length, in requests, beyond which an
/// arrival must evict a lower-priority request or is refused.
pub(crate) const DYN_ADMISSION_QUEUE_MARGIN: &str = "DYN_ADMISSION_QUEUE_MARGIN";

/// The message an arrival refused at the margin carries.
const MARGIN_REJECTED_MESSAGE: &str = "Server overloaded: engine queue at the admission margin";

/// The message an evicted request's stream ends with.
const EVICTED_MESSAGE: &str = "Server overloaded: request evicted for a higher-priority request";

/// Parse the margin. Unset means the policy is off; an unparseable value is
/// warned about and also leaves it off.
pub(crate) fn margin_from_raw(raw: Option<&str>) -> Option<u64> {
    let raw = raw?;
    match raw.trim().parse::<u64>() {
        Ok(margin) => Some(margin),
        Err(err) => {
            tracing::warn!(
                env = DYN_ADMISSION_QUEUE_MARGIN,
                value = %raw,
                %err,
                "Ignoring invalid admission queue margin; expected a non-negative integer"
            );
            None
        }
    }
}

/// The engine-queue estimate the margin is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueueEstimate {
    reported: u64,
    admitted_since_report: u64,
    /// Admitted requests still being dispatched to the engine. A report
    /// cannot include them yet, so reports never reset this count.
    dispatching: u64,
    /// Bumped by every complete report, so a refund can tell whether the
    /// admission it undoes is still counted in the current interval.
    generation: u64,
}

impl QueueEstimate {
    fn total(self) -> u64 {
        self.reported
            .saturating_add(self.admitted_since_report)
            .saturating_add(self.dispatching)
    }
}

/// One admitted request.
struct Entry {
    priority: i32,
    /// Fired when this request is evicted; its stream kills the request and
    /// ends with the overload error.
    evict: CancellationToken,
}

#[derive(Default)]
struct MarginState {
    /// Latest waiting count per data-parallel rank.
    waiting_by_rank: HashMap<u32, u64>,
    /// Ranks that reported since the last complete report.
    fresh_ranks: HashSet<u32>,
    /// Absent until the engine first reports: unenforced.
    estimate: Option<QueueEstimate>,
    /// Admitted requests, keyed by a monotone admission sequence that also
    /// orders them within a priority for victim tie-breaking.
    entries: HashMap<u64, Entry>,
    next_seq: u64,
}

impl MarginState {
    fn waiting_total(&self) -> u64 {
        self.waiting_by_rank.values().copied().sum()
    }

    /// A complete observation supersedes the admissions counted since the
    /// previous one.
    fn report_complete(&mut self) {
        let waiting = self.waiting_total();
        let (generation, dispatching) = self.estimate.map_or((1, 0), |estimate| {
            (estimate.generation.wrapping_add(1), estimate.dispatching)
        });
        self.estimate = Some(QueueEstimate {
            reported: waiting,
            admitted_since_report: 0,
            dispatching,
            generation,
        });
    }

    /// A partial observation refreshes the depth but keeps the counted
    /// admissions; the first observation counts as complete.
    fn report_partial(&mut self) {
        let waiting = self.waiting_total();
        match self.estimate.as_mut() {
            Some(estimate) => estimate.reported = waiting,
            None => self.report_complete(),
        }
    }

    /// Lowest-priority entry strictly below `priority`, most recently admitted
    /// within that priority.
    fn victim_below(&self, priority: i32) -> Option<u64> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.priority < priority)
            .min_by(|(a_seq, a), (b_seq, b)| a.priority.cmp(&b.priority).then(b_seq.cmp(a_seq)))
            .map(|(seq, _)| *seq)
    }
}

/// The margin policy. One per process, owned by the backend admission gate.
pub(crate) struct MarginGate {
    margin: u64,
    state: Mutex<MarginState>,
    metrics: Arc<BackendAdmissionMetrics>,
}

impl MarginGate {
    pub(crate) fn new(margin: u64, metrics: Arc<BackendAdmissionMetrics>) -> Arc<Self> {
        Arc::new(Self {
            margin,
            state: Mutex::new(MarginState::default()),
            metrics,
        })
    }

    /// Record one data-parallel rank's engine waiting count. Every call is a
    /// fresh observation from the engine.
    pub(crate) fn record_waiting(&self, dp_rank: u32, waiting: u64) {
        let mut state = self.state.lock();
        state.waiting_by_rank.insert(dp_rank, waiting);
        state.fresh_ranks.insert(dp_rank);
        let complete = state
            .waiting_by_rank
            .keys()
            .all(|rank| state.fresh_ranks.contains(rank));
        if complete {
            state.fresh_ranks.clear();
            state.report_complete();
        } else {
            state.report_partial();
        }
    }

    /// Admit one request at `priority`, evicting a lower-priority admitted
    /// request if the queue estimate is at the margin, or refuse it.
    pub(crate) fn admit(self: &Arc<Self>, priority: i32) -> Result<MarginCharge, DynamoError> {
        let (charge, evicted) = {
            let mut state = self.state.lock();
            let mut evicted = None;
            if let Some(estimate) = state.estimate
                && estimate.total() >= self.margin
            {
                let Some(victim_seq) = state.victim_below(priority) else {
                    drop(state);
                    self.metrics.received_direct();
                    self.metrics.rejected_queue_full();
                    tracing::warn!(
                        queued = estimate.total(),
                        margin = self.margin,
                        priority,
                        "Engine queue at the admission margin with no lower-priority request, \
                         rejecting request"
                    );
                    return Err(DynamoError::builder()
                        .error_type(ErrorType::WorkerOverloaded)
                        .message(MARGIN_REJECTED_MESSAGE)
                        .build());
                };
                let victim = state
                    .entries
                    .remove(&victim_seq)
                    .expect("victim was just found");
                evicted = Some(victim);
            }
            // An eviction does not shrink the estimate: the victim leaves the
            // engine only once its kill lands.
            let counted = state.estimate.as_mut().map(|estimate| {
                estimate.dispatching = estimate.dispatching.saturating_add(1);
                Counted::Dispatching
            });
            let seq = state.next_seq;
            state.next_seq = state.next_seq.wrapping_add(1);
            let evict = CancellationToken::new();
            state.entries.insert(
                seq,
                Entry {
                    priority,
                    evict: evict.clone(),
                },
            );
            (
                MarginCharge {
                    gate: Arc::clone(self),
                    seq,
                    evict,
                    counted,
                },
                evicted,
            )
        };
        self.metrics.received_direct();
        if let Some(victim) = evicted {
            tracing::info!(
                victim_priority = victim.priority,
                priority,
                "Evicting a lower-priority request at the admission margin"
            );
            victim.evict.cancel();
            self.metrics.rejected_evicted();
        }
        Ok(charge)
    }

    fn refund(&self, counted: Counted) {
        let mut state = self.state.lock();
        let Some(estimate) = state.estimate.as_mut() else {
            return;
        };
        match counted {
            Counted::Dispatching => {
                estimate.dispatching = estimate.dispatching.saturating_sub(1);
            }
            Counted::Queued(generation) if estimate.generation == generation => {
                estimate.admitted_since_report = estimate.admitted_since_report.saturating_sub(1);
            }
            Counted::Queued(_) => {}
        }
    }

    /// The engine has the request: count it until the next complete report.
    fn dispatched(&self) -> Option<Counted> {
        let mut state = self.state.lock();
        let estimate = state.estimate.as_mut()?;
        estimate.dispatching = estimate.dispatching.saturating_sub(1);
        estimate.admitted_since_report = estimate.admitted_since_report.saturating_add(1);
        Some(Counted::Queued(estimate.generation))
    }

    fn release(&self, seq: u64) {
        self.state.lock().entries.remove(&seq);
    }

    /// Engine-queue estimate, `None` until the engine has reported.
    #[cfg(test)]
    fn queue_estimate(&self) -> Option<u64> {
        self.state.lock().estimate.map(QueueEstimate::total)
    }

    #[cfg(test)]
    fn reported_queue(&self) -> Option<u64> {
        self.state.lock().estimate.map(|estimate| estimate.reported)
    }

    #[cfg(test)]
    fn inflight(&self) -> usize {
        self.inflight_for_test()
    }

    #[cfg(test)]
    pub(crate) fn inflight_for_test(&self) -> usize {
        self.state.lock().entries.len()
    }
}

/// One admitted request's hold on the margin: its registry entry and, if the
/// engine had reported when it was admitted, its count against the queue
/// estimate. Dropping it gives back both.
pub(crate) struct MarginCharge {
    gate: Arc<MarginGate>,
    seq: u64,
    evict: CancellationToken,
    /// How the admission is counted against the queue estimate.
    counted: Option<Counted>,
}

/// How one admission is counted against the queue estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Counted {
    /// Still being dispatched to the engine.
    Dispatching,
    /// With the engine, until the complete report of a later generation.
    Queued(u64),
}

impl MarginCharge {
    /// Give back this admission's count against the queue estimate: the
    /// request has left the engine queue. Idempotent; a no-op once a later
    /// complete report has superseded the count.
    fn refund_queue_credit(&mut self) {
        if let Some(counted) = self.counted.take() {
            self.gate.refund(counted);
        }
    }

    /// The engine has the request, so a later complete report includes it.
    fn mark_dispatched(&mut self) {
        if self.counted == Some(Counted::Dispatching) {
            self.counted = self.gate.dispatched();
        }
    }

    /// Wrap the engine's stream so it refunds at the first item, releases at
    /// the end, and ends early if the request is evicted, handing its charge
    /// to `eviction` to hold until the request is killed.
    pub(crate) fn attach<R: Data>(
        self,
        inner: EngineStream<R>,
        eviction: Eviction,
    ) -> MarginAdmittedStream<R> {
        let evicted = Box::pin(self.evict.clone().cancelled_owned());
        MarginAdmittedStream {
            inner,
            evicted,
            charge: Some(self),
            eviction,
        }
    }

    /// Handle through which the caller learns whether this request was
    /// evicted.
    pub(crate) fn eviction(&self) -> Eviction {
        Eviction(Some(Arc::new(EvictionState {
            token: self.evict.clone(),
            held: Mutex::new(None),
        })))
    }
}

impl Drop for MarginCharge {
    fn drop(&mut self) {
        self.refund_queue_credit();
        self.gate.release(self.seq);
    }
}

struct EvictionState {
    token: CancellationToken,
    /// An evicted request's charge, kept so the request still counts against
    /// the queue estimate until it is killed.
    held: Mutex<Option<MarginCharge>>,
}

/// Whether an admitted request was evicted by the margin policy.
#[derive(Clone, Default)]
pub struct Eviction(Option<Arc<EvictionState>>);

impl std::fmt::Debug for Eviction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eviction")
            .field("evicted", &self.evicted())
            .finish()
    }
}

impl Eviction {
    /// True once the request has been evicted; its stream has ended early and
    /// the caller must send [`Self::error`] and then [`Self::kill`] it.
    pub fn evicted(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(|state| state.token.is_cancelled())
    }

    /// Resolve once the request is evicted; never for one that cannot be.
    pub(crate) async fn wait(&self) {
        match self.0.as_ref() {
            Some(state) => state.token.cancelled().await,
            None => std::future::pending().await,
        }
    }

    /// Kill an evicted request so the engine aborts it, and only then give
    /// back its count against the queue estimate.
    pub fn kill(&self, context: &dyn crate::engine::AsyncEngineContext) {
        context.kill();
        if let Some(state) = self.0.as_ref() {
            drop(state.held.lock().take());
        }
    }

    fn hold(&self, charge: MarginCharge) {
        if let Some(state) = self.0.as_ref() {
            *state.held.lock() = Some(charge);
        }
    }

    /// Whether `error` is the one an evicted request fails with.
    pub fn is_eviction(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| {
            cause
                .downcast_ref::<DynamoError>()
                .is_some_and(|error| error.message() == EVICTED_MESSAGE)
        })
    }

    /// The error a request evicted before the engine had it fails with. It
    /// never started, so it is refused like an arrival at the margin, and
    /// the frontend places it on another worker.
    pub fn dispatch_error() -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::WorkerOverloaded)
            .message(EVICTED_MESSAGE)
            .build()
    }

    /// The error an evicted request's stream ends with.
    pub fn error() -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message(EVICTED_MESSAGE)
            .build()
    }
}

/// The engine's stream under the margin policy.
pub(crate) struct MarginAdmittedStream<R: Data> {
    inner: EngineStream<R>,
    evicted: Pin<Box<WaitForCancellationFutureOwned>>,
    /// Taken at end-of-stream or eviction, so a caller that keeps the stream
    /// around does not keep its entry.
    charge: Option<MarginCharge>,
    eviction: Eviction,
}

impl<R: Data> Stream for MarginAdmittedStream<R> {
    type Item = R;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.charge.is_some() && self.evicted.as_mut().poll(cx).is_ready() {
            // End the stream; the caller sends the overload error and then
            // kills the request, which is when its count is given back.
            if let Some(charge) = self.charge.take() {
                self.eviction.hold(charge);
            }
            return Poll::Ready(None);
        }
        let polled = self.inner.as_mut().poll_next(cx);
        // A lazy engine stream (a Python generator) submits the request on its
        // first poll, so only now can a later report include it.
        if let Some(charge) = self.charge.as_mut() {
            charge.mark_dispatched();
        }
        match &polled {
            Poll::Ready(Some(_)) => {
                if let Some(charge) = self.charge.as_mut() {
                    charge.refund_queue_credit();
                }
            }
            Poll::Ready(None) => drop(self.charge.take()),
            Poll::Pending => {}
        }
        polled
    }
}

impl<R: Data> AsyncEngineContextProvider for MarginAdmittedStream<R> {
    fn context(&self) -> Arc<dyn crate::engine::AsyncEngineContext> {
        self.inner.context()
    }
}

impl<R: Data> AsyncEngineStream<R> for MarginAdmittedStream<R> {}

impl<R: Data> std::fmt::Debug for MarginAdmittedStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarginAdmittedStream")
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{AsyncEngineContext, ResponseStream};
    use crate::pipeline::context::Controller;
    use futures::StreamExt;

    fn gate(margin: u64) -> Arc<MarginGate> {
        MarginGate::new(margin, Arc::new(BackendAdmissionMetrics::new(1, 0)))
    }

    fn rejected(result: Result<MarginCharge, DynamoError>) -> DynamoError {
        match result {
            Ok(_) => panic!("expected a rejection"),
            Err(error) => error,
        }
    }

    #[test]
    fn unreported_engine_admits_freely() {
        let gate = gate(0);
        let _a = gate.admit(i32::MIN).expect("unenforced");
        assert_eq!(gate.queue_estimate(), None);
    }

    #[test]
    fn admits_below_margin_and_rejects_at_margin_same_priority() {
        let gate = gate(2);
        gate.record_waiting(0, 1);
        let _a = gate.admit(0).expect("below the margin");
        gate.record_waiting(0, 2);
        let error = rejected(gate.admit(0));
        assert_eq!(error.error_type(), ErrorType::WorkerOverloaded);
        assert_eq!(gate.inflight(), 1);
    }

    #[test]
    fn admissions_since_the_last_report_count_toward_the_margin() {
        let gate = gate(3);
        gate.record_waiting(0, 1);
        let mut a = gate.admit(0).unwrap();
        assert_eq!(gate.queue_estimate(), Some(2));
        let mut b = gate.admit(0).unwrap();
        assert_eq!(gate.queue_estimate(), Some(3));
        a.mark_dispatched();
        b.mark_dispatched();
        rejected(gate.admit(0));
        assert_eq!(
            gate.reported_queue(),
            Some(1),
            "the raw report is untouched"
        );

        gate.record_waiting(0, 0);
        assert_eq!(gate.queue_estimate(), Some(0), "a fresh report supersedes");
        let _c = gate.admit(0).unwrap();
    }

    #[test]
    fn releasing_the_entry_alone_keeps_the_queue_credit_until_refunded() {
        let gate = gate(2);
        gate.record_waiting(0, 0);
        let mut a = gate.admit(0).unwrap();
        let _b = gate.admit(0).unwrap();
        gate.release(a.seq);
        assert_eq!(gate.queue_estimate(), Some(2));
        rejected(gate.admit(0));
        a.refund_queue_credit();
        assert_eq!(gate.queue_estimate(), Some(1));
    }

    #[test]
    fn queue_credit_refunds_when_the_request_leaves_the_queue() {
        // An engine whose idle rank never reports afresh never completes a
        // report; each admission must still be given back once the request
        // leaves the engine queue.
        let gate = gate(2);
        gate.record_waiting(0, 0);
        gate.record_waiting(1, 0);
        gate.record_waiting(0, 0); // rank 1 idle from here on: partial only
        let mut a = gate.admit(0).unwrap();
        let mut b = gate.admit(0).unwrap();
        rejected(gate.admit(0));
        a.refund_queue_credit();
        assert_eq!(gate.queue_estimate(), Some(1));
        let mut c = gate.admit(0).unwrap();
        b.refund_queue_credit();
        c.refund_queue_credit();
        assert_eq!(gate.queue_estimate(), Some(0));
        drop((a, b, c));
        assert_eq!(gate.queue_estimate(), Some(0), "drop never double-refunds");
        assert_eq!(gate.inflight(), 0);
    }

    #[test]
    fn partial_reports_refresh_the_depth_but_keep_local_admissions() {
        let gate = gate(3);
        gate.record_waiting(0, 1);
        assert_eq!(
            gate.queue_estimate(),
            Some(1),
            "first observation is complete"
        );
        gate.record_waiting(1, 0);
        // Rank 1 is new: ranks {0, 1} known, only 1 fresh -> partial.
        let mut a = gate.admit(0).unwrap();
        a.mark_dispatched();
        assert_eq!(gate.queue_estimate(), Some(2));
        gate.record_waiting(1, 0);
        assert_eq!(gate.queue_estimate(), Some(2), "local admission kept");
        gate.record_waiting(0, 0);
        assert_eq!(gate.queue_estimate(), Some(0), "complete report resets");
    }

    #[test]
    fn a_report_does_not_forget_admissions_still_being_dispatched() {
        // A report taken before the engine has the request cannot include it.
        let gate = gate(1);
        gate.record_waiting(0, 0);
        let mut a = gate.admit(0).unwrap();
        gate.record_waiting(0, 0);
        assert_eq!(gate.queue_estimate(), Some(1), "still counted");
        rejected(gate.admit(0));

        a.mark_dispatched();
        assert_eq!(gate.queue_estimate(), Some(1), "now counted as queued");
        gate.record_waiting(0, 1);
        assert_eq!(gate.queue_estimate(), Some(1), "the report includes it");
        drop(a);
        assert_eq!(
            gate.queue_estimate(),
            Some(1),
            "superseded, nothing to refund"
        );

        let b = gate.admit(-1);
        assert!(b.is_err(), "at the margin");
        gate.record_waiting(0, 0);
        let c = gate.admit(0).unwrap();
        drop(c);
        assert_eq!(gate.queue_estimate(), Some(0), "a failed dispatch refunds");
    }

    #[test]
    fn waiting_is_summed_across_ranks() {
        let gate = gate(10);
        gate.record_waiting(0, 3);
        gate.record_waiting(1, 4);
        assert_eq!(gate.reported_queue(), Some(7));
    }

    #[test]
    fn dispatch_failure_refunds_within_its_report_interval() {
        let gate = gate(2);
        gate.record_waiting(0, 0);
        let mut a = gate.admit(0).unwrap();
        a.refund_queue_credit();
        a.refund_queue_credit();
        assert_eq!(gate.queue_estimate(), Some(0), "refund is idempotent");

        let mut b = gate.admit(0).unwrap();
        gate.record_waiting(0, 1);
        b.refund_queue_credit();
        assert_eq!(
            gate.queue_estimate(),
            Some(1),
            "a later complete report already superseded the count"
        );
    }

    #[test]
    fn eviction_admission_still_counts_toward_the_estimate() {
        let gate = gate(1);
        gate.record_waiting(0, 0);
        let low = gate.admit(-1).unwrap();
        assert_eq!(gate.queue_estimate(), Some(1));
        let _high = gate.admit(0).expect("evicts the lower-priority request");
        assert!(low.eviction().evicted());
        assert_eq!(gate.queue_estimate(), Some(2));
    }

    #[test]
    fn full_queue_with_no_admitted_requests_rejects() {
        // The engine can report waiting work this process never admitted
        // through the margin; with no victim the request must be refused.
        let gate = gate(1);
        gate.record_waiting(0, 5);
        rejected(gate.admit(0));
    }

    #[test]
    fn evicts_lowest_priority_victim_at_margin() {
        let gate = gate(1);
        let bg = gate.admit(-100).unwrap();
        let flex = gate.admit(-10).unwrap();
        gate.record_waiting(0, 1);
        let _rt = gate.admit(0).unwrap();
        assert!(bg.eviction().evicted());
        assert!(!flex.eviction().evicted());
        assert_eq!(gate.inflight(), 2, "slot transferred, not leaked");
    }

    #[test]
    fn eviction_tie_breaks_most_recently_admitted() {
        let gate = gate(0);
        let older = gate.admit(-5).unwrap();
        let newer = gate.admit(-5).unwrap();
        gate.record_waiting(0, 0);
        let _arrival = gate.admit(0).unwrap();
        assert!(newer.eviction().evicted());
        assert!(!older.eviction().evicted());
    }

    #[test]
    fn equal_priority_is_not_a_victim() {
        let gate = gate(0);
        let _a = gate.admit(-7).unwrap();
        gate.record_waiting(0, 0);
        rejected(gate.admit(-7));
    }

    #[test]
    fn victims_release_is_a_no_op() {
        let gate = gate(0);
        let victim = gate.admit(-1).unwrap();
        gate.record_waiting(0, 0);
        let _arrival = gate.admit(0).unwrap();
        assert_eq!(gate.inflight(), 1);
        drop(victim);
        assert_eq!(gate.inflight(), 1);
    }

    #[test]
    fn margin_parsing() {
        assert_eq!(margin_from_raw(None), None);
        assert_eq!(margin_from_raw(Some("64")), Some(64));
        assert_eq!(margin_from_raw(Some(" 0 ")), Some(0));
        assert_eq!(margin_from_raw(Some("-1")), None);
        assert_eq!(margin_from_raw(Some("abc")), None);
    }

    fn engine_stream(items: Vec<u32>, pending_after: bool) -> (EngineStream<u32>, Arc<Controller>) {
        let controller = Arc::new(Controller::default());
        let stream = futures::stream::iter(items);
        let stream: Pin<Box<dyn Stream<Item = u32> + Send>> = if pending_after {
            Box::pin(stream.chain(futures::stream::pending()))
        } else {
            Box::pin(stream)
        };
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        (ResponseStream::new(stream, context), controller)
    }

    #[tokio::test]
    async fn stream_refunds_at_first_item_and_releases_at_end() {
        let gate = gate(5);
        gate.record_waiting(0, 0);
        let charge = gate.admit(0).unwrap();
        let eviction = charge.eviction();
        let (inner, _controller) = engine_stream(vec![1, 2], false);
        let mut stream = Box::pin(charge.attach(inner, eviction));
        assert_eq!(gate.queue_estimate(), Some(1));
        assert_eq!(stream.next().await, Some(1));
        assert_eq!(gate.queue_estimate(), Some(0), "running, no longer queued");
        assert_eq!(gate.inflight(), 1);
        assert_eq!(stream.next().await, Some(2));
        assert_eq!(stream.next().await, None);
        assert_eq!(gate.inflight(), 0);
    }

    #[tokio::test]
    async fn evicted_waiting_request_counts_until_killed() {
        let gate = gate(1);
        gate.record_waiting(0, 0);
        let victim = gate.admit(-1).unwrap();
        let eviction = victim.eviction();
        // The victim is still waiting in the engine: no item yet.
        let (inner, controller) = engine_stream(vec![], true);
        let mut stream = Box::pin(victim.attach(inner, eviction.clone()));

        let _arrival = gate.admit(0).expect("evicts the waiting request");
        assert!(eviction.evicted());
        // The stream would otherwise stay pending forever.
        assert_eq!(stream.next().await, None);
        assert_eq!(gate.inflight(), 1, "the victim is no longer a candidate");
        assert_eq!(
            gate.queue_estimate(),
            Some(2),
            "the victim still counts until the engine is told to abort it"
        );
        assert!(!controller.is_killed());

        eviction.kill(controller.as_ref());
        assert!(controller.is_killed());
        assert_eq!(gate.queue_estimate(), Some(1));
        assert_eq!(Eviction::error().error_type(), ErrorType::ResourceExhausted);
    }
}
