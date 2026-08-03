// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use dynamo_runtime::pipeline::{Data, Error, ManyOut, SingleIn, async_trait};

use crate::session_affinity::AffinityTarget;

/// Why an exact target was supplied to a KV-routing strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvRoutePin {
    Explicit(AffinityTarget),
    Affinity(AffinityTarget),
}

/// Constraints shared by the token-overlap and reported-load strategies.
pub(crate) struct KvRoutingConstraints {
    pub(crate) pin: Option<KvRoutePin>,
    pub(crate) allowed_worker_ids: Option<HashSet<u64>>,
    pub(crate) session_id: String,
    pub(crate) affinity_action: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvDispatchMode {
    Exact,
    AllowFallback,
}

/// Request-scoped accounting owned by a routing strategy.
///
/// Token routing uses this to own scheduler state and response metrics. Text
/// routing uses it to own the frontend in-flight count for the selected rank.
#[async_trait]
pub(crate) trait KvRouteReservation<U: Data>: Send {
    fn start_dispatch(&mut self) {}

    async fn abort(&mut self);

    fn into_stream(self, stream: ManyOut<U>) -> ManyOut<U>;
}

pub(crate) struct KvSelectedRoute<U: Data, Reservation> {
    pub(crate) target: AffinityTarget,
    pub(crate) reservation: Reservation,
    pub(crate) dispatch_mode: KvDispatchMode,
    pub(crate) dispatch_span: tracing::Span,
    pub(crate) _response: std::marker::PhantomData<U>,
}

impl<U: Data, Reservation> KvSelectedRoute<U, Reservation> {
    pub(crate) fn new(
        target: AffinityTarget,
        reservation: Reservation,
        dispatch_mode: KvDispatchMode,
        dispatch_span: tracing::Span,
    ) -> Self {
        Self {
            target,
            reservation,
            dispatch_mode,
            dispatch_span,
            _response: std::marker::PhantomData,
        }
    }
}

pub(crate) enum KvRoutingOutcome<U: Data, Reservation> {
    Dispatch(KvSelectedRoute<U, Reservation>),
    LocalResponse(ManyOut<U>),
}

/// The strategy-specific part of KV routing.
///
/// `KvPushRouter` owns affinity and dispatch. Implementations only interpret
/// their request type, select an exact target, and reserve their local load
/// accounting until the response stream finishes.
#[async_trait]
pub(crate) trait KvRoutingStrategy<T>: Clone + Send + Sync + 'static
where
    T: Data,
{
    type Response: Data;
    type Reservation: KvRouteReservation<Self::Response>;

    fn explicit_target(&self, request: &SingleIn<T>) -> Result<Option<AffinityTarget>, Error>;

    fn is_query_only(&self, _request: &SingleIn<T>) -> bool {
        false
    }

    async fn select_and_reserve(
        &self,
        request: &SingleIn<T>,
        constraints: KvRoutingConstraints,
        affinity_active: bool,
    ) -> Result<KvRoutingOutcome<Self::Response, Self::Reservation>, Error>;

    fn apply_target(&self, request: &mut T, target: AffinityTarget);
}
