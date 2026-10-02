// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dw-spillover`: a Dynamo worker-selection policy.
//!
//! Cost per eligible worker, lowest wins:
//!
//! ```text
//! cost = baseline cost (Dynamo's default scorer: uncached prompt + in-flight load)      1. affinity
//!      + (proxy tier ? tier.penalty_blocks + tier.weight_blocks
//!                    : primary occupancy > threshold ? failover_penalty_blocks : 0)    2. failover
//!      + pending_weight_blocks * active_requests                                          3. preference
//! ```
//!
//! `tier.weight_blocks` applies only to a proxy worker, never to a primary candidate; the
//! `pending_weight_blocks * active_requests` term applies to every candidate.
//!
//! Proxy workers are identified by their data-parallel rank (see `params::TierParameters`).
//! Models with no entry in the parameters route exactly like Dynamo's default policy.

pub mod baseline;
pub mod params;
pub mod policy;
pub mod tier;

/// Policy type name used in the router-policy YAML.
pub const POLICY_TYPE: &str = "dw-spillover";

pub use params::{ModelParameters, SpilloverParameters, TierParameters};
pub use policy::{build_policy, provider, register};
pub use tier::TierScorer;
