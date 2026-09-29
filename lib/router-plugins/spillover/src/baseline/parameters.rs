// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copied from ai-dynamo/dynamo@494d6e24 lib/router-plugins/builtin/src/default/

//! Startup parameters for the default scorer and picker.

use dynamo_kv_router::KvRouterConfig;

/// Only the startup values consumed by the default scorer and picker.
#[derive(Clone, Copy)]
pub(super) struct PolicyParameters {
    pub(super) overlap_score_credit: f64,
    pub(super) overlap_score_credit_decay: f64,
    pub(super) prefill_load_scale: f64,
    pub(super) decode_active_request_weight: f64,
    pub(super) host_cache_hit_weight: f64,
    pub(super) disk_cache_hit_weight: f64,
    pub(super) shared_cache_multiplier: f64,
    pub(super) router_temperature: f64,
}

impl From<&KvRouterConfig> for PolicyParameters {
    fn from(config: &KvRouterConfig) -> Self {
        Self {
            overlap_score_credit: config.overlap_score_credit,
            overlap_score_credit_decay: config.overlap_score_credit_decay,
            prefill_load_scale: config.prefill_load_scale,
            decode_active_request_weight: config.decode_active_request_weight,
            host_cache_hit_weight: config.host_cache_hit_weight,
            disk_cache_hit_weight: config.disk_cache_hit_weight,
            shared_cache_multiplier: config.shared_cache_multiplier,
            router_temperature: config.router_temperature,
        }
    }
}
