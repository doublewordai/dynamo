// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Doubleword's build-time catalog: registers the `dw-spillover` worker-selection policy.

use dynamo_kv_router::plugins::{RouterPluginRegistry, WorkerSelectionPolicyRegistryError};

/// Register policies linked into this image.
///
/// Adds `dw-spillover` (lib/router-plugins/spillover). It is only used when a router-policy
/// YAML selects it; `default` still selects Dynamo's built-in worker selector. The policies
/// Dynamo ships are registered separately from `dynamo-custom-policy-builtin`, so this adds
/// policies alongside them rather than displacing them.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    dw_spillover_policy::register(registry)
}
