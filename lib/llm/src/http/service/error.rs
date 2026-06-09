// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// Implementation of the Completion Engines served by the HTTP service should
/// map their custom errors to to this error type if they wish to return error
/// codes besides 500.
#[derive(Debug, Error)]
#[error("HTTP Error {code}: {message}")]
pub struct HttpError {
    pub code: u16,
    pub message: String,
}

/// Whether `code` is a genuine HTTP error status (4xx or 5xx).
///
/// Single source of truth for the "is this a real error code we trust?" check
/// used when surfacing a preserved upstream status, so the streaming and
/// non-streaming paths can't drift. Equivalent to
/// `StatusCode::is_client_error() || is_server_error()`.
pub(crate) fn is_http_error_code(code: u16) -> bool {
    (400..600).contains(&code)
}
