// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap profile endpoint.
//!
//! Serves the jemalloc heap profile of this process in pprof format so a
//! continuous profiler can scrape it. The process must run with jemalloc as
//! its global allocator and profiling activated through `_RJEM_MALLOC_CONF`.

use axum::{
    Router,
    http::{Method, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::get,
};

use super::RouteDoc;

/// Build the heap profile route at `path`.
pub fn router(path: String) -> (Vec<RouteDoc>, Router) {
    let docs = vec![RouteDoc::new(Method::GET, &path)];
    let router = Router::new().route(&path, get(handler));
    (docs, router)
}

async fn handler() -> Response {
    let Some(prof_ctl) = jemalloc_pprof::PROF_CTL.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "jemalloc profiling is not available in this process",
        )
            .into_response();
    };
    // Dumping and symbolizing the profile is CPU-bound and can take a while,
    // so it runs on the blocking pool rather than on a runtime worker.
    let prof_ctl = prof_ctl.clone();
    let dump = tokio::task::spawn_blocking(move || {
        let mut prof_ctl = prof_ctl.blocking_lock();
        if !prof_ctl.activated() {
            return Ok(None);
        }
        prof_ctl.dump_pprof().map(Some)
    })
    .await;
    match dump {
        Ok(Ok(Some(pprof))) => (
            StatusCode::OK,
            [(CONTENT_TYPE, "application/octet-stream")],
            pprof,
        )
            .into_response(),
        Ok(Ok(None)) => (
            StatusCode::FORBIDDEN,
            "heap profiling is not active; set _RJEM_MALLOC_CONF=prof:true,prof_active:true",
        )
            .into_response(),
        Ok(Err(err)) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}
