// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap profile endpoint.
//!
//! Serves the jemalloc heap profile of this process in pprof format so a
//! continuous profiler can scrape it. The process must run with jemalloc as
//! its global allocator and profiling activated through `_RJEM_MALLOC_CONF`.

use std::collections::HashMap;
use std::io::{Read, Write};

use axum::{
    Router,
    http::{Method, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::get,
};
use prost::Message;

use super::RouteDoc;
use super::pprof_proto::{Function, Line, Profile};

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
        prof_ctl
            .dump_pprof()
            .and_then(|pprof| name_unsymbolized_frames(&pprof))
            .map(Some)
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

/// Give every location that symbolization left without a function a name
/// derived from its mapping and address, such as `libc.so.6+0x9caa3`.
///
/// Profile viewers build their trees from function names; a stack whose
/// outermost frame has none (thread-entry code in a stripped library, say)
/// cannot be placed and the whole profile collapses into a single node.
fn name_unsymbolized_frames(gzipped: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(gzipped).read_to_end(&mut raw)?;
    let mut profile = Profile::decode(raw.as_slice())?;

    let mapping_by_id: HashMap<u64, (u64, u64, String)> = profile
        .mapping
        .iter()
        .map(|m| {
            let file = profile
                .string_table
                .get(usize::try_from(m.filename).unwrap_or(0))
                .map(|f| f.rsplit('/').next().unwrap_or(f).to_string())
                .unwrap_or_default();
            (m.id, (m.memory_start, m.file_offset, file))
        })
        .collect();
    let first_free_function_id = profile.function.iter().map(|f| f.id).max().unwrap_or(0) + 1;
    let unsymbolized = profile.location.iter_mut().filter(|l| l.line.is_empty());
    for (function_id, location) in (first_free_function_id..).zip(unsymbolized) {
        let name = match mapping_by_id.get(&location.mapping_id) {
            Some((start, file_offset, file)) if !file.is_empty() => format!(
                "{file}+0x{:x}",
                location
                    .address
                    .wrapping_sub(*start)
                    .wrapping_add(*file_offset)
            ),
            _ => format!("unknown+0x{:x}", location.address),
        };
        profile.string_table.push(name);
        let name_index = i64::try_from(profile.string_table.len() - 1)?;
        profile.function.push(Function {
            id: function_id,
            name: name_index,
            system_name: name_index,
            filename: 0,
            start_line: 0,
        });
        location.line.push(Line {
            function_id,
            line: 0,
        });
    }

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&profile.encode_to_vec())?;
    Ok(encoder.finish()?)
}
