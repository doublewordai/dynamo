// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dw-proxy-worker --config proxy.yaml`
//!
//! Registers as a Tokens-input chat worker with the same model card as the SGLang workers it
//! joins, at a reserved DP rank, and serves each request from a third-party provider.
//!
//! We drive the process through `dynamo_backend_common::run`
//! (`lib/backend-common/src/run.rs`), the same entry point as the mock engine in
//! `lib/backend-common/examples/mocker/src/main.rs`: parse CLI -> build an
//! `LLMEngine` -> hand it to `run`, which owns the runtime and signal flow.

mod config;
mod engine;
mod kv;
mod metrics;
mod registration;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "dw-proxy-worker",
    about = "Dynamo worker that looks like SGLang to the router and serves from a third-party provider"
)]
struct Args {
    /// Path to the proxy YAML config.
    #[arg(long)]
    config: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // Install the subscriber before the first `tracing` call: `run` also calls
    // this, but its `init` runs after the startup line and engine construction
    // below, so without this those events are dropped. `init` is idempotent.
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let config = config::load(&args.config)?;
    tracing::info!(
        namespace = %config.namespace,
        component = %config.component,
        endpoint = %config.endpoint,
        dp_rank = config.dp_rank,
        tier = %config.tier,
        provider = %config.provider.name,
        "starting dw-proxy-worker"
    );

    let worker = registration::worker_config(&config);
    // `ProxyEngine::new` resolves a hub-id model path from the offline cache, so
    // it needs an async context. `run` below builds its own Dynamo runtime, so
    // keep this one scoped and drop it first.
    let engine = {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        runtime.block_on(engine::ProxyEngine::new(config))?
    };
    dynamo_backend_common::run(Arc::new(engine), worker)
}
