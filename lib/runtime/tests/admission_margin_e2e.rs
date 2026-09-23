// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the engine-queue admission margin over a real
//! request plane. The admission gate is process-global and reads
//! `DYN_ADMISSION_QUEUE_MARGIN` once, so this must remain in its own
//! integration test binary.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Error;
use async_trait::async_trait;
use futures::StreamExt;

use dynamo_runtime::admission_gate::{self, ADMISSION_PRIORITY_METADATA_KEY};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider, DataStream},
    pipeline::{ManyOut, ResponseStream, SingleIn, context::Context, network::Ingress},
    protocols::annotated::Annotated,
    protocols::maybe_error::MaybeError,
};

/// Yields the request value once, then holds the request as running. Like a
/// real engine's cancellation monitor, a separate task watches the request
/// context and counts kills.
struct HoldingEngine {
    killed: Arc<AtomicUsize>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u64>, ManyOut<Annotated<u64>>, Error> for HoldingEngine {
    async fn generate(&self, input: SingleIn<u64>) -> Result<ManyOut<Annotated<u64>>, Error> {
        let ctx = input.context();
        let (value, _) = input.into_parts();
        let killed = Arc::clone(&self.killed);
        let watch = Arc::clone(&ctx);
        tokio::spawn(async move {
            watch.killed().await;
            killed.fetch_add(1, Ordering::SeqCst);
        });
        let stream = futures::stream::once(async move { Annotated::from_data(value) })
            .chain(futures::stream::pending());
        let stream: DataStream<Annotated<u64>> = Box::pin(stream);
        Ok(ResponseStream::new(stream, ctx))
    }
}

/// Never finishes dispatching: its `generate` blocks, as a backend can while
/// it waits to submit. A separate task counts kills of the request.
struct StuckDispatchEngine {
    killed: Arc<AtomicUsize>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u64>, ManyOut<Annotated<u64>>, Error> for StuckDispatchEngine {
    async fn generate(&self, input: SingleIn<u64>) -> Result<ManyOut<Annotated<u64>>, Error> {
        let ctx = input.context();
        let killed = Arc::clone(&self.killed);
        tokio::spawn(async move {
            ctx.killed().await;
            killed.fetch_add(1, Ordering::SeqCst);
        });
        std::future::pending().await
    }
}

fn request(value: u64, priority: Option<i32>) -> SingleIn<u64> {
    let mut request = Context::new(value);
    if let Some(priority) = priority {
        request.insert_metadata(ADMISSION_PRIORITY_METADATA_KEY, priority.to_string());
    }
    request
}

#[tokio::test]
async fn margin_rejects_at_the_margin_and_evicts_a_running_lower_priority_request() {
    // This test binary owns the process and sets the margin before the first
    // request touches the gate.
    unsafe {
        std::env::set_var("DYN_ADMISSION_QUEUE_MARGIN", "1");
    }

    let rt = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let ns = drt.namespace("test_admission_margin".to_string()).unwrap();
    let component = ns.component("worker".to_string()).unwrap();
    let endpoint = component.endpoint("generate".to_string());

    let killed = Arc::new(AtomicUsize::new(0));
    let ingress = Ingress::for_engine(Arc::new(HoldingEngine {
        killed: Arc::clone(&killed),
    }))
    .unwrap();
    let endpoint_for_server = endpoint.clone();
    tokio::spawn(async move {
        let _ = endpoint_for_server
            .endpoint_builder()
            .handler(ingress)
            .start()
            .await;
    });

    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();
    let probe = client.clone();
    let router = PushRouter::<u64, Annotated<u64>>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();

    // The engine reports an empty queue: enforcement starts.
    admission_gate::record_engine_waiting(0, 0);

    // A low-priority request is admitted and starts running.
    let mut low = router.generate(request(1, Some(-1))).await.unwrap();
    let first = low.next().await.expect("the running request yields");
    assert_eq!(first.data, Some(1));

    // The engine now reports one waiting request: the queue is at the margin.
    admission_gate::record_engine_waiting(0, 1);

    // An arrival of the same priority has no victim and is refused as a
    // worker-scoped overload, which the frontend's migration retries on
    // another worker.
    let refusal = router
        .generate(request(2, Some(-1)))
        .await
        .expect_err("refused before a response stream opens");
    assert!(
        refusal.chain().any(|cause| cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.reason().as_str() == "capacity.worker_overloaded")),
        "unexpected refusal: {refusal:#}"
    );

    // The refusal is backpressure: the worker stays routable, on the router's
    // one-second overload lease rather than quarantined as faulty. Wait the
    // lease out, since this endpoint has no other worker to steer to.
    let worker = probe.instance_ids()[0];
    assert!(
        probe
            .available_instance_ids()
            .is_some_and(|available| available.contains(&worker)),
        "an admission refusal must not quarantine the worker"
    );
    assert!(
        probe
            .overloaded_instance_ids()
            .is_some_and(|overloaded| overloaded.contains(&worker))
    );
    tokio::time::sleep(Duration::from_millis(1100)).await;

    // A request the frontend did not stamp is not subject to the margin; it
    // stays under the default engine request limit.
    let mut unstamped = router.generate(request(3, None)).await.unwrap();
    assert_eq!(unstamped.next().await.unwrap().data, Some(3));

    // A higher-priority arrival evicts the running low-priority request.
    let mut high = router.generate(request(4, Some(5))).await.unwrap();
    assert_eq!(high.next().await.unwrap().data, Some(4));

    let evicted = tokio::time::timeout(Duration::from_secs(10), low.next())
        .await
        .expect("the evicted stream ends promptly")
        .expect("the evicted stream carries an error item");
    let error = evicted.err().expect("the item is an error");
    assert_eq!(error.error_type(), ErrorType::ResourceExhausted);
    tokio::time::timeout(Duration::from_secs(10), async {
        while killed.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the evicted request is killed in the engine");

    drop((high, unstamped));

    // A request evicted while its backend is still being dispatched to never
    // started: it is refused as worker backpressure, then aborted.
    let stuck_endpoint = component.endpoint("stuck".to_string());
    let stuck_killed = Arc::new(AtomicUsize::new(0));
    let stuck_ingress = Ingress::for_engine(Arc::new(StuckDispatchEngine {
        killed: Arc::clone(&stuck_killed),
    }))
    .unwrap();
    let stuck_for_server = stuck_endpoint.clone();
    tokio::spawn(async move {
        let _ = stuck_for_server
            .endpoint_builder()
            .handler(stuck_ingress)
            .start()
            .await;
    });
    let stuck_client = stuck_endpoint.client().await.unwrap();
    stuck_client.wait_for_instances().await.unwrap();
    let stuck_router =
        PushRouter::<u64, Annotated<u64>>::from_client(stuck_client, RouterMode::RoundRobin)
            .await
            .unwrap();
    admission_gate::record_engine_waiting(0, 0);
    let stuck = tokio::spawn(async move {
        stuck_router
            .generate(request(5, Some(-10)))
            .await
            .map(|_| ())
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    admission_gate::record_engine_waiting(0, 1);
    let mut evictor = router.generate(request(6, Some(10))).await.unwrap();
    assert_eq!(evictor.next().await.unwrap().data, Some(6));
    let refusal = tokio::time::timeout(Duration::from_secs(10), stuck)
        .await
        .expect("the evicted dispatch is answered promptly")
        .unwrap()
        .expect_err("an evicted dispatch fails before a stream opens");
    assert!(
        refusal.chain().any(|cause| cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.reason().as_str() == "capacity.worker_overloaded")),
        "unexpected refusal: {refusal:#}"
    );
    tokio::time::timeout(Duration::from_secs(10), async {
        while stuck_killed.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the evicted request is killed after its refusal is sent");

    drop(evictor);
    drt.shutdown();
}
