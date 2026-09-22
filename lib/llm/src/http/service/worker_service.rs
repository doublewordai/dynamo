// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Successful HTTP work attributed to its serving workers. The HTTP guard and
//! response stream have independent lifetimes; publish only after both finish.
//! Worker IDs use the runtime's hexadecimal spelling, with a decimal alias for
//! joining the existing frontend load metrics. No request IDs become labels.

use prometheus::{IntCounterVec, Opts, Registry};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

static COMPLETED: LazyLock<IntCounterVec> = LazyLock::new(|| {
    counter(
        "completed_requests",
        "Successful HTTP requests served by a worker phase",
    )
});
static OUTPUT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    counter(
        "output_tokens",
        "Output tokens in successful HTTP requests served by a worker phase",
    )
});
static INPUT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    counter(
        "input_tokens",
        "Input tokens in successful HTTP requests served by a worker phase",
    )
});

fn counter(name: &str, help: &str) -> IntCounterVec {
    IntCounterVec::new(
        Opts::new(format!("dynamo_frontend_worker_{name}_total"), help),
        &["model", "worker_id", "worker_id_decimal", "phase"],
    )
    .expect("valid worker service metric")
}

pub fn register(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(COMPLETED.clone()))?;
    registry.register(Box::new(OUTPUT.clone()))?;
    registry.register(Box::new(INPUT.clone()))?;
    Ok(())
}

/// Which workers have series, under which model names (a worker serves
/// its base model and its adapters), and which workers discovery has
/// withdrawn. A request that finishes after its worker left, however long
/// it runs, must not recreate the worker's series, so publication and
/// withdrawal both run under this lock. A withdrawn worker costs one id
/// until it returns.
#[derive(Default)]
struct Workers {
    models: HashMap<u64, HashSet<String>>,
    retired: HashSet<u64>,
}

static WORKERS: LazyLock<Mutex<Workers>> = LazyLock::new(|| Mutex::new(Workers::default()));

fn workers() -> std::sync::MutexGuard<'static, Workers> {
    WORKERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Discovery withdrew the worker: drop its series, for every model it
/// served, and refuse late ones.
pub fn remove(worker: u64) {
    let mut workers = workers();
    workers.retired.insert(worker);
    let hex = format!("{worker:x}");
    let decimal = worker.to_string();
    for model in workers.models.remove(&worker).unwrap_or_default() {
        for phase in ["aggregated", "prefill", "decode"] {
            for metric in [&*COMPLETED, &*OUTPUT, &*INPUT] {
                let _ = metric.remove_label_values(&[&model, &hex, &decimal, phase]);
            }
        }
    }
}

/// Discovery admitted the worker, again if it had left: an adapter reloaded
/// in the same process returns under the same instance id.
pub fn readmit(worker: u64) {
    workers().retired.remove(&worker);
}

#[derive(Default)]
pub struct Summary {
    pub prefill: Option<u64>,
    pub decode: Option<u64>,
    pub input: u64,
    pub output: u64,
    pub ambiguous: bool,
}

#[derive(Default)]
struct State {
    success: Option<bool>,
    summary: Option<Summary>,
    published: bool,
}

pub struct Request {
    model: String,
    state: Mutex<State>,
}

impl Request {
    pub fn new(model: String) -> Self {
        Self {
            model,
            state: Mutex::new(State::default()),
        }
    }

    pub fn finish(&self, success: bool) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.success = Some(success);
        self.publish(&mut state);
    }

    pub fn observe(&self, summary: Summary) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.summary = Some(summary);
        self.publish(&mut state);
    }

    fn publish(&self, state: &mut State) {
        if state.published || state.success != Some(true) {
            return;
        }
        let Some(summary) = &state.summary else {
            return;
        };
        state.published = true;
        // Retries/migrations that change workers cannot be charged entirely to
        // one worker. Missing attribution is absent, never a fabricated zero.
        if summary.ambiguous {
            return;
        }
        let mut workers = workers();
        let mut record = |worker: u64, phase: &str, input: u64, output: u64| {
            if workers.retired.contains(&worker) {
                return;
            }
            workers
                .models
                .entry(worker)
                .or_default()
                .insert(self.model.clone());
            let hex = format!("{worker:x}");
            let decimal = worker.to_string();
            let labels = &[self.model.as_str(), hex.as_str(), decimal.as_str(), phase];
            COMPLETED.with_label_values(labels).inc();
            INPUT.with_label_values(labels).inc_by(input);
            OUTPUT.with_label_values(labels).inc_by(output);
        };
        match (summary.prefill, summary.decode) {
            (Some(p), Some(d)) if p != d => {
                record(p, "prefill", summary.input, 0);
                record(d, "decode", 0, summary.output);
            }
            // Aggregated routing records the same worker in both roles.
            (Some(p), Some(d)) if p == d => record(d, "aggregated", summary.input, summary.output),
            // A decode-only annotation is insufficient to rule out a separate
            // prefill worker. Expose it as decode, not whole-request capacity.
            (_, Some(d)) => record(d, "decode", 0, summary.output),
            // Prefill that reached the end of the sequence answered alone.
            (Some(p), _) => record(p, "prefill", summary.input, summary.output),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary() -> Summary {
        Summary {
            prefill: Some(42),
            decode: Some(42),
            input: 20,
            output: 7,
            ambiguous: false,
        }
    }

    #[test]
    fn publishes_once_in_either_drop_order_and_excludes_failures() {
        for first in [true, false] {
            let model = format!("service-order-{first}");
            let request = Request::new(model.clone());
            let labels = &[model.as_str(), "2a", "42", "aggregated"];
            if first {
                request.finish(true);
            } else {
                request.observe(summary());
            }
            assert_eq!(COMPLETED.with_label_values(labels).get(), 0);
            if first {
                request.observe(summary());
            } else {
                request.finish(true);
            }
            request.finish(true);
            assert_eq!(COMPLETED.with_label_values(labels).get(), 1);
            assert_eq!(OUTPUT.with_label_values(labels).get(), 7);
            assert_eq!(INPUT.with_label_values(labels).get(), 20);
        }
        let failed = Request::new("service-failed".into());
        failed.observe(summary());
        failed.finish(false);
        assert_eq!(
            COMPLETED
                .with_label_values(&["service-failed", "2a", "42", "aggregated"])
                .get(),
            0
        );
    }

    #[test]
    fn splits_disaggregated_work_and_excludes_ambiguous_workers() {
        let request = Request::new("service-split".into());
        request.observe(Summary {
            decode: Some(43),
            ..summary()
        });
        request.finish(true);
        assert_eq!(
            INPUT
                .with_label_values(&["service-split", "2a", "42", "prefill"])
                .get(),
            20
        );
        assert_eq!(
            OUTPUT
                .with_label_values(&["service-split", "2b", "43", "decode"])
                .get(),
            7
        );
        assert_eq!(
            COMPLETED
                .with_label_values(&["service-split", "2b", "43", "aggregated"])
                .get(),
            0
        );
        let request = Request::new("service-migrated".into());
        request.observe(Summary {
            ambiguous: true,
            ..summary()
        });
        request.finish(true);
        assert_eq!(
            COMPLETED
                .with_label_values(&["service-migrated", "2a", "42", "aggregated"])
                .get(),
            0
        );
        remove(43);
        assert_eq!(
            OUTPUT
                .with_label_values(&["service-split", "2b", "43", "decode"])
                .get(),
            0
        );
        // A request finishing after its worker left does not bring it back;
        // one finishing after the worker returned counts again.
        let late = Request::new("service-split".into());
        late.observe(Summary {
            decode: Some(43),
            ..summary()
        });
        late.finish(true);
        assert_eq!(
            OUTPUT
                .with_label_values(&["service-split", "2b", "43", "decode"])
                .get(),
            0
        );
        readmit(43);
        let returned = Request::new("service-split".into());
        returned.observe(Summary {
            decode: Some(43),
            ..summary()
        });
        returned.finish(true);
        assert_eq!(
            OUTPUT
                .with_label_values(&["service-split", "2b", "43", "decode"])
                .get(),
            7
        );
    }
}
