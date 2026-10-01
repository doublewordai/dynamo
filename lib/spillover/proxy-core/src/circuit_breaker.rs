// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-proxy circuit breaker.
//!
//! A proxy stays in the router's rotation while its provider is down, so every
//! request routed to it pays a provider attempt before the frontend migrates
//! the request to another worker. For a refused connection that is
//! milliseconds, but a blackholed host costs the connect timeout and a provider
//! that accepts and hangs costs the read timeout -- per request, for as long as
//! the outage lasts. The breaker trips after a configurable number of
//! consecutive provider-side failures and refuses new requests immediately,
//! before any provider call, with the same migratable error the proxy already
//! uses, so migration still works and the failed attempt is gone.
//!
//! The state machine is pure and clock-injected ([`CircuitBreaker::admit`] and
//! [`CircuitBreaker::record`] take an [`Instant`]), so it is unit-testable
//! without sleeping. The worker owns it behind a mutex; `proxy-core` has no
//! runtime dependency.
//!
//! States:
//!
//! - **Closed**: requests call the provider. Consecutive provider failures are
//!   counted; [`CircuitBreakerConfig::failure_threshold`] of them open the
//!   breaker.
//! - **Open**: requests are refused until
//!   [`CircuitBreakerConfig::cooldown_ms`] has elapsed.
//! - **Half-open**: exactly one request (the probe) is let through; concurrent
//!   requests are refused. A successful probe closes the breaker; a failed
//!   probe reopens it with the cooldown doubled, capped at
//!   [`CircuitBreakerConfig::max_cooldown_ms`]. Closing resets the backoff.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Tunables for the per-proxy circuit breaker.
///
/// The block is optional in the proxy config and this type's serde defaults
/// apply when it is present but a field is omitted. Absent entirely, the worker
/// still installs a breaker with these defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircuitBreakerConfig {
    /// Consecutive provider-side failures that open the breaker. Must be > 0.
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    /// How long the breaker stays open before a probe is let through, in
    /// milliseconds. Must be > 0.
    #[serde(default = "default_cooldown_ms")]
    pub cooldown_ms: u64,
    /// Ceiling for the doubling performed on each failed probe, in
    /// milliseconds. Must be >= [`Self::cooldown_ms`].
    #[serde(default = "default_max_cooldown_ms")]
    pub max_cooldown_ms: u64,
}

pub const DEFAULT_FAILURE_THRESHOLD: u32 = 5;
pub const DEFAULT_COOLDOWN_MS: u64 = 30_000;
pub const DEFAULT_MAX_COOLDOWN_MS: u64 = 300_000;

fn default_failure_threshold() -> u32 {
    DEFAULT_FAILURE_THRESHOLD
}

fn default_cooldown_ms() -> u64 {
    DEFAULT_COOLDOWN_MS
}

fn default_max_cooldown_ms() -> u64 {
    DEFAULT_MAX_COOLDOWN_MS
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            cooldown_ms: DEFAULT_COOLDOWN_MS,
            max_cooldown_ms: DEFAULT_MAX_COOLDOWN_MS,
        }
    }
}

/// The breaker's current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// What [`CircuitBreaker::admit`] decided for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// The breaker is closed: call the provider normally.
    Closed,
    /// The cooldown elapsed: this request is the single half-open probe.
    Probe,
    /// The breaker is open, or a probe is already in flight: refuse immediately.
    Refused,
}

/// What a request outcome means to the breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitHealth {
    /// The provider served the request (`Outcome::Ok`). Resets the streak.
    Success,
    /// The provider answered, but rejected the request (`Outcome::Rejected` /
    /// `Outcome::ContentFiltered`). It proves reachability, so it resets the
    /// streak without counting as a failure.
    Answered,
    /// A provider-side failure that counts toward opening the breaker.
    Failure,
}

/// A state change worth logging once. Returned by [`CircuitBreaker::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// The breaker just opened (or reopened after a failed probe). Carries the
    /// failure count and the cooldown it will hold for.
    Opened { failures: u32, cooldown_ms: u64 },
    /// The breaker just closed after a successful probe.
    Closed,
}

/// A safety-net cooldown used when the configured cooldown cannot be added to `now`.
const FAR_FUTURE_COOLDOWN: Duration = Duration::from_secs(365 * 24 * 60 * 60);

/// The breaker state machine. Not internally synchronized; the caller owns the
/// locking.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    consecutive_failures: u32,
    /// Current cooldown, doubled on each failed probe and reset on close.
    cooldown: Duration,
    /// When the open breaker may admit a probe. `Some` exactly while open.
    open_until: Option<Instant>,
    /// Incremented on every probe admission, so a probe's owner can hand back exactly its own
    /// probe and never a later one.
    probe_generation: u64,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            cooldown: Duration::from_millis(config.cooldown_ms),
            open_until: None,
            probe_generation: 0,
        }
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// True while the breaker is open or half-open, i.e. the value of the
    /// `proxy_circuit_open` gauge.
    pub fn is_open(&self) -> bool {
        self.state != CircuitState::Closed
    }

    /// Decide whether this request may call the provider.
    ///
    /// While open, the first call after `now` reaches the cooldown transitions
    /// to half-open and returns [`Admission::Probe`]; every other call returns
    /// [`Admission::Refused`].
    pub fn admit(&mut self, now: Instant) -> Admission {
        match self.state {
            CircuitState::Closed => Admission::Closed,
            CircuitState::Open => {
                if self.open_until.is_some_and(|until| now >= until) {
                    self.state = CircuitState::HalfOpen;
                    self.open_until = None;
                    self.probe_generation = self.probe_generation.wrapping_add(1);
                    Admission::Probe
                } else {
                    Admission::Refused
                }
            }
            // Exactly one probe at a time; its owner hands it back if it ends without a
            // verdict (see `release_probe`).
            CircuitState::HalfOpen => Admission::Refused,
        }
    }

    /// Generation of the most recently admitted probe. The owner of a [`Admission::Probe`]
    /// reads it right after `admit` and passes it to [`Self::release_probe`].
    pub fn probe_generation(&self) -> u64 {
        self.probe_generation
    }

    /// Give back probe `generation` if it is still the outstanding probe, because it ended
    /// without a provider verdict (cancelled, refused by the proxy before calling the provider,
    /// or its stream dropped). The next request then probes instead of the breaker staying
    /// half-open. A no-op once the probe's verdict was recorded or a later probe was admitted.
    pub fn release_probe(&mut self, now: Instant, generation: u64) {
        if self.state == CircuitState::HalfOpen && generation == self.probe_generation {
            self.state = CircuitState::Open;
            self.open_until = Some(now);
        }
    }

    /// Fold one request's outcome back into the breaker. Returns the transition
    /// to log, if this outcome changed the state.
    ///
    /// `admission` must be the value [`Self::admit`] returned for this request.
    /// A `Refused` admission is a no-op; a stale `Closed` admission recorded
    /// after the breaker already opened is ignored so an in-flight request
    /// cannot reopen a half-open breaker.
    pub fn record(
        &mut self,
        now: Instant,
        admission: Admission,
        health: CircuitHealth,
    ) -> Option<Transition> {
        match admission {
            Admission::Refused => None,
            Admission::Probe => {
                if self.state != CircuitState::HalfOpen {
                    return None;
                }
                match health {
                    CircuitHealth::Failure => {
                        self.reopen(now);
                        Some(Transition::Opened {
                            failures: self.consecutive_failures,
                            cooldown_ms: self.cooldown.as_millis() as u64,
                        })
                    }
                    CircuitHealth::Success | CircuitHealth::Answered => {
                        self.close();
                        Some(Transition::Closed)
                    }
                }
            }
            Admission::Closed => {
                if self.state != CircuitState::Closed {
                    return None;
                }
                match health {
                    CircuitHealth::Failure => {
                        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                        if self.consecutive_failures >= self.config.failure_threshold {
                            self.cooldown = Duration::from_millis(self.config.cooldown_ms);
                            self.open(now);
                            Some(Transition::Opened {
                                failures: self.consecutive_failures,
                                cooldown_ms: self.cooldown.as_millis() as u64,
                            })
                        } else {
                            None
                        }
                    }
                    CircuitHealth::Success | CircuitHealth::Answered => {
                        self.consecutive_failures = 0;
                        self.cooldown = Duration::from_millis(self.config.cooldown_ms);
                        None
                    }
                }
            }
        }
    }

    /// Enter the open state with the current cooldown.
    fn open(&mut self, now: Instant) {
        self.state = CircuitState::Open;
        // A configured cooldown can be enormous and `Instant + Duration` panics on overflow.
        // Validation bounds the configured values, but the state machine must stay panic-free
        // even when constructed directly, so saturate to a far-future deadline instead.
        self.open_until = Some(
            now.checked_add(self.cooldown)
                .or_else(|| now.checked_add(FAR_FUTURE_COOLDOWN))
                .unwrap_or(now),
        );
    }

    /// A failed probe: double the cooldown up to the cap, then open again.
    fn reopen(&mut self, now: Instant) {
        let doubled = self
            .cooldown
            .checked_mul(2)
            .unwrap_or(Duration::from_millis(u64::MAX));
        self.cooldown = doubled.min(Duration::from_millis(self.config.max_cooldown_ms));
        self.open(now);
    }

    /// A successful probe: close and reset the streak and the backoff.
    fn close(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.cooldown = Duration::from_millis(self.config.cooldown_ms);
        self.open_until = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(
        failure_threshold: u32,
        cooldown_ms: u64,
        max_cooldown_ms: u64,
    ) -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold,
            cooldown_ms,
            max_cooldown_ms,
        }
    }

    /// Record a failure under the admission the breaker would have returned.
    fn fail(breaker: &mut CircuitBreaker, now: Instant) -> Option<Transition> {
        let admission = breaker.admit(now);
        breaker.record(now, admission, CircuitHealth::Failure)
    }

    /// Open the breaker and return the time its first probe is due.
    fn opened(breaker: &mut CircuitBreaker, now: Instant) -> Instant {
        for _ in 0..breaker.config.failure_threshold {
            fail(breaker, now);
        }
        assert_eq!(breaker.state(), CircuitState::Open);
        now + Duration::from_millis(breaker.config.cooldown_ms)
    }

    #[test]
    fn a_released_probe_lets_the_next_request_probe() {
        let t0 = Instant::now();
        let mut breaker = CircuitBreaker::new(config(2, 1_000, 8_000));
        let due = opened(&mut breaker, t0);
        assert_eq!(breaker.admit(due), Admission::Probe);
        let generation = breaker.probe_generation();
        // Exactly one probe, however long it takes.
        assert_eq!(
            breaker.admit(due + Duration::from_secs(3_600)),
            Admission::Refused
        );
        breaker.release_probe(due, generation);
        assert_eq!(breaker.state(), CircuitState::Open);
        assert_eq!(breaker.admit(due), Admission::Probe);
    }

    #[test]
    fn a_stale_release_cannot_free_a_newer_probe() {
        let t0 = Instant::now();
        let mut breaker = CircuitBreaker::new(config(2, 1_000, 8_000));
        let due = opened(&mut breaker, t0);
        let first = breaker.admit(due);
        let first_generation = breaker.probe_generation();
        // The first probe fails, the breaker reopens, and a second probe is admitted later.
        breaker.record(due, first, CircuitHealth::Failure);
        let second_due = due + Duration::from_millis(2_000);
        assert_eq!(breaker.admit(second_due), Admission::Probe);
        // The first probe's owner drops late: it must not release the second probe.
        breaker.release_probe(second_due, first_generation);
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        assert_eq!(breaker.admit(second_due), Admission::Refused);
    }

    #[test]
    fn release_after_a_verdict_is_a_no_op() {
        let t0 = Instant::now();
        let mut breaker = CircuitBreaker::new(config(2, 1_000, 8_000));
        let due = opened(&mut breaker, t0);
        let probe = breaker.admit(due);
        let generation = breaker.probe_generation();
        breaker.record(due, probe, CircuitHealth::Success);
        breaker.release_probe(due, generation);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn defaults_are_the_documented_values() {
        let default = CircuitBreakerConfig::default();
        assert_eq!(default.failure_threshold, 5);
        assert_eq!(default.cooldown_ms, 30_000);
        assert_eq!(default.max_cooldown_ms, 300_000);
    }

    #[test]
    fn opens_after_the_threshold_and_admits_nothing_until_the_cooldown() {
        let start = Instant::now();
        let mut breaker = CircuitBreaker::new(config(3, 1_000, 10_000));
        assert_eq!(breaker.admit(start), Admission::Closed);
        assert_eq!(fail(&mut breaker, start), None);
        assert_eq!(fail(&mut breaker, start), None);
        assert_eq!(breaker.state(), CircuitState::Closed);

        let opened = fail(&mut breaker, start).expect("third failure opens");
        assert_eq!(
            opened,
            Transition::Opened {
                failures: 3,
                cooldown_ms: 1_000
            }
        );
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(breaker.is_open());

        // Still inside the cooldown: refused.
        assert_eq!(
            breaker.admit(start + Duration::from_millis(999)),
            Admission::Refused
        );
        // At the deadline the first call is the probe and concurrent ones are refused.
        assert_eq!(
            breaker.admit(start + Duration::from_millis(1_000)),
            Admission::Probe
        );
        assert_eq!(
            breaker.admit(start + Duration::from_millis(1_001)),
            Admission::Refused
        );
    }

    #[test]
    fn success_resets_the_consecutive_streak() {
        let start = Instant::now();
        let mut breaker = CircuitBreaker::new(config(3, 1_000, 10_000));
        fail(&mut breaker, start);
        fail(&mut breaker, start);
        assert_eq!(
            breaker.record(start, Admission::Closed, CircuitHealth::Success),
            None
        );
        // The streak is back to zero: two more failures do not trip.
        assert_eq!(fail(&mut breaker, start), None);
        assert_eq!(fail(&mut breaker, start), None);
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(
            fail(&mut breaker, start),
            Some(Transition::Opened {
                failures: 3,
                cooldown_ms: 1_000
            })
        );
    }

    #[test]
    fn a_provider_answer_resets_the_streak_without_counting() {
        let start = Instant::now();
        let mut breaker = CircuitBreaker::new(config(3, 1_000, 10_000));
        fail(&mut breaker, start);
        fail(&mut breaker, start);
        // Rejected / ContentFiltered prove reachability.
        assert_eq!(
            breaker.record(start, Admission::Closed, CircuitHealth::Answered),
            None
        );
        assert_eq!(fail(&mut breaker, start), None);
        assert_eq!(fail(&mut breaker, start), None);
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(
            fail(&mut breaker, start),
            Some(Transition::Opened {
                failures: 3,
                cooldown_ms: 1_000
            })
        );
    }

    #[test]
    fn a_successful_probe_closes_and_a_failed_probe_doubles_the_cooldown() {
        let start = Instant::now();
        let mut breaker = CircuitBreaker::new(config(1, 1_000, 8_000));
        fail(&mut breaker, start);

        // First probe fails: 1000 -> 2000.
        let probe_at = start + Duration::from_millis(1_000);
        assert_eq!(breaker.admit(probe_at), Admission::Probe);
        assert_eq!(
            breaker.record(probe_at, Admission::Probe, CircuitHealth::Failure),
            Some(Transition::Opened {
                failures: 1,
                cooldown_ms: 2_000
            })
        );
        assert_eq!(
            breaker.admit(probe_at + Duration::from_millis(1_999)),
            Admission::Refused
        );

        // Second probe fails: 2000 -> 4000.
        let probe_at = probe_at + Duration::from_millis(2_000);
        assert_eq!(breaker.admit(probe_at), Admission::Probe);
        assert_eq!(
            breaker.record(probe_at, Admission::Probe, CircuitHealth::Failure),
            Some(Transition::Opened {
                failures: 1,
                cooldown_ms: 4_000
            })
        );

        // Third probe succeeds: close, and the backoff resets to the base cooldown.
        let probe_at = probe_at + Duration::from_millis(4_000);
        assert_eq!(breaker.admit(probe_at), Admission::Probe);
        assert_eq!(
            breaker.record(probe_at, Admission::Probe, CircuitHealth::Success),
            Some(Transition::Closed)
        );
        assert_eq!(breaker.state(), CircuitState::Closed);

        assert_eq!(
            fail(&mut breaker, probe_at),
            Some(Transition::Opened {
                failures: 1,
                cooldown_ms: 1_000
            })
        );
        assert_eq!(
            breaker.admit(probe_at + Duration::from_millis(1_000)),
            Admission::Probe
        );
    }

    #[test]
    fn the_cooldown_doubles_up_to_the_cap() {
        let start = Instant::now();
        let mut breaker = CircuitBreaker::new(config(1, 1_000, 2_500));
        fail(&mut breaker, start);

        let mut at = start + Duration::from_millis(1_000);
        assert_eq!(breaker.admit(at), Admission::Probe);
        assert_eq!(
            breaker.record(at, Admission::Probe, CircuitHealth::Failure),
            Some(Transition::Opened {
                failures: 1,
                cooldown_ms: 2_000
            })
        );
        at += Duration::from_millis(2_000);
        assert_eq!(breaker.admit(at), Admission::Probe);
        // 2000 * 2 = 4000, capped at 2500.
        assert_eq!(
            breaker.record(at, Admission::Probe, CircuitHealth::Failure),
            Some(Transition::Opened {
                failures: 1,
                cooldown_ms: 2_500
            })
        );
        at += Duration::from_millis(2_500);
        assert_eq!(breaker.admit(at), Admission::Probe);
    }

    #[test]
    fn a_huge_cooldown_saturates_instead_of_panicking() {
        let start = Instant::now();
        // `u64::MAX` milliseconds overflows `Instant + Duration`; `open` must saturate rather
        // than panic. Validation bounds the configured value, but the state machine is also
        // constructed directly.
        let mut breaker = CircuitBreaker::new(config(1, u64::MAX, u64::MAX));
        fail(&mut breaker, start);
        assert_eq!(breaker.state(), CircuitState::Open);
        // The saturated deadline is in the future, so this is not immediately a probe.
        assert_eq!(breaker.admit(start), Admission::Refused);
    }

    #[test]
    fn a_stale_closed_outcome_cannot_reopen_a_half_open_breaker() {
        let start = Instant::now();
        let mut breaker = CircuitBreaker::new(config(1, 1_000, 10_000));
        fail(&mut breaker, start);
        let probe_at = start + Duration::from_millis(1_000);
        assert_eq!(breaker.admit(probe_at), Admission::Probe);

        // A request admitted while closed fails after the breaker opened.
        assert_eq!(
            breaker.record(probe_at, Admission::Closed, CircuitHealth::Failure),
            None
        );
        assert_eq!(
            breaker.state(),
            CircuitState::HalfOpen,
            "the probe is still outstanding"
        );
    }
}
