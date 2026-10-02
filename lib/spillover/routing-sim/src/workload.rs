// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Workload generation: sessions of multi-turn conversations arriving over time.
//!
//! Everything is generated from `seed` before the event loop starts, so the same scenario always
//! produces the same arrivals and turns. Each turn's prompt extends the previous turn's prompt
//! and output, and every session shares the same system prompt.

use fastrand::Rng;

use crate::config::{FloatDist, IntDist, RatePoint, Scenario, WorkloadConfig};
use crate::hash::synth_tokens;

/// One turn's user input and requested output length.
#[derive(Debug, Clone)]
pub struct TurnSpec {
    pub user_tokens: Vec<u32>,
    pub output_tokens: usize,
    /// Delay after the previous turn before this turn starts.
    pub think_time: f64,
}

/// A session of one or more turns starting at `start_time`.
#[derive(Debug, Clone)]
pub struct SessionSpec {
    pub start_time: f64,
    pub turns: Vec<TurnSpec>,
}

/// The fully generated workload for a scenario.
#[derive(Debug, Clone)]
pub struct Workload {
    pub system_tokens: Vec<u32>,
    pub sessions: Vec<SessionSpec>,
}

impl IntDist {
    fn sample(&self, rng: &mut Rng) -> usize {
        if self.max <= self.min {
            self.min
        } else {
            rng.usize(self.min..=self.max)
        }
    }
}

impl FloatDist {
    fn sample(&self, rng: &mut Rng) -> f64 {
        if self.max <= self.min {
            self.min
        } else {
            self.min + rng.f64() * (self.max - self.min)
        }
    }
}

impl Workload {
    pub fn generate(scenario: &Scenario) -> Self {
        let mut rng = Rng::with_seed(scenario.seed);
        let system_tokens = synth_tokens("system", scenario.workload.system_prompt_tokens);
        let starts = arrival_times(&scenario.arrival_rate, scenario.duration_seconds, &mut rng);
        let sessions = starts
            .into_iter()
            .enumerate()
            .map(|(index, start_time)| {
                generate_session(index, start_time, &scenario.workload, &mut rng)
            })
            .collect();
        Self {
            system_tokens,
            sessions,
        }
    }
}

fn generate_session(
    index: usize,
    start_time: f64,
    workload: &WorkloadConfig,
    rng: &mut Rng,
) -> SessionSpec {
    let turns = workload.turns_per_session.sample(rng).max(1);
    let mut specs = Vec::with_capacity(turns);
    for turn in 0..turns {
        let user_count = workload.user_tokens.sample(rng);
        let output_tokens = workload.output_tokens.sample(rng);
        let think_time = if turn == 0 {
            0.0
        } else {
            workload.think_time_seconds.sample(rng)
        };
        specs.push(TurnSpec {
            user_tokens: synth_tokens(&format!("user-{index}-{turn}"), user_count),
            output_tokens,
            think_time,
        });
    }
    SessionSpec {
        start_time,
        turns: specs,
    }
}

/// Session start times over `[0, duration]`, sampled by thinning a piecewise-linear rate profile.
fn arrival_times(profile: &[RatePoint], duration: f64, rng: &mut Rng) -> Vec<f64> {
    if profile.is_empty() || duration <= 0.0 {
        return Vec::new();
    }
    let max_rate = profile.iter().map(|p| p.rate).fold(0.0_f64, f64::max);
    if max_rate <= 0.0 {
        return Vec::new();
    }
    let mut times = Vec::new();
    let mut time = 0.0;
    loop {
        // Exponential inter-arrival at the maximum rate, then thin by the actual rate.
        let uniform = rng.f64().max(f64::MIN_POSITIVE);
        time += -uniform.ln() / max_rate;
        if time > duration {
            break;
        }
        let rate = rate_at(profile, time);
        if rng.f64() < rate / max_rate {
            times.push(time);
        }
    }
    times
}

/// Piecewise-linear interpolation of the arrival-rate profile.
///
/// The profile's times must be non-decreasing; `Scenario::validate` rejects a scenario whose
/// `arrival_rate` times are not, so this only ever sees a sorted profile.
pub fn rate_at(profile: &[RatePoint], time: f64) -> f64 {
    let Some(first) = profile.first() else {
        return 0.0;
    };
    if time <= first.time {
        return first.rate;
    }
    for pair in profile.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if time >= a.time && time <= b.time {
            let span = b.time - a.time;
            if span <= 0.0 {
                return b.rate;
            }
            let fraction = (time - a.time) / span;
            return a.rate + fraction * (b.rate - a.rate);
        }
    }
    profile.last().map(|p| p.rate).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrivals_are_deterministic() {
        let mut scenario = crate::scenarios::load_builtin("low_load").unwrap();
        scenario.seed = 7;
        let a = Workload::generate(&scenario);
        let b = Workload::generate(&scenario);
        assert_eq!(a.sessions.len(), b.sessions.len());
        for (x, y) in a.sessions.iter().zip(&b.sessions) {
            assert_eq!(x.start_time, y.start_time);
            assert_eq!(x.turns.len(), y.turns.len());
        }
    }
}
