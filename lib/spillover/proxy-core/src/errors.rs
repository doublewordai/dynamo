// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider failures, classified by what the worker should do about them.

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum UpstreamError {
    /// HTTP 429, or a provider rate-limit error.
    #[error("provider rate limited")]
    RateLimited { retry_after_ms: Option<u64> },
    /// HTTP 402, 408, 5xx or 529: the provider account or the provider itself could not
    /// serve this request right now, but another worker/tier may be able to.
    #[error("provider unavailable (status {status})")]
    Unavailable { status: u16 },
    /// Any other 4xx: the request itself was rejected. Retrying elsewhere won't help.
    #[error("provider rejected the request (status {status}): {message}")]
    Rejected { status: u16, message: String },
    /// Could not connect, TLS failure, or connect timeout.
    #[error("transport error: {0}")]
    Transport(String),
    /// The SSE stream ended without `[DONE]` or a finish_reason, or failed to parse.
    #[error("stream broken: {0}")]
    StreamBroken(String),
    /// The provider sent an error object inside the stream (OpenRouter does this).
    #[error("provider error in stream: {0}")]
    InStream(String),
}

impl UpstreamError {
    /// Classify an HTTP error response. `retry_after` is the raw `Retry-After` header.
    pub fn from_status(status: u16, body: &str, retry_after: Option<&str>) -> Self {
        if status == 429 {
            return Self::RateLimited {
                retry_after_ms: retry_after.and_then(parse_retry_after_ms),
            };
        }
        // 402 means the provider account is out of credits, not that the request is
        // bad, so fail over instead of returning a client error.
        if status == 402 || status == 408 || (500..=599).contains(&status) {
            return Self::Unavailable { status };
        }
        Self::Rejected {
            status,
            message: rejection_message(body),
        }
    }

    /// True if another worker could reasonably serve the same request: rate limits,
    /// unavailability, transport failures, broken or errored streams. False for `Rejected`.
    pub fn retry_elsewhere(&self) -> bool {
        !matches!(self, Self::Rejected { .. })
    }
}

/// `Retry-After` is either delta-seconds or an HTTP-date; return milliseconds either way.
fn parse_retry_after_ms(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(seconds.saturating_mul(1000));
    }
    let target_ms = parse_http_date_ms(raw)?;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_millis() as i64;
    Some(target_ms.saturating_sub(now_ms).max(0) as u64)
}

/// The preferred `Retry-After` HTTP-date form, e.g. `Wed, 21 Oct 2015 07:28:00 GMT`.
fn parse_http_date_ms(raw: &str) -> Option<i64> {
    let raw = raw.strip_suffix(" GMT")?;
    let mut parts = raw.split_whitespace();
    let _weekday = parts.next()?;
    let day: i64 = parts.next()?.trim_end_matches(',').parse().ok()?;
    let month = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    // Reject years that could overflow the epoch conversion below. HTTP-date years
    // are four digits; an oversized value is hostile input, not a retry deadline.
    if !(1970..=9999).contains(&year) {
        return None;
    }
    let mut clock = parts.next()?.split(':');
    let hour: i64 = clock.next()?.parse().ok()?;
    let minute: i64 = clock.next()?.parse().ok()?;
    let second: i64 = clock.next()?.parse().ok()?;
    if clock.next().is_some()
        || !(1..=31).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=60).contains(&second)
    {
        return None;
    }
    let days = days_from_civil(year, month, day);
    days.checked_mul(86_400)?
        .checked_add(hour * 3_600 + minute * 60 + second)?
        .checked_mul(1000)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The provider's `error.message` when the body is JSON, else the body capped at 500 chars.
fn rejection_message(body: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(message) = value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(serde_json::Value::as_str)
    {
        return message.chars().take(500).collect();
    }
    body.chars().take(500).collect()
}
