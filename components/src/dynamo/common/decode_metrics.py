# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-observed accepted output tokens per active decode-sequence second.

Call observe only from the scheduler, including its real idle path. A transport
heartbeat must never call it. Keep the matching SGLang implementation in sync;
that engine cannot depend on Dynamo's Python package.
"""

import time
from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass


@dataclass
class _Interval:
    start: float
    end: float
    sequences: int
    tokens: int


class DecodeMetricsTracker:
    WINDOW_SECONDS = 1.0
    PUBLISH_SECONDS = 0.25

    def __init__(self) -> None:
        self._decoding: set[str] = set()
        self._intervals: deque[_Interval] = deque()
        self._last_observation: float | None = None
        self._last_publish: float | None = None
        self._revision = 0

    def observe(
        self,
        output_tokens: Mapping[str, int],
        live_request_ids: Collection[str],
        num_running_reqs: int,
        num_waiting_reqs: int,
        *,
        now: float | None = None,
        unix_ms: int | None = None,
    ) -> dict | None:
        """Observe committed output deltas and requests still owned by the engine.

        ``live_request_ids`` includes preempted decode requests. Finished requests
        can appear in output_tokens and must already be absent from the live set.
        Tokens in the first output batch establish decoding but are excluded:
        no decode time has elapsed yet (including first-step speculative output).
        Subsequent deltas count all accepted tokens, including speculative tokens.
        """
        if now is None:
            now = time.monotonic()
        tokens = sum(n for key, n in output_tokens.items() if key in self._decoding)
        if self._last_observation is not None:
            if now < self._last_observation:
                raise ValueError("scheduler observation time must be monotonic")
            if (
                not tokens
                and self._intervals
                and not self._intervals[-1].tokens
                and self._intervals[-1].sequences == len(self._decoding)
            ):
                # Queue stalls can poll much faster than model execution. Coalesce
                # empty intervals without moving any accepted-token timestamps.
                self._intervals[-1].end = now
            else:
                self._intervals.append(
                    _Interval(self._last_observation, now, len(self._decoding), tokens)
                )
        self._last_observation = now
        self._decoding.update(key for key, n in output_tokens.items() if n > 0)
        self._decoding.intersection_update(live_request_ids)
        cutoff = now - self.WINDOW_SECONDS
        while self._intervals and self._intervals[0].end <= cutoff:
            self._intervals.popleft()
        idle = num_running_reqs == 0 and num_waiting_reqs == 0
        if idle:
            # Even a rate-limited idle observation separates independent busy periods.
            self._intervals.clear()
        if (
            self._last_publish is not None
            and now - self._last_publish < self.PUBLISH_SECONDS
        ):
            return None
        self._last_publish = now
        self._revision += 1
        seconds = sum(
            max(0.0, sample.end - max(sample.start, cutoff)) * sample.sequences
            for sample in self._intervals
        )
        rate = (
            sum(sample.tokens for sample in self._intervals) / seconds
            if seconds > 0 and not idle
            else None
        )
        return {
            "tokens_per_user_second": rate,
            "num_running_reqs": num_running_reqs,
            "num_waiting_reqs": num_waiting_reqs,
            "observation_revision": self._revision,
            "observed_at_unix_ms": (
                unix_ms if unix_ms is not None else time.time_ns() // 1_000_000
            ),
        }
