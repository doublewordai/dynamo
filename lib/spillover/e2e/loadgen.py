#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn conversation load generator for the Level 2 e2e simulation.

Drives an OpenAI-compatible Dynamo frontend with sessions that share a system
prompt and extend the conversation each turn, at an arrival-rate profile with
think time. Requests stream and opt into
``nvext.extra_fields=["worker_id", "engine_data"]`` so each turn records which
worker (and DP rank) served it and, for proxy workers, the served-by tag; the
report turns that into per-tier shares, stickiness and a provider check.

Proxy workers need no help from the generator: they advertise the ``chat_request``
capability, so the frontend's KV router hands them the chat request itself.

Standard library only; see ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from urllib.parse import urlparse

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant in a load simulation. Answer briefly. "
    "Remember the shared context so cache-affinity routing can be observed."
)


def _text_length(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return 0


def _absorb_chunk(chunk: dict, result: dict) -> None:
    """Fold one ``chat.completion.chunk`` into the running result."""
    choices = chunk.get("choices") or []
    if choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            result["content"] += content
            result["first_token"] = result["first_token"] or time.monotonic()
        reasoning = delta.get("reasoning")
        if not reasoning:
            reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            result["reasoning"] += reasoning
            result["first_token"] = result["first_token"] or time.monotonic()
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            result["tool_calls"] += 1
            result["first_token"] = result["first_token"] or time.monotonic()

    usage = chunk.get("usage")
    if isinstance(usage, dict):
        result["usage"] = usage

    nvext = chunk.get("nvext")
    if isinstance(nvext, dict):
        worker_id = nvext.get("worker_id")
        if isinstance(worker_id, dict):
            decode_id = worker_id.get("decode_worker_id")
            prefill_id = worker_id.get("prefill_worker_id")
            result["worker_id"] = decode_id if decode_id is not None else prefill_id
            result["decode_dp_rank"] = worker_id.get("decode_dp_rank")
            result["prefill_dp_rank"] = worker_id.get("prefill_dp_rank")
        engine_data = nvext.get("engine_data")
        if isinstance(engine_data, dict):
            # The proxy stamps {served_by, tier}; the mocker does not stamp one,
            # so this stays None for hosted turns.
            result["served_by"] = engine_data.get("served_by")
            result["engine_tier"] = engine_data.get("tier")


def stream_chat(
    host: str,
    port: int,
    path: str,
    body: dict,
    timeout: float,
) -> dict:
    """POST a streaming chat request and parse SSE until ``[DONE]``."""
    payload = json.dumps(body).encode()
    result = {
        "status": 0,
        "error": None,
        "content": "",
        "reasoning": "",
        "tool_calls": 0,
        "worker_id": None,
        "decode_dp_rank": None,
        "prefill_dp_rank": None,
        "served_by": None,
        "engine_tier": None,
        "usage": None,
        "first_token": None,
        "bytes": 0,
    }
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        response = connection.getresponse()
        result["status"] = response.status
        if response.status != 200:
            error_body = response.read(4096)
            result["error"] = error_body.decode("utf-8", "replace")
            return result
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text or not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict):
                _absorb_chunk(chunk, result)
                result["bytes"] += len(line)
    except Exception as exc:  # noqa: BLE001 - the run must record any failure
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    return result


def interpolate_rate(
    t: float, profile: list[tuple[float, float]], default: float
) -> float:
    """Piecewise-linear arrival rate (sessions/second) at time ``t``."""
    if not profile:
        return default
    if t <= profile[0][0]:
        return profile[0][1]
    for (t0, r0), (t1, r1) in zip(profile, profile[1:]):
        if t <= t1:
            if t1 == t0:
                return r1
            frac = (t - t0) / (t1 - t0)
            return r0 + frac * (r1 - r0)
    return profile[-1][1]


def schedule_starts(
    sessions: int,
    duration: float,
    profile: list[tuple[float, float]],
    default_rate: float,
    seed: int,
) -> list[float]:
    """Poisson session arrivals until ``sessions`` are placed or ``duration`` ends."""
    rng = random.Random(seed)
    starts: list[float] = []
    t = 0.0
    while len(starts) < sessions:
        if duration > 0 and t >= duration:
            break
        rate = interpolate_rate(t, profile, default_rate)
        if rate <= 0:
            # No arrivals now. Advance in coarse steps but honour the window so
            # a zero-rate tail terminates instead of spinning forever.
            t += 1.0
            continue
        t += rng.expovariate(rate)
        if duration > 0 and t > duration:
            break
        starts.append(t)
    return starts


class Recorder:
    def __init__(self, path: str | None) -> None:
        self._lock = threading.Lock()
        self._file = open(path, "w", encoding="utf-8") if path else None

    def write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def run_session(
    session_id: int,
    start_at: float,
    args: argparse.Namespace,
    recorder: Recorder,
    host: str,
    port: int,
    path: str,
    t0: float,
) -> None:
    delay = start_at - (time.monotonic() - t0)
    if delay > 0:
        time.sleep(delay)

    rng = random.Random(args.seed + session_id)
    conversation: list[dict] = [{"role": "system", "content": args.system_prompt}]
    previous_worker: int | None = None
    for turn in range(args.turns):
        conversation.append(
            {
                "role": "user",
                "content": (
                    f"Session {session_id}, turn {turn}. "
                    "Continue the conversation and be concise."
                ),
            }
        )
        body = {
            "model": args.model,
            "messages": conversation,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "nvext": {"extra_fields": ["worker_id", "engine_data"]},
        }
        arrival_ts = time.time()
        started = time.monotonic()
        result = stream_chat(host, port, path, body, args.timeout)
        finished = time.monotonic()
        first_token = result.get("first_token")
        usage = result.get("usage") or {}

        record = {
            "session": session_id,
            "turn": turn,
            "model": args.model,
            "arrival_ts": arrival_ts,
            "start_ts": arrival_ts,
            "end_ts": time.time(),
            "latency_ms": round((finished - started) * 1000, 3),
            "ttft_ms": (
                round((first_token - started) * 1000, 3) if first_token else None
            ),
            "status": result["status"],
            "error": result["error"],
            "worker_id": result["worker_id"],
            "decode_dp_rank": result["decode_dp_rank"],
            "prefill_dp_rank": result["prefill_dp_rank"],
            "served_by": result["served_by"],
            "engine_tier": result["engine_tier"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "content_chars": len(result["content"]),
            "previous_worker_id": previous_worker,
        }
        recorder.write(record)

        if isinstance(record["worker_id"], int):
            previous_worker = record["worker_id"]
        # Extend the conversation with the assistant turn so the next prompt
        # shares its whole prefix; the upstream provider's cache does the rest.
        conversation.append({"role": "assistant", "content": result["content"] or ""})
        if turn < args.turns - 1:
            jitter = 1.0 + rng.uniform(-args.think_jitter, args.think_jitter)
            time.sleep(max(0.0, args.think_time * jitter))


def load_profile(path: str | None) -> list[tuple[float, float]]:
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    profile = sorted((float(point["t"]), float(point["rate"])) for point in raw)
    return profile


def summarize(records: list[dict], args: argparse.Namespace) -> dict:
    ok = [r for r in records if r["status"] == 200 and not r["error"]]
    failures = len(records) - len(ok)
    per_worker: dict[str, int] = {}
    per_served_by: dict[str, int] = {}
    for record in records:
        key = str(record.get("worker_id"))
        per_worker[key] = per_worker.get(key, 0) + 1
        served_by = record.get("served_by")
        if served_by:
            per_served_by[str(served_by)] = per_served_by.get(str(served_by), 0) + 1

    def percentile(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
        return round(ordered[index], 3)

    latencies = [r["latency_ms"] for r in ok]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    return {
        "model": args.model,
        "requests": len(records),
        "failed_requests": failures,
        "requests_per_worker": per_worker,
        "requests_per_served_by": per_served_by,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p50_ttft_ms": percentile(ttfts, 50),
        "p95_ttft_ms": percentile(ttfts, 95),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000", help="frontend base URL"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--think-time", type=float, default=1.0, help="seconds between turns"
    )
    parser.add_argument(
        "--think-jitter", type=float, default=0.25, help="fraction of think time"
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="sessions per second when no profile is given",
    )
    parser.add_argument("--arrival-profile", default=None, help="JSON [{t,rate}, ...]")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="cap on scheduling window in seconds (0 = until --sessions)",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="per-request seconds"
    )
    parser.add_argument("--out", default=None, help="JSONL per-request records")
    parser.add_argument("--summary", default=None, help="JSON aggregate summary")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parsed = urlparse(args.url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path.rstrip("/") + "/v1/chat/completions"

    profile = load_profile(args.arrival_profile)
    starts = schedule_starts(
        args.sessions, args.duration, profile, args.arrival_rate, args.seed
    )
    if not starts:
        print("loadgen: no sessions scheduled", flush=True)
        return 1

    recorder = Recorder(args.out)
    # The mock provider is static, so we cannot rely on /v1/models; a short
    # settle delay lets the frontend discover workers started just before us.
    t0 = time.monotonic()
    records: list[dict] = []

    try:
        # Size the pool to the number of scheduled sessions so every submitted
        # task begins immediately and sleeps until its own deadline. A smaller
        # pool would make `delay` compute against the time a worker frees up,
        # collapsing later sessions into back-to-back bursts and ignoring the
        # arrival profile the Level 1 twin simulates.
        with ThreadPoolExecutor(max_workers=max(1, len(starts))) as pool:
            futures = []
            for session_id, start_at in enumerate(starts):
                future = pool.submit(
                    run_session,
                    session_id,
                    start_at,
                    args,
                    recorder,
                    host,
                    port,
                    path,
                    t0,
                )
                futures.append(future)
            for future in futures:
                future.result()
    finally:
        recorder.close()

    # Re-read the JSONL so the summary reflects exactly what was written.
    if args.out:
        with open(args.out, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]

    summary = summarize(records, args)
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["failed_requests"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
