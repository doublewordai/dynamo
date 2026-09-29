#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the proxy metrics scraped during an e2e run.

``run.sh`` appends a snapshot of each proxy's Prometheus endpoint to
``reports/metrics-<tier>.prom`` every few seconds. This reads the file, takes
the last complete snapshot, and checks the proxy metrics surface is live and
labelled by tier/provider: requests, tokens, TTFT and the virtual cache.

Exits non-zero if a proxy never reported ``dynamo_component_proxy_requests_total``
or reported zero requests. Standard library only; see ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
import re

# Metric families the proxy registers; see
# lib/spillover/proxy-worker/src/metrics.rs.
_REQUIRED = (
    "dynamo_component_proxy_requests_total",
    "dynamo_component_proxy_completion_tokens_total",
)

# A Prometheus text sample: name{labels} value [timestamp].
_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>\S+)"
)


def last_snapshot(text: str) -> str:
    """Return the text after the final ``# scrape`` marker.

    A scrape failure writes ``# scrape failed`` and no samples, so the last
    *successful* snapshot is the one to read.
    """
    marker = "# scrape "
    positions = [m.start() for m in re.finditer(re.escape(marker), text)]
    for start in reversed(positions):
        end = text.find(marker, start + len(marker))
        chunk = text[start : end if end != -1 else len(text)]
        if "# scrape failed" not in chunk and _sample_count(chunk):
            return chunk
    return ""


def _sample_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _SAMPLE.match(line))


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        raw = raw[1:]
    if raw.endswith("}"):
        raw = raw[:-1]
    labels: dict[str, str] = {}
    for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', raw):
        labels[match.group(1)] = match.group(2).encode().decode("unicode_escape")
    return labels


def samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    out: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        out.append((match.group("name"), parse_labels(match.group("labels")), value))
    return out


def metric_total(
    rows: list[tuple[str, dict[str, str], float]], name: str
) -> float | None:
    values = [value for metric, _, value in rows if metric == name]
    if not values:
        return None
    # Counters are scraped cumulatively; the last snapshot's single sample (or
    # the sum of per-provider samples) is the run total.
    return sum(values)


def check_one(tier: str, path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    snapshot = last_snapshot(text)
    rows = samples(snapshot)
    requests = metric_total(rows, "dynamo_component_proxy_requests_total")
    completions = metric_total(rows, "dynamo_component_proxy_completion_tokens_total")
    providers = {
        labels.get("provider")
        for metric, labels, _ in rows
        if metric == "dynamo_component_proxy_requests_total" and labels.get("provider")
    }
    tiers = {
        labels.get("tier")
        for metric, labels, _ in rows
        if metric == "dynamo_component_proxy_requests_total" and labels.get("tier")
    }
    problems: list[str] = []
    for name in _REQUIRED:
        if metric_total(rows, name) is None:
            problems.append(f"missing {name}")
    if requests is None or requests <= 0:
        problems.append("no proxy requests recorded")
    if tier not in providers:
        problems.append(f"no provider={tier!r} label on proxy_requests_total")
    if tier not in tiers:
        problems.append(f"no tier={tier!r} label on proxy_requests_total")
    return {
        "tier": tier,
        "snapshot_samples": len(rows),
        "requests": requests,
        "completion_tokens": completions,
        "providers": sorted(p for p in providers if p),
        "tiers": sorted(t for t in tiers if t),
        "ok": not problems,
        "problems": problems,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="scraped Prometheus text as TIER=PATH (repeatable)",
    )
    parser.add_argument("--out", default=None, help="write the JSON result here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.metrics:
        raise SystemExit("at least one --metrics TIER=PATH is required")

    results = {}
    for value in args.metrics:
        tier, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--metrics expects TIER=PATH, got {value!r}")
        results[tier] = check_one(tier, path)

    payload = {"proxies": results, "ok": all(r["ok"] for r in results.values())}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
