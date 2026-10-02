#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the proxy metrics scraped during an e2e run.

``run.sh`` appends a snapshot of each proxy's Prometheus endpoint to
``reports/metrics-<tier>.prom`` every few seconds. This reads the file, takes
the last complete snapshot, and checks the proxy metrics surface is live and
labelled by tier/provider: requests, tokens, TTFT and the virtual cache.

Exits non-zero if a proxy never reported ``dynamo_component_proxy_requests_total``
or reported zero requests. A tier listed as optional with ``--required-tier``
(or all tiers with ``--no-required-tiers``) may legitimately serve nothing; its
metrics surface is still checked, but zero requests are not a failure.
Standard library only; see ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
import re

# Metric families the proxy registers; see
# lib/spillover/proxy-worker/src/metrics.rs. A histogram is present as its
# `_count`/`_sum` series, and the virtual-cache gauge must be exposed even
# while it is zero.
_REQUIRED = (
    "dynamo_component_proxy_requests_total",
    "dynamo_component_proxy_completion_tokens_total",
    "dynamo_component_proxy_time_to_first_token_seconds_count",
    "dynamo_component_proxy_time_to_first_token_seconds_sum",
)
# Exposed from startup, so required even for a tier that served nothing.
_REQUIRED_AT_START = "dynamo_component_proxy_virtual_cache_blocks"

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


def _unescape_label(value: str) -> str:
    """Unescape a Prometheus label value without mangling non-ASCII text.

    ``bytes.decode("unicode_escape")`` reinterprets UTF-8 as Latin-1 and
    corrupts labels like ``tier="模型"``. Prometheus only escapes ``\\``,
    ``"`` and ``\n``.
    """
    out: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            escaped = value[index + 1]
            out.append({"n": "\n", "\\": "\\", '"': '"'}.get(escaped, escaped))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


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
        labels[match.group(1)] = _unescape_label(match.group(2))
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


def check_one(provider: str, tier: str, path: str, *, required: bool = True) -> dict:
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
    if metric_total(rows, _REQUIRED_AT_START) is None:
        problems.append(f"missing {_REQUIRED_AT_START}")
    if required:
        for name in _REQUIRED:
            if metric_total(rows, name) is None:
                problems.append(f"missing {name}")
        if requests is None or requests <= 0:
            problems.append("no proxy requests recorded")
        # `tier` and `provider` are independent config fields; check each
        # against its own set rather than assuming the deployment names them
        # identically. An unrequired tier that served nothing has no request
        # series to carry these labels.
        if provider not in providers:
            problems.append(f"no provider={provider!r} label on proxy_requests_total")
        if tier not in tiers:
            problems.append(f"no tier={tier!r} label on proxy_requests_total")
    return {
        "tier": tier,
        "provider": provider,
        "required": required,
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
    parser.add_argument(
        "--providers",
        action="append",
        default=[],
        help="expected provider name as TIER=PROVIDER (repeatable)",
    )
    parser.add_argument(
        "--required-tier",
        action="append",
        default=[],
        help="tier that must have served requests (repeatable); default: every tier",
    )
    parser.add_argument(
        "--no-required-tiers",
        action="store_true",
        help="no tier must have served requests (metrics surface only)",
    )
    parser.add_argument("--out", default=None, help="write the JSON result here")
    return parser.parse_args(argv)


def parse_providers(values: list[str]) -> dict[str, str]:
    providers: dict[str, str] = {}
    for value in values:
        tier, _, provider = value.partition("=")
        if not provider:
            raise SystemExit(f"--providers expects TIER=PROVIDER, got {value!r}")
        providers[tier] = provider
    return providers


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.metrics:
        raise SystemExit("at least one --metrics TIER=PATH is required")

    providers = parse_providers(args.providers)
    if args.no_required_tiers:
        required_tiers: set[str] | None = set()
    elif args.required_tier:
        required_tiers = set(args.required_tier)
    else:
        required_tiers = None
    results = {}
    for value in args.metrics:
        tier, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--metrics expects TIER=PATH, got {value!r}")
        required = required_tiers is None or tier in required_tiers
        results[tier] = check_one(
            providers.get(tier, tier), tier, path, required=required
        )

    payload = {"proxies": results, "ok": all(r["ok"] for r in results.values())}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
