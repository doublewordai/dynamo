#!/usr/bin/env python3
"""Report for the Level 2 e2e spillover simulation.

Reads the load generator's JSONL records (one line per turn) plus the fake
providers' request logs and prints a markdown report: per-worker and per-tier
traffic shares over time, stickiness, and errors. Optionally compares the
result against a Level 1 ``routing-sim`` JSON report within tolerances.

Standard library only; see ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

_DEFAULT_TIERS = [
    {"name": "proxy-x", "ranks": [1000, 1999]},
    {"name": "proxy-y", "ranks": [2000, 2999]},
]

# Smallest absolute difference always accepted by `compare`. Shares of a few
# percent are noisy in a short run, and a purely relative tolerance would judge
# them on a near-zero reference and fail on a fraction of a percentage point.
_ABS_TOLERANCE = 0.02


def classify(dp_rank: object, tiers: list[dict]) -> str:
    """Map a DP rank to its tier, or 'hosted'/'unknown'."""
    if dp_rank is None:
        return "unknown"
    try:
        rank = int(dp_rank)
    except (TypeError, ValueError):
        return "unknown"
    for tier in tiers:
        low, high = tier["ranks"]
        if low <= rank <= high:
            return tier["name"]
    return "hosted"


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def annotate(records: list[dict], tiers: list[dict]) -> None:
    """Add class and previous-worker fields used by share and stickiness math."""
    by_session: dict[object, list[dict]] = defaultdict(list)
    for record in records:
        record["class"] = classify(record.get("decode_dp_rank"), tiers)
        record["failed"] = record.get("status") != 200 or bool(record.get("error"))
        by_session[record.get("session")].append(record)
    for session_records in by_session.values():
        session_records.sort(key=lambda r: (r.get("turn") or 0, r.get("start_ts") or 0))
        previous = None
        for record in session_records:
            record["previous_worker_id"] = previous
            if isinstance(record.get("worker_id"), int):
                previous = record["worker_id"]


def class_stats(records: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    total = len(records)
    names = sorted({record["class"] for record in records})
    for name in names:
        subset = [record for record in records if record["class"] == name]
        ok = [record for record in subset if not record["failed"]]
        stats[name] = {
            "requests": len(subset),
            "share": round(len(subset) / total, 4) if total else 0.0,
            "failed": sum(1 for record in subset if record["failed"]),
            "p50_latency_ms": percentile([r["latency_ms"] for r in ok], 50),
            "p95_latency_ms": percentile([r["latency_ms"] for r in ok], 95),
            "p50_ttft_ms": percentile(
                [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None], 50
            ),
        }
    return stats


def stickiness(records: list[dict], subset: list[dict] | None = None) -> dict:
    subset = records if subset is None else subset
    qualifying = 0
    stayed = 0
    for record in subset:
        previous = record.get("previous_worker_id")
        current = record.get("worker_id")
        if previous is None or current is None:
            continue
        qualifying += 1
        if previous == current:
            stayed += 1
    return {
        "follow_ups": qualifying,
        "stayed": stayed,
        "rate": round(stayed / qualifying, 4) if qualifying else None,
    }


def class_stickiness(records: list[dict]) -> dict:
    """Follow-ups that stayed in the previous turn's class (hosted / proxy tier).

    Level 1 reports class stickiness, so this is the metric the e2e report
    compares against; worker stickiness stays stricter (same DP rank).
    """
    by_session: dict[object, list[dict]] = defaultdict(list)
    for record in records:
        by_session[record.get("session")].append(record)
    qualifying = 0
    stayed = 0
    for session_records in by_session.values():
        session_records.sort(key=lambda r: (r.get("turn") or 0, r.get("start_ts") or 0))
        previous = None
        for record in session_records:
            current = record.get("class")
            if previous is not None and current is not None:
                qualifying += 1
                if previous == current:
                    stayed += 1
            previous = current
    return {
        "follow_ups": qualifying,
        "stayed": stayed,
        "rate": round(stayed / qualifying, 4) if qualifying else None,
    }


def windows(records: list[dict], bin_seconds: float) -> list[dict]:
    if not records:
        return []
    t0 = min(r.get("start_ts") or 0.0 for r in records)
    bins: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        start = record.get("start_ts") or t0
        bins[int((start - t0) // bin_seconds)].append(record)
    result = []
    for index in sorted(bins):
        subset = bins[index]
        stats = class_stats(subset)
        result.append(
            {
                "start_s": round(index * bin_seconds, 1),
                "end_s": round((index + 1) * bin_seconds, 1),
                "requests": len(subset),
                "failed": sum(1 for record in subset if record["failed"]),
                "classes": {name: stat for name, stat in stats.items()},
                "stickiness": stickiness(records, subset)["rate"],
            }
        )
    return result


def provider_stats(paths: dict[str, str]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for name, path in paths.items():
        records = read_jsonl(path)
        by_status: dict[str, int] = defaultdict(int)
        prompt_chars = 0
        for record in records:
            by_status[str(record.get("status"))] += 1
            prompt_chars += int(record.get("prompt_chars") or 0)
        stats[name] = {
            "requests": len(records),
            "by_status": dict(sorted(by_status.items())),
            "mean_prompt_chars": round(prompt_chars / len(records), 1) if records else 0.0,
        }
    return stats


def build_report(records: list[dict], tiers: list[dict], bin_seconds: float) -> dict:
    annotate(records, tiers)
    overall_stats = class_stats(records)
    return {
        "requests": len(records),
        "failed_requests": sum(1 for record in records if record["failed"]),
        "classes": overall_stats,
        "stickiness": stickiness(records),
        "class_stickiness": class_stickiness(records),
        "windows": windows(records, bin_seconds),
    }


def format_markdown(report: dict, providers: dict[str, dict], comparison: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# e2e spillover report")
    lines.append("")
    lines.append(
        f"Requests: **{report['requests']}** "
        f"({report['failed_requests']} failed)"
    )
    lines.append("")

    lines.append("## Per worker class (overall)")
    lines.append("")
    lines.append("| class | requests | share | failed | p50 ms | p95 ms | p50 ttft ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, stat in sorted(report["classes"].items()):
        share = f"{stat['share'] * 100:.1f}%"
        lines.append(
            f"| {name} | {stat['requests']} | {share} | {stat['failed']} | "
            f"{_fmt(stat['p50_latency_ms'])} | {_fmt(stat['p95_latency_ms'])} | "
            f"{_fmt(stat['p50_ttft_ms'])} |"
        )
    lines.append("")

    lines.append("## Shares over time")
    lines.append("")
    class_names = sorted(report["classes"].keys())
    header = "| t (s) | requests | errors | " + " | ".join(class_names) + " |"
    lines.append(header)
    lines.append("|---:|---:|---:|" + "|".join(["---:"] * len(class_names)) + "|")
    for window in report["windows"]:
        shares = []
        for name in class_names:
            stat = window["classes"].get(name)
            shares.append(f"{stat['share'] * 100:.0f}%" if stat else "0%")
        lines.append(
            f"| {window['start_s']:.0f}-{window['end_s']:.0f} | {window['requests']} | "
            f"{window['failed']} | " + " | ".join(shares) + " |"
        )
    lines.append("")

    sticky = report["stickiness"]
    rate = sticky["rate"]
    rate_text = f"{rate * 100:.1f}%" if rate is not None else "n/a"
    lines.append("## Stickiness")
    lines.append("")
    lines.append(
        f"Follow-up turns served by the previous turn's worker: "
        f"**{sticky['stayed']} / {sticky['follow_ups']}** ({rate_text})."
    )
    class_sticky = report.get("class_stickiness", {})
    class_rate = class_sticky.get("rate")
    class_rate_text = f"{class_rate * 100:.1f}%" if class_rate is not None else "n/a"
    lines.append(
        f"Follow-up turns that stayed in the previous turn's class: "
        f"**{class_sticky.get('stayed', 0)} / {class_sticky.get('follow_ups', 0)}** "
        f"({class_rate_text})."
    )
    lines.append("")

    if providers:
        lines.append("## Fake providers")
        lines.append("")
        lines.append("| provider | requests | by status | mean prompt chars |")
        lines.append("|---|---:|---|---:|")
        for name, stat in sorted(providers.items()):
            statuses = ", ".join(f"{code}:{count}" for code, count in stat["by_status"].items())
            lines.append(
                f"| {name} | {stat['requests']} | {statuses} | "
                f"{stat['mean_prompt_chars']} |"
            )
        lines.append("")

    if comparison:
        lines.append("## Comparison with Level 1")
        lines.append("")
        lines.append("| metric | e2e | level 1 | delta | tolerance | result |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in comparison:
            lines.append(
                f"| {row['metric']} | {_fmt(row['e2e'])} | {_fmt(row['baseline'])} | "
                f"{_fmt(row['delta'])} | {row['tolerance']} | {row['result']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def flatten(obj: object, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        return flat
    else:
        flat[prefix] = obj
    return flat


def lookup(flat: dict[str, object], path: str) -> object | None:
    if path in flat:
        return flat[path]
    # Tolerate schema drift: match on the final two path segments.
    segments = path.split(".")
    suffix = ".".join(segments[-2:])
    for key, value in flat.items():
        if key.endswith("." + suffix) or key == suffix:
            return value
    last = segments[-1]
    for key, value in flat.items():
        if key.split(".")[-1] == last:
            return value
    return None


def normalize_baseline(baseline: dict) -> dict:
    """Map a `routing-sim` JSON report onto the e2e report shape.

    `routing-sim` aggregates per class in `overall.by_tier` and reports shares as
    fractions of all requests, so we rebuild the `classes`/`stickiness` keys the
    comparison looks for. Already-normalized e2e reports are returned unchanged.
    """
    if "report" in baseline:
        return baseline
    overall = baseline.get("overall")
    if not isinstance(overall, dict):
        return baseline
    requests = overall.get("requests") or 0
    classes = {
        "hosted": {
            "requests": overall.get("hosted", 0),
            "share": overall.get("hosted_share", 0.0),
        }
    }
    for tier, count in (overall.get("by_tier") or {}).items():
        classes[tier] = {
            "requests": count,
            "share": count / requests if requests else 0.0,
        }
    return {
        "classes": classes,
        "class_stickiness": {
            "rate": overall.get("class_stickiness", overall.get("worker_stickiness", 0.0))
        },
        "failed_requests": overall.get("failures", 0),
    }


def compare(report: dict, baseline: dict, tolerance: float) -> list[dict]:
    base_norm = normalize_baseline(baseline)
    e2e = flatten(report)
    base = flatten(base_norm)
    baseline_classes = base_norm.get("classes", {})
    metrics: list[str] = ["class_stickiness.rate", "failed_requests"]
    for name in report["classes"]:
        metrics.append(f"classes.{name}.share")
    rows = []
    for metric in metrics:
        e2e_value = lookup(e2e, metric)
        if metric.startswith("classes.") and metric.endswith(".share"):
            name = metric.split(".")[1]
            # A tier absent from the Level 1 report simply served no requests.
            base_value = baseline_classes.get(name, {}).get("share", 0.0)
        else:
            base_value = lookup(base, metric)
        if e2e_value is None or base_value is None:
            rows.append(
                {
                    "metric": metric,
                    "e2e": e2e_value,
                    "baseline": base_value,
                    "delta": None,
                    "tolerance": tolerance,
                    "result": "n/a",
                }
            )
            continue
        try:
            delta = float(e2e_value) - float(base_value)
        except (TypeError, ValueError):
            rows.append(
                {
                    "metric": metric,
                    "e2e": e2e_value,
                    "baseline": base_value,
                    "delta": None,
                    "tolerance": tolerance,
                    "result": "n/a",
                }
            )
            continue
        scale = abs(float(base_value)) if float(base_value) != 0 else 1.0
        band = max(tolerance * scale, _ABS_TOLERANCE)
        result = "pass" if abs(delta) <= band else "FAIL"
        rows.append(
            {
                "metric": metric,
                "e2e": e2e_value,
                "baseline": base_value,
                "delta": round(delta, 4),
                "tolerance": tolerance,
                "result": result,
            }
        )
    return rows


def parse_provider_logs(values: list[str]) -> dict[str, str]:
    logs = {}
    for value in values:
        name, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--provider-log expects NAME=PATH, got {value!r}")
        logs[name] = path
    return logs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loadgen",
        action="append",
        default=[],
        help="JSONL file from loadgen.py (repeatable)",
    )
    parser.add_argument(
        "--provider-log",
        action="append",
        default=[],
        help="fake provider JSONL as NAME=PATH (repeatable)",
    )
    parser.add_argument("--tier-map", default=None, help="JSON [{\"name\",\"ranks\":[lo,hi]}]")
    parser.add_argument("--bin-seconds", type=float, default=10.0)
    parser.add_argument("--baseline", default=None, help="Level 1 routing-sim JSON report")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="relative tolerance for baseline comparison",
    )
    parser.add_argument("--json", default=None, help="write the normalized JSON report")
    parser.add_argument("--markdown", default=None, help="write markdown (default stdout)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.loadgen:
        raise SystemExit("at least one --loadgen file is required")

    if args.tier_map:
        with open(args.tier_map, encoding="utf-8") as handle:
            tiers = json.load(handle)
    else:
        tiers = _DEFAULT_TIERS

    records: list[dict] = []
    for path in args.loadgen:
        records.extend(read_jsonl(path))
    records.sort(key=lambda r: r.get("start_ts") or 0.0)

    report = build_report(records, tiers, args.bin_seconds)
    providers = provider_stats(parse_provider_logs(args.provider_log))

    comparison: list[dict] = []
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        comparison = compare(report, baseline, args.tolerance)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"report": report, "providers": providers, "comparison": comparison}, handle, indent=2)
            handle.write("\n")

    markdown = format_markdown(report, providers, comparison)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    else:
        sys.stdout.write(markdown)
    return 1 if any(row["result"] == "FAIL" for row in comparison) else 0


if __name__ == "__main__":
    raise SystemExit(main())
