#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Level 2 e2e scripts and run helpers.

Run with ``python3 -m unittest discover -s lib/spillover/e2e -p 'test_*.py'``
(also wired into .github/workflows/spillover.yml). Standard library only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import check_metrics  # noqa: E402
import fake_provider  # noqa: E402
import loadgen  # noqa: E402
import report  # noqa: E402
import run_helpers  # noqa: E402


def _prom(labels: str) -> str:
    return (
        "# scrape 1\n"
        f'dynamo_component_proxy_requests_total{{{labels},outcome="ok"}} 5\n'
        f"dynamo_component_proxy_completion_tokens_total{{{labels}}} 10\n"
        f"dynamo_component_proxy_time_to_first_token_seconds_count{{{labels}}} 5\n"
        f"dynamo_component_proxy_time_to_first_token_seconds_sum{{{labels}}} 0.5\n"
        f"dynamo_component_proxy_virtual_cache_blocks{{{labels}}} 0\n"
    )


class ScheduleStartsTest(unittest.TestCase):
    def test_zero_rate_tail_terminates_and_honours_duration(self) -> None:
        # Regression for a `rate <= 0` loop that advanced one second at a time
        # without ever consulting the scheduling window (r13-5).
        starts = loadgen.schedule_starts(5, 10.0, [(0.0, 0.0)], 0.0, 0)
        self.assertEqual(starts, [])

    def test_zero_rate_tail_main_terminates(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "loadgen.py"),
                "--arrival-rate",
                "0",
                "--duration",
                "10",
                "--sessions",
                "5",
            ],
            capture_output=True,
            timeout=20,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("no sessions scheduled", result.stdout)


class PoolSizingTest(unittest.TestCase):
    def test_pool_sized_to_scheduled_sessions(self) -> None:
        # Regression for batch submission to a fixed 64-worker pool, where late
        # sessions were released as bursts when a worker freed up (r13-1).
        captured: dict[str, int] = {}
        real_executor = loadgen.ThreadPoolExecutor

        class RecordingExecutor(real_executor):
            def __init__(self, max_workers=None, *args, **kwargs):
                captured["max_workers"] = max_workers
                super().__init__(max_workers=max_workers, *args, **kwargs)

        sessions = 80
        starts = [0.0] * sessions
        fake_result = {
            "status": 200,
            "error": None,
            "content": "x",
            "reasoning": "",
            "tool_calls": 0,
            "worker_id": 1,
            "decode_dp_rank": None,
            "prefill_dp_rank": None,
            "served_by": None,
            "engine_tier": None,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "first_token": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "loadgen.jsonl")
            argv = [
                "--url",
                "http://127.0.0.1:1",
                "--model",
                "m",
                "--sessions",
                str(sessions),
                "--turns",
                "1",
                "--think-time",
                "0",
                "--out",
                out,
            ]
            patches = [
                mock.patch.object(loadgen, "schedule_starts", return_value=starts),
                mock.patch.object(loadgen, "ThreadPoolExecutor", RecordingExecutor),
                mock.patch.object(loadgen, "stream_chat", return_value=fake_result),
            ]
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                loadgen.main(argv)
        self.assertEqual(captured["max_workers"], sessions)


class CheckMetricsTest(unittest.TestCase):
    def _write(self, text: str) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".prom", delete=False, encoding="utf-8"
        )
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_requires_ttft_and_vcache(self) -> None:
        # Regression for `_REQUIRED` omitting TTFT and the virtual-cache gauge
        # while the docstring claimed to check them (r13-2).
        path = self._write(_prom('provider="provx",tier="tierx"'))
        result = check_metrics.check_one("provx", "tierx", path)
        self.assertTrue(result["ok"], result["problems"])

        incomplete = self._write(
            "# scrape 1\n"
            'dynamo_component_proxy_requests_total{provider="p",tier="t"} 1\n'
            'dynamo_component_proxy_completion_tokens_total{provider="p",tier="t"} 1\n'
        )
        result = check_metrics.check_one("p", "t", incomplete)
        self.assertFalse(result["ok"])
        joined = " ".join(result["problems"])
        self.assertIn("time_to_first_token_seconds_count", joined)
        self.assertIn("virtual_cache_blocks", joined)

    def test_provider_and_tier_checked_independently(self) -> None:
        # Regression for requiring `provider == tier` even though the config
        # keeps them independent (r13-10).
        path = self._write(_prom('provider="openrouter",tier="proxy-x"'))
        result = check_metrics.check_one("openrouter", "proxy-x", path)
        self.assertTrue(result["ok"], result["problems"])

        result = check_metrics.check_one("proxy-x", "proxy-x", path)
        self.assertFalse(result["ok"])
        self.assertIn("no provider='proxy-x'", " ".join(result["problems"]))


class ReportTest(unittest.TestCase):
    def test_class_stickiness_excludes_unknown(self) -> None:
        # Regression for counting "unknown" as a class, which reported 1.0
        # when all worker identity was missing (r13-11).
        records = [
            {"session": 1, "turn": 0, "class": "unknown"},
            {"session": 1, "turn": 1, "class": "unknown"},
            {"session": 1, "turn": 2, "class": "unknown"},
        ]
        result = report.class_stickiness(records)
        self.assertEqual(result["follow_ups"], 0)
        self.assertIsNone(result["rate"])
        self.assertEqual(result["unknown_follow_ups"], 2)

    def test_compare_reports_absolute_band(self) -> None:
        # Regression for rendering the relative tolerance while the verdict
        # used the stricter absolute floor (r13-9).
        stats = {
            "requests": 50,
            "share": 0.5,
            "failed": 0,
            "p50_latency_ms": 1.0,
            "p95_latency_ms": 2.0,
            "p50_ttft_ms": 1.0,
        }
        report_obj = {
            "requests": 100,
            "failed_requests": 0,
            "classes": {"proxy-x": dict(stats), "hosted": dict(stats)},
            "stickiness": {"stayed": 0, "follow_ups": 0, "rate": None},
            "class_stickiness": {"stayed": 0, "follow_ups": 0, "rate": 0.6},
            "served_by": {"tagged": 0, "untagged": 0, "mismatched": 0},
            "windows": [],
        }
        baseline = {
            "overall": {
                "requests": 100,
                "hosted": 50,
                "hosted_share": 0.5,
                "by_tier": {"proxy-x": 50},
                "class_stickiness": 0.6,
                "failures": 0,
            }
        }
        rows = report.compare(report_obj, baseline, 0.1)
        share = next(r for r in rows if r["metric"] == "classes.proxy-x.share")
        self.assertEqual(share["band"], 0.05)

        markdown = report.format_markdown(report_obj, {}, rows)
        self.assertIn("| metric | e2e | level 1 | delta | band | result |", markdown)
        self.assertIn("0.05", markdown)

    def test_classify_without_tier_map_is_unknown_not_hosted(self) -> None:
        # Regression for a hardcoded rank table labelling a re-ranked proxy as
        # hosted, which made an all-hosted run a silent pass (s11-7).
        self.assertEqual(report.classify(1000, []), "unknown")
        tiers = [{"name": "proxy-x", "ranks": [1000, 1999]}]
        self.assertEqual(report.classify(1000, tiers), "proxy-x")
        self.assertEqual(report.classify(3, tiers), "hosted")

    def _report(self, hosted: int, proxies: dict[str, int]) -> dict:
        classes = {"hosted": {"requests": hosted, "failed": 0}}
        for name, count in proxies.items():
            classes[name] = {"requests": count, "failed": 0}
        return {
            "requests": hosted + sum(proxies.values()),
            "failed_requests": 0,
            "classes": classes,
            "served_by": {"ok": True},
        }

    def test_routing_checks_fail_when_no_traffic_spilled(self) -> None:
        # Regression for run.sh/report.py only failing on an untagged response,
        # so an all-hosted run passed the only whole-path test (s11-1/s11-7).
        rows = report.routing_checks(
            self._report(100, {}),
            min_proxy_share=0.05,
            max_hosted_share=None,
            required_tiers=["proxy-x"],
        )
        results = {row["metric"]: row["result"] for row in rows}
        self.assertEqual(results["proxy_share"], "FAIL")
        self.assertEqual(results["tier.proxy-x.requests"], "FAIL")

    def test_routing_checks_pass_when_spilled(self) -> None:
        rows = report.routing_checks(
            self._report(50, {"proxy-x": 40, "proxy-y": 10}),
            min_proxy_share=0.05,
            max_hosted_share=0.8,
            required_tiers=["proxy-x", "proxy-y"],
        )
        self.assertTrue(rows)
        self.assertTrue(all(row["result"] == "pass" for row in rows), rows)

    def test_routing_checks_fail_on_failed_requests(self) -> None:
        report_obj = self._report(0, {"proxy-x": 10})
        report_obj["failed_requests"] = 1
        rows = report.routing_checks(
            report_obj,
            min_proxy_share=0.0,
            max_hosted_share=None,
            required_tiers=[],
        )
        failed = next(r for r in rows if r["metric"] == "failed_requests")
        self.assertEqual(failed["result"], "FAIL")


class FakeProviderTest(unittest.TestCase):
    def test_mid_stream_abort_logs_499(self) -> None:
        # Regression for catching only ConnectionResetError/BrokenPipeError and
        # logging aborted streams as 200 (r13-12).
        class AbortWriter:
            def write(self, _data: bytes) -> None:
                pass

            async def drain(self) -> None:
                raise ConnectionAbortedError()

            def close(self) -> None:
                pass

            async def wait_closed(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "provider.jsonl")
            args = argparse.Namespace(
                concurrency=8,
                error_rate=0.0,
                error_status=503,
                max_tokens=4,
                ttft_ms=1.0,
                tps=1000.0,
                reasoning=False,
                reasoning_tokens=0,
                tool_call=False,
                log=log,
                name="p",
                model="m",
                seed=0,
                retry_after=1,
            )
            provider = fake_provider.FakeProvider(args)
            try:
                asyncio.run(
                    provider._chat(AbortWriter(), json.dumps({"model": "m"}).encode())
                )
            finally:
                provider.close()
            with open(log, encoding="utf-8") as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record["status"], 499)


class RunHelpersTest(unittest.TestCase):
    def test_render_template_is_literal(self) -> None:
        # Regression for sed interpolation corrupting `&`, `|` and `\` (r13-13).
        rendered = run_helpers.render_template(
            "path=${MODEL_PATH}", {"MODEL_PATH": r"/a&b|c\d"}
        )
        self.assertEqual(rendered, r"path=/a&b|c\d")

    def test_sanitize_model_dir_is_char_wise(self) -> None:
        # Regression for byte-wise `tr` disagreeing with the generator (r13-8).
        self.assertEqual(run_helpers.sanitize_model_dir("GLM-5.3@中文"), "GLM-5.3___")
        self.assertEqual(
            run_helpers.sanitize_model_dir("Qwen/Qwen3-0.6B"), "Qwen_Qwen3-0.6B"
        )

    def test_check_ports_detects_in_use_and_duplicates(self) -> None:
        # Regression for run.sh never checking ports (r13-3).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            in_use = sock.getsockname()[1]
            problems = run_helpers.check_ports([in_use])
            self.assertTrue(any("already in use" in p for p in problems), problems)
        duplicates = run_helpers.check_ports([1, 1])
        self.assertIn("port 1 is configured more than once", duplicates)

    def test_tier_map_derives_ranges_from_generated_configs(self) -> None:
        # Regression for report.py's default tier map being a hardcoded rank
        # table that can drift from the deployment generator (s11-7).
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for tier, rank in (("proxy-x", 1000), ("proxy-x", 1001), ("proxy-y", 2000)):
                path = os.path.join(tmp, f"{tier}-{rank}.yaml")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(
                        "model_path: m\n"
                        f"dp_rank: {rank}\n"
                        f"tier: {tier}\n"
                        "router_config:\n  mode: kv\n"
                    )
                paths[(tier, rank)] = path
            tier_map = run_helpers.build_tier_map(list(paths.values()))
        self.assertEqual(
            tier_map,
            [
                {"name": "proxy-x", "ranks": [1000, 1001]},
                {"name": "proxy-y", "ranks": [2000, 2000]},
            ],
        )

    def test_spillover_workflow_triggers_on_the_chat_request_integration(self) -> None:
        # Regression for a follow-up that changes only lib/llm (where the
        # chat-request integration lives) not running the spillover job, so
        # dw-proxy-worker could stop compiling unnoticed (s12-7).
        workflow = os.path.join(
            HERE, "..", "..", "..", ".github", "workflows", "spillover.yml"
        )
        with open(workflow, encoding="utf-8") as handle:
            text = handle.read()
        for path in ("lib/llm/**", "lib/bindings/python/rust/llm/**"):
            self.assertGreaterEqual(
                text.count(f"- '{path}'"),
                2,
                f"{path} must be in both pull_request and push path filters",
            )


if __name__ == "__main__":
    unittest.main()
