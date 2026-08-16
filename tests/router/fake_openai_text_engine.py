# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small OpenAI-compatible server used by text KV-router integration tests."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class EngineState:
    def __init__(
        self,
        label: str,
        loads: list[float],
        queue_depths: list[float],
        total_kv_blocks: int,
    ) -> None:
        self.label = label
        self._loads = loads
        self._queue_depths = queue_depths
        self.total_kv_blocks = total_kv_blocks
        self._metrics_enabled = True
        self._hold_seconds = 0.0
        self._lock = threading.Lock()

    def loads(self) -> list[float]:
        with self._lock:
            return list(self._loads)

    def set_loads(self, loads: list[float]) -> None:
        with self._lock:
            self._loads = loads
            del self._queue_depths[len(loads) :]
            self._queue_depths.extend(
                0.0 for _ in range(len(loads) - len(self._queue_depths))
            )

    def queue_depths(self) -> list[float]:
        with self._lock:
            return list(self._queue_depths)

    def set_queue_depths(self, depths: list[float]) -> None:
        with self._lock:
            self._queue_depths = depths

    def hold_seconds(self) -> float:
        with self._lock:
            return self._hold_seconds

    def set_hold_seconds(self, seconds: float) -> None:
        with self._lock:
            self._hold_seconds = seconds

    def metrics_enabled(self) -> bool:
        with self._lock:
            return self._metrics_enabled

    def set_metrics_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._metrics_enabled = enabled


class Handler(BaseHTTPRequestHandler):
    server: "FakeEngineServer"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/health":
            self._json_response({"status": "ready"})
            return
        if self.path == "/get_server_info":
            self._json_response(
                {
                    "max_total_num_tokens": self.server.state.total_kv_blocks,
                    "max_prefill_tokens": 1000,
                    "page_size": 1,
                    "dp_size": len(self.server.state.loads()),
                    "max_running_requests": 64,
                    "internal_states": [
                        {"effective_max_running_requests_per_dp": 64}
                        for _ in self.server.state.loads()
                    ],
                }
            )
            return
        if self.path == "/metrics":
            if not self.server.state.metrics_enabled():
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self._metrics_response()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/admin/loads":
            payload = self._read_json()
            raw_loads = payload.get("loads")
            if not isinstance(raw_loads, list) or not raw_loads:
                self.send_error(
                    HTTPStatus.BAD_REQUEST, "loads must be a non-empty list"
                )
                return
            loads = [float(load) for load in raw_loads]
            if any(load < 0.0 or load > 1.0 for load in loads):
                self.send_error(HTTPStatus.BAD_REQUEST, "loads must be between 0 and 1")
                return
            self.server.state.set_loads(loads)
            self._json_response({"loads": loads})
            return
        if self.path == "/admin/queue":
            payload = self._read_json()
            raw_depths = payload.get("queue")
            if not isinstance(raw_depths, list) or not raw_depths:
                self.send_error(
                    HTTPStatus.BAD_REQUEST, "queue must be a non-empty list"
                )
                return
            depths = [float(depth) for depth in raw_depths]
            if any(depth < 0.0 for depth in depths):
                self.send_error(HTTPStatus.BAD_REQUEST, "queue depths must be >= 0")
                return
            self.server.state.set_queue_depths(depths)
            self._json_response({"queue": depths})
            return
        if self.path == "/admin/hold":
            payload = self._read_json()
            seconds = payload.get("seconds")
            if not isinstance(seconds, (int, float)) or seconds < 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "seconds must be >= 0")
                return
            self.server.state.set_hold_seconds(float(seconds))
            self._json_response({"seconds": seconds})
            return
        if self.path == "/admin/metrics":
            payload = self._read_json()
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                self.send_error(HTTPStatus.BAD_REQUEST, "enabled must be a boolean")
                return
            self.server.state.set_metrics_enabled(enabled)
            self._json_response({"enabled": enabled})
            return
        if self.path == "/v1/chat/completions":
            self._chat_completion()
            return
        if self.path == "/abort_request":
            self._json_response({"aborted": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _metrics_response(self) -> None:
        lines = [
            "# TYPE sglang:token_usage gauge",
            "# TYPE sglang:num_queue_reqs gauge",
            "# TYPE sglang:num_running_reqs gauge",
        ]
        queue_depths = self.server.state.queue_depths()
        for rank, load in enumerate(self.server.state.loads()):
            queued = queue_depths[rank] if rank < len(queue_depths) else 0.0
            lines.extend(
                [
                    f'sglang:token_usage{{dp_rank="{rank}"}} {load}',
                    f'sglang:num_queue_reqs{{dp_rank="{rank}"}} {queued}',
                    f'sglang:num_running_reqs{{dp_rank="{rank}"}} 0',
                ]
            )
        body = ("\n".join(lines) + "\n").encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chat_completion(self) -> None:
        request = self._read_json()
        rank = self.headers.get("X-Data-Parallel-Rank", "none")
        content = f"{self.server.state.label}:rank-{rank}"
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        common: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.get("model", "text-kv-e2e"),
        }
        chunks = [
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **common,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ]

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        first, rest = chunks[0], chunks[1:]
        self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
        self.wfile.flush()
        # When a hold is configured, keep the stream open after the first
        # chunk so tests can act on long-lived in-flight requests.
        hold = self.server.state.hold_seconds()
        if hold > 0:
            time.sleep(hold)
        for chunk in rest:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    def _json_response(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class FakeEngineServer(ThreadingHTTPServer):
    def __init__(self, port: int, state: EngineState) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.state = state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--loads", required=True)
    parser.add_argument("--queues")
    parser.add_argument("--total-kv-blocks", default=1000, type=int)
    args = parser.parse_args()

    loads = [float(load) for load in args.loads.split(",")]
    queue_depths = (
        [float(depth) for depth in args.queues.split(",")]
        if args.queues is not None
        else [0.0 for _ in loads]
    )
    if len(queue_depths) != len(loads):
        parser.error("--queues must contain one value per --loads rank")
    FakeEngineServer(
        args.port,
        EngineState(args.label, loads, queue_depths, args.total_kv_blocks),
    ).serve_forever()


if __name__ == "__main__":
    main()
