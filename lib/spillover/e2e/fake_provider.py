#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible fake provider for the Level 2 e2e simulation.

Serves ``POST /v1/chat/completions`` as a chunked SSE stream so the real
``dw-proxy-worker`` upstream client, renderers and retokenizer are exercised
without a model or network. Every request is appended as one JSON line with the
provider name and prompt size so the report can attribute load to a tier.

Standard library only (asyncio); see ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import time

# Reasons for the status codes we actually emit, so responses look normal to
# reqwest and curl.
_REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

_WORDS = (
    "the quick brown fox jumps over a lazy dog while routers rank workers by "
    "cache affinity and spill overflow traffic to proxied providers".split()
)

_TOOL_NAME = "get_weather"
_TOOL_ARGUMENTS = '{"city":"Berlin","unit":"celsius"}'


def _text_length(content: object) -> int:
    """Prompt size proxy: character count, handling multimodal content parts."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return 0


async def _read_request(reader: asyncio.StreamReader):
    """Parse one HTTP/1.1 request. Returns None on a closed connection."""
    try:
        request_line = await reader.readline()
    except (ConnectionError, asyncio.IncompleteReadError):
        return None
    if not request_line:
        return None
    parts = request_line.decode("latin-1").split()
    if len(parts) < 3:
        return None
    method, target = parts[0], parts[1]

    headers: dict[str, str] = {}
    while True:
        try:
            line = await reader.readline()
        except (ConnectionError, asyncio.IncompleteReadError):
            return None
        if not line or line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("latin-1").partition(":")
        headers[key.strip().lower()] = value.strip()

    body = b""
    length = int(headers.get("content-length") or 0)
    if length:
        try:
            body = await reader.readexactly(length)
        except (ConnectionError, asyncio.IncompleteReadError):
            return None
    return method, target, headers, body


def _head(
    status: int,
    content_type: str,
    extra_headers: dict[str, str] | None = None,
    content_length: int | None = None,
) -> bytes:
    lines = [
        f"HTTP/1.1 {status} {_REASONS.get(status, 'OK')}",
        f"Content-Type: {content_type}",
    ]
    if content_length is None:
        lines.append("Transfer-Encoding: chunked")
    else:
        lines.append(f"Content-Length: {content_length}")
    for key, value in (extra_headers or {}).items():
        lines.append(f"{key}: {value}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


async def _send_json(
    writer: asyncio.StreamWriter,
    status: int,
    payload: object,
    extra_headers: dict[str, str] | None = None,
) -> None:
    body = json.dumps(payload).encode()
    writer.write(_head(status, "application/json", extra_headers, len(body)) + body)
    await writer.drain()


async def _sse_start(writer: asyncio.StreamWriter) -> None:
    writer.write(_head(200, "text/event-stream", {"Cache-Control": "no-cache"}))
    await writer.drain()


async def _sse_write(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
    await writer.drain()


async def _sse_event(writer: asyncio.StreamWriter, payload: object) -> None:
    data = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
    await _sse_write(writer, data)


async def _sse_done(writer: asyncio.StreamWriter) -> None:
    await _sse_write(writer, b"data: [DONE]\n\n")
    writer.write(b"0\r\n\r\n")
    await writer.drain()


class FakeProvider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.active = 0
        # A single event loop means a plain synchronous append is atomic enough.
        self._rng = random.Random(args.seed)
        self._log_file = open(args.log, "a", encoding="utf-8") if args.log else None

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await _read_request(reader)
            if request is None:
                return
            method, target, _headers, body = request
            path = target.split("?", 1)[0]
            if method == "GET" and path in ("/health", "/healthz"):
                await _send_json(writer, 200, {"status": "ok"})
            elif method == "GET" and path == "/v1/models":
                await _send_json(
                    writer,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.args.model,
                                "object": "model",
                                "owned_by": self.args.name,
                            }
                        ],
                    },
                )
            elif method == "POST" and path in (
                "/v1/chat/completions",
                "/chat/completions",
            ):
                await self._chat(writer, body)
            else:
                await _send_json(
                    writer,
                    404,
                    {
                        "error": {
                            "message": f"unknown path {path}",
                            "type": "invalid_request_error",
                        }
                    },
                )
        except (ConnectionError, asyncio.IncompleteReadError):
            # The load generator or proxy disconnected; nothing to answer.
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass

    async def _chat(self, writer: asyncio.StreamWriter, body: bytes) -> None:
        started = time.monotonic()
        try:
            request = json.loads(body or b"{}")
        except json.JSONDecodeError:
            await _send_json(
                writer,
                400,
                {
                    "error": {
                        "message": "invalid JSON body",
                        "type": "invalid_request_error",
                    }
                },
            )
            self._log("<invalid>", 0, 0, 400, started)
            return
        if not isinstance(request, dict):
            request = {}

        model = request.get("model") or self.args.model
        prompt_chars = sum(
            _text_length(message.get("content"))
            for message in request.get("messages", [])
            if isinstance(message, dict)
        )

        if self.active >= self.args.concurrency:
            await _send_json(
                writer,
                429,
                {
                    "error": {
                        "message": "simulated concurrency limit",
                        "type": "rate_limit_error",
                    }
                },
                {"Retry-After": str(self.args.retry_after)},
            )
            self._log(model, prompt_chars, 0, 429, started)
            return

        if self.args.error_rate > 0 and self._rng.random() < self.args.error_rate:
            await _send_json(
                writer,
                self.args.error_status,
                {
                    "error": {
                        "message": "simulated provider failure",
                        "type": "server_error",
                    }
                },
            )
            self._log(model, prompt_chars, 0, self.args.error_status, started)
            return

        output_tokens = max(1, self.args.max_tokens)
        body_max = request.get("max_tokens") or request.get("max_completion_tokens")
        if isinstance(body_max, int) and body_max > 0:
            output_tokens = min(output_tokens, body_max)

        self.active += 1
        # 499 (client closed request) if the stream did not reach `_sse_done`;
        # a mid-stream disconnect must not be logged as a clean 200.
        status = 499
        try:
            await _sse_start(writer)
            await asyncio.sleep(self.args.ttft_ms / 1000.0)

            await _sse_event(
                writer,
                self._chunk(model, {"role": "assistant", "content": ""}),
            )

            if self.args.reasoning:
                for _ in range(self.args.reasoning_tokens):
                    await _sse_event(
                        writer,
                        self._chunk(model, {"reasoning": self._word()}),
                    )
                    await asyncio.sleep(1.0 / self.args.tps)

            if self.args.tool_call:
                await self._stream_tool_call(writer, model)
            else:
                for _ in range(output_tokens):
                    await _sse_event(
                        writer, self._chunk(model, {"content": self._word()})
                    )
                    await asyncio.sleep(1.0 / self.args.tps)

            finish_reason = "tool_calls" if self.args.tool_call else "stop"
            usage = {
                "prompt_tokens": max(1, prompt_chars // 4),
                "completion_tokens": output_tokens,
                "total_tokens": max(1, prompt_chars // 4) + output_tokens,
            }
            await _sse_event(
                writer,
                self._chunk(model, {}, finish_reason=finish_reason, usage=usage),
            )
            await _sse_done(writer)
            status = 200
        except ConnectionError:
            # Client cancelled mid-stream; the worker maps that to a migration.
            pass
        finally:
            self.active -= 1
            self._log(model, prompt_chars, output_tokens, status, started)

    async def _stream_tool_call(self, writer: asyncio.StreamWriter, model: str) -> None:
        call_id = f"call_{self._rng.randrange(1 << 30):08x}"
        await _sse_event(
            writer,
            self._chunk(
                model,
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": _TOOL_NAME, "arguments": ""},
                        }
                    ]
                },
            ),
        )
        for piece in _split_tool_arguments(_TOOL_ARGUMENTS):
            await _sse_event(
                writer,
                self._chunk(
                    model,
                    {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]},
                ),
            )
            await asyncio.sleep(1.0 / self.args.tps)

    def _chunk(
        self,
        model: str,
        delta: dict,
        finish_reason: str | None = None,
        usage: dict | None = None,
    ) -> dict:
        payload = {
            "id": f"chatcmpl-fake-{self._rng.randrange(1 << 30):08x}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    def _word(self) -> str:
        return self._rng.choice(_WORDS)

    def _log(
        self,
        model: str,
        prompt_chars: int,
        output_tokens: int,
        status: int,
        started: float,
    ) -> None:
        record = {
            "ts": time.time(),
            "provider": self.args.name,
            "model": model,
            "prompt_chars": prompt_chars,
            "output_tokens": output_tokens,
            "status": status,
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
        }
        line = json.dumps(record, separators=(",", ":"))
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()
        else:
            print(line, flush=True)


def _split_tool_arguments(arguments: str, pieces: int = 4) -> list[str]:
    """Split arguments into small pieces, mimicking streamed tool-call deltas."""
    if pieces < 1:
        return [arguments]
    step = max(1, len(arguments) // pieces)
    return [arguments[i : i + step] for i in range(0, len(arguments), step)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--name", default="fake-provider", help="tier name in logs")
    parser.add_argument(
        "--model", default="fake-model", help="model id echoed to clients"
    )
    parser.add_argument(
        "--ttft-ms", type=float, default=50.0, help="time to first token"
    )
    parser.add_argument(
        "--tps", type=float, default=100.0, help="decode tokens per second"
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="output length cap")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="in-flight requests above which 429 is returned",
    )
    parser.add_argument(
        "--retry-after", type=int, default=1, help="Retry-After seconds"
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help="fraction of requests answered with --error-status",
    )
    parser.add_argument("--error-status", type=int, default=503)
    parser.add_argument(
        "--reasoning", action="store_true", help="emit reasoning deltas"
    )
    parser.add_argument("--reasoning-tokens", type=int, default=8)
    parser.add_argument(
        "--tool-call", action="store_true", help="emit one streamed tool call"
    )
    parser.add_argument(
        "--log", default=None, help="JSONL log path (stdout when omitted)"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


async def _serve(args: argparse.Namespace) -> None:
    provider = FakeProvider(args)
    server = await asyncio.start_server(provider.handle, args.host, args.port)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(
        f"fake_provider {args.name} listening on {addresses} "
        f"(ttft={args.ttft_ms}ms tps={args.tps} concurrency={args.concurrency} "
        f"error_rate={args.error_rate})",
        flush=True,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        await stop.wait()
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
    provider.close()


def main() -> int:
    args = parse_args()
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
