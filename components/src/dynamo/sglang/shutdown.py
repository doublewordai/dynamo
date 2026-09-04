# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
import signal
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict

from dynamo._core import DistributedRuntime
from dynamo.common.utils.graceful_shutdown import graceful_shutdown_with_discovery

# Engine whose in-flight requests the shutdown drain waits for. Set once the
# engine exists; the drain is a no-op while it is unset.
_drain_engine: Any = None
# Requests the endpoint has accepted and whose handler has not finished, so a
# request still in handler-side work before it reaches the engine is drained too.
_handler_in_flight = 0
_DRAIN_POLL_SECS = 0.5
_DRAIN_LOG_EVERY_SECS = 10.0
# The engine must stay empty this long before the drain is declared done, so a
# request accepted by the endpoint but not yet handed to the engine, or one
# sent by a client that has not yet seen the discovery removal, is waited for.
_DRAIN_QUIET_SECS = 2.0


def register_drain_engine(engine: Any) -> None:
    """Make the shutdown drain wait for this engine's in-flight requests."""
    global _drain_engine
    _drain_engine = engine


def track_in_flight(generate: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a handler's generate so the drain can count requests it is
    serving, including those not yet handed to the engine."""

    async def tracked(request: Any, context: Any) -> Any:
        global _handler_in_flight
        _handler_in_flight += 1
        try:
            async for item in generate(request, context):
                yield item
        finally:
            _handler_in_flight -= 1

    return tracked


def handler_in_flight_count() -> int:
    """Requests currently inside a tracked handler."""
    return _handler_in_flight


def in_flight_request_count(engine: Any) -> int:
    """Requests the engine has accepted and not yet finished."""
    manager = getattr(engine, "tokenizer_manager", None)
    states = getattr(manager, "rid_to_state", None)
    if states is None:
        return 0
    return len(states)


async def drain_in_flight() -> None:
    """Wait until the registered engine has held no in-flight requests for a
    quiet period.

    In flight means inside a tracked handler (accepted by the endpoint, whether
    or not it has reached the engine yet) or in the engine's own request table.
    The caller bounds this with the drain timeout. The endpoints were
    unregistered before this runs, so new arrivals tail off; the quiet period
    covers a client that has not yet seen the removal.
    """
    engine = _drain_engine

    def remaining_in_flight() -> int:
        # Handler-only workers (diffusion, multimodal encode) register no
        # engine; their in-flight work is the tracked handlers alone.
        engine_count = in_flight_request_count(engine) if engine is not None else 0
        return engine_count + handler_in_flight_count()

    loop = asyncio.get_running_loop()
    started = loop.time()
    last_log = started
    empty_since = None
    remaining = remaining_in_flight()
    logging.info("Drain: %d in-flight requests at start", remaining)
    while True:
        now = loop.time()
        if remaining == 0:
            if empty_since is None:
                empty_since = now
            elif now - empty_since >= _DRAIN_QUIET_SECS:
                break
        else:
            empty_since = None
        if now - last_log >= _DRAIN_LOG_EVERY_SECS:
            logging.info(
                "Drain: %d in-flight requests after %.0fs", remaining, now - started
            )
            last_log = now
        await asyncio.sleep(_DRAIN_POLL_SECS)
        remaining = remaining_in_flight()
    logging.info(
        "Drain: no in-flight requests after %.1fs",
        asyncio.get_running_loop().time() - started,
    )


SignalCallback = Callable[..., Any]


def install_graceful_shutdown(
    loop: asyncio.AbstractEventLoop,
    runtime: DistributedRuntime,
    endpoints: list[str],
    shutdown_event: asyncio.Event,
    *,
    signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
) -> Callable[[], Awaitable[None]]:
    """
    Set up graceful shutdown with discovery unregister, grace period, and a
    drain of the registered engine's in-flight requests (see
    register_drain_engine) bounded by DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS.

    Owns OS-level SIGTERM/SIGINT via signal.signal() so SGLang's internal
    loop.add_signal_handler registrations cannot replace our handler.
    Monkey-patches loop.add_signal_handler to capture (defer) those
    registrations. Returns run_deferred_handlers to be invoked in init
    finally blocks (after the asyncio loop / serve_endpoint is done).
    """
    deferred_handlers: DefaultDict[
        int, list[tuple[SignalCallback, tuple[Any, ...]]]
    ] = defaultdict(
        list
    )  # type: ignore[assignment]

    shutdown_started = False
    shutdown_signum: int | None = None
    deferred_handlers_ran = False

    async def run_deferred_handlers() -> None:
        nonlocal deferred_handlers_ran
        if not shutdown_started or deferred_handlers_ran:
            return
        deferred_handlers_ran = True

        signums = (
            [shutdown_signum]
            if shutdown_signum is not None
            else list(deferred_handlers.keys())
        )
        for sig in signums:
            for cb, args in list(deferred_handlers.get(sig, [])):
                try:
                    res = cb(*args)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    logging.exception("Deferred signal callback failed: %r", cb)

    async def _shutdown_sequence(signum: int, frame: Any | None) -> None:
        nonlocal shutdown_started, shutdown_signum
        if shutdown_started:
            return
        shutdown_signum = signum
        shutdown_started = True

        logging.info("Received signal %s, starting graceful shutdown", signum)
        await graceful_shutdown_with_discovery(
            runtime,
            endpoints,
            shutdown_event=shutdown_event,
            grace_period_s=None,
            drain_callback=drain_in_flight,
        )

    def _schedule_shutdown(signum: int, frame: Any | None) -> None:
        def _kick() -> None:
            asyncio.create_task(_shutdown_sequence(signum, frame))

        loop.call_soon_threadsafe(_kick)

    def _os_signal_handler(signum: int, frame: Any) -> None:
        _schedule_shutdown(signum, frame)

    for sig in signals:
        signal.signal(sig, _os_signal_handler)

    orig_add = loop.add_signal_handler

    def watching_add_signal_handler(sig: int, callback: SignalCallback, *args: Any):
        if sig in signals:
            logging.debug(
                "Captured underlying service trying to register for loop.add_signal_handler(%s, %r, ...).",
                sig,
                callback,
            )
            deferred_handlers[sig].append((callback, args))
            return None
        return orig_add(sig, callback, *args)

    loop.add_signal_handler = watching_add_signal_handler  # type: ignore[assignment]

    return run_deferred_handlers
