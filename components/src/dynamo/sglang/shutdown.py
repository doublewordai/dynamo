# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
import signal
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict

import psutil

from dynamo._core import DistributedRuntime
from dynamo.common.utils.graceful_shutdown import (
    get_drain_timeout_seconds,
    get_grace_period_seconds,
    graceful_shutdown_with_discovery,
)

SignalCallback = Callable[..., Any]

# A non-leader node's scheduler processes. Such a node serves no endpoint
# and has no request table: its schedulers work on the leader's requests and
# exit once the leader has drained and closed the distributed engine. The
# leader drains on the request plane's accepted-request counters instead.
_peer_schedulers: list[psutil.Process] = []
_PEER_POLL_SECS = 0.5
# The leader's engine cleanup budget, after its own grace and drain.
_LEADER_CLEANUP_SECS = 30.0


def register_drain_engine(engine: Any) -> None:
    """On a non-leader node, make the shutdown drain wait for this engine's
    scheduler processes; the leader needs nothing from the engine."""
    global _peer_schedulers
    _peer_schedulers = []
    server_args = getattr(engine, "server_args", None)
    if getattr(server_args, "node_rank", 0) <= 0:
        return
    for pid in engine.get_all_child_pids():
        try:
            _peer_schedulers.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            pass


def _running(process: psutil.Process) -> bool:
    # Process objects keep their creation time, so a reused PID does not
    # extend the wait.
    try:
        return process.is_running() and process.status() not in (
            psutil.STATUS_ZOMBIE,
            psutil.STATUS_DEAD,
        )
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # An unreadable process is not evidence of a finished scheduler.
        return True


async def wait_for_peer_schedulers() -> None:
    """Wait until the registered scheduler processes have exited."""
    logging.info("Drain: waiting for this node's schedulers to exit")
    while any(_running(process) for process in _peer_schedulers):
        await asyncio.sleep(_PEER_POLL_SECS)
    logging.info("Drain: this node's schedulers have exited")


def peer_wait_budget_seconds() -> float:
    """How long a non-leader node waits for the leader to finish: the leader's
    grace period, then, when it serves through gateway children, their own
    grace period and drain, then its engine cleanup."""
    return (
        2 * get_grace_period_seconds()
        + get_drain_timeout_seconds()
        + _LEADER_CLEANUP_SECS
    )


def install_graceful_shutdown(
    loop: asyncio.AbstractEventLoop,
    runtime: DistributedRuntime,
    endpoints: list[str],
    shutdown_event: asyncio.Event,
    *,
    signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
) -> Callable[[], Awaitable[None]]:
    """
    Set up graceful shutdown with discovery unregister, grace period and a
    drain: the leader waits for the requests its endpoints accepted, a
    non-leader node for its schedulers (see register_drain_engine), bounded
    by DYN_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_SECS.

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
        if _peer_schedulers:
            # A non-leader node outlives the leader's whole sequence, which is
            # longer than one drain timeout; its own endpoints serve nothing.
            try:
                await asyncio.wait_for(
                    wait_for_peer_schedulers(), timeout=peer_wait_budget_seconds()
                )
            except asyncio.TimeoutError:
                logging.warning("Drain: this node's schedulers are still running")
        await graceful_shutdown_with_discovery(
            runtime,
            endpoints,
            shutdown_event=shutdown_event,
            grace_period_s=None,
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
