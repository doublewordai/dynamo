# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model prefetch helpers."""

import asyncio
import logging
import multiprocessing
import os
import sys
from multiprocessing.connection import Connection

from dynamo.common.snapshot.lifecycle import (
    SENTINEL_POLL_INTERVAL_SEC,
    is_snapshot_enabled,
)

logger = logging.getLogger(__name__)


def _fetch_model_process_main(
    remote_name: str,
    ignore_weights: bool,
    result_conn: Connection,
) -> None:
    from dynamo._core import fetch_model

    async def fetch_model_async() -> str:
        return await fetch_model(remote_name, ignore_weights)

    try:
        local_path = asyncio.run(fetch_model_async())
        result_conn.send(("ok", local_path))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:
        result_conn.send(("error", repr(exc)))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


async def fetch_model_in_subprocess(
    remote_name: str,
    ignore_weights: bool = False,
) -> str:
    """Fetch a model in a short-lived process before snapshotting."""
    logger.info("Fetching model %s in a subprocess", remote_name)
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_fetch_model_process_main,
        args=(remote_name, ignore_weights, child_conn),
        name="dynamo-model-fetch",
    )
    proc.start()
    child_conn.close()
    try:
        while proc.is_alive():
            await asyncio.sleep(SENTINEL_POLL_INTERVAL_SEC)
        proc.join()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()

    if proc.exitcode != 0:
        message = parent_conn.recv()[1] if parent_conn.poll() else "unknown error"
        raise RuntimeError(
            f"Model fetch subprocess failed for {remote_name!r} "
            f"with exit code {proc.exitcode}: {message}"
        )

    if not parent_conn.poll():
        raise RuntimeError(
            f"Model fetch subprocess returned no result for {remote_name!r}"
        )
    status, payload = parent_conn.recv()
    if status != "ok":
        raise RuntimeError(
            f"Model fetch subprocess failed for {remote_name!r}: {payload}"
        )
    return payload


async def _fetch_model_once(remote_name: str, ignore_weights: bool) -> str:
    if is_snapshot_enabled():
        # Keep Hugging Face TCP sockets out of the snapshotted process.
        return await fetch_model_in_subprocess(remote_name, ignore_weights)

    from dynamo.llm import fetch_model as llm_fetch_model

    return await llm_fetch_model(remote_name, ignore_weights)


def _fetch_retry_settings() -> tuple[int, float, float]:
    """(attempts, first delay in seconds, delay cap in seconds)."""
    attempts = max(1, int(os.environ.get("DYN_MODEL_FETCH_ATTEMPTS", "6")))
    base = max(0.0, float(os.environ.get("DYN_MODEL_FETCH_RETRY_BASE_SECS", "10")))
    cap = max(base, float(os.environ.get("DYN_MODEL_FETCH_RETRY_MAX_SECS", "120")))
    return attempts, base, cap


async def fetch_model(remote_name: str, ignore_weights: bool = False) -> str:
    """Fetch a model into the local cache, retrying transient download failures.

    The downloader keeps every file it has already completed, so a retry only
    re-lists the repository and fetches what is still missing. Without this, a
    single Hugging Face 429 on one shard of a multi-hundred-GB checkpoint exits
    the worker and the pod crash-loops through its startup budget.

    The downloader reports every failure as a plain exception, so permanent
    ones (unknown repository, denied access) are retried too; the defaults
    bound that extra delay to under five minutes before the error propagates.
    """
    attempts, base, cap = _fetch_retry_settings()
    for attempt in range(1, attempts + 1):
        try:
            return await _fetch_model_once(remote_name, ignore_weights)
        except ImportError:
            raise
        except Exception as exc:
            if attempt >= attempts:
                raise
            delay = min(cap, base * 2 ** (attempt - 1))
            logger.warning(
                "Model fetch attempt %d/%d for %r failed: %s; retrying in %.0f s",
                attempt,
                attempts,
                remote_name,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
