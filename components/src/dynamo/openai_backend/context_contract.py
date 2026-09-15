# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the public context contract before a worker enters discovery."""

import logging
from urllib.parse import urlsplit, urlunsplit

import httpx

LOGGER = logging.getLogger(__name__)


def positive_context_length(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError("expected context length must be positive")
    return result


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


async def verify_context_contract(
    engine_url: str, model_names: set[str], expected: int
) -> None:
    """Fail closed if engine metadata cannot confirm the advertised limit.

    This checks the engine's resolved limit, not the HF architecture limit.
    It complements a deployment test near the context boundary; metadata alone
    cannot prove that the engine has enough cache to execute that request.
    """
    if expected <= 0:
        raise ValueError("expected context length must be positive")
    parts = urlsplit(engine_url)
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    observed = None
    async with httpx.AsyncClient(trust_env=False, timeout=10.0) as client:
        response = await client.get(origin + "/get_server_info")
        if response.status_code == 200:
            info = response.json()
            if isinstance(info, dict):
                observed = _positive_int(info.get("context_length"))
        elif response.status_code not in (404, 405):
            response.raise_for_status()
        if observed is None:
            response = await client.get(origin + "/v1/models")
            response.raise_for_status()
            body = response.json()
            models = body.get("data", []) if isinstance(body, dict) else []
            limits = [
                value
                for model in models
                if isinstance(model, dict) and model.get("id") in model_names
                if (value := _positive_int(model.get("max_model_len"))) is not None
            ]
            if limits:
                observed = min(limits)
    if observed is None or observed < expected:
        raise RuntimeError(
            "Refusing worker registration: expected context length "
            f"{expected}, engine reports {observed!r}. Qualify a matching worker "
            "before enabling this model."
        )
    LOGGER.info("Context contract verified: expected=%s engine=%s", expected, observed)
