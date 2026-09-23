# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine identities are independent of caller IDs and distributed traces."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamo._core import Context


def new_engine_request_id(context: Context | None) -> str:
    """Allocate an ID per engine invocation, including redispatch of a context.

    Keep W3C propagation separate: multiple requests may share a trace, and
    Dynamo's native context ID may be supplied by a caller. Use the returned
    ID for both dispatch and cancellation, even when tracing is disabled.
    """
    request_id = str(uuid.uuid4())
    if context is not None:
        context.current_span().set_attribute("sglang.request_id", request_id)
    return request_id


def new_embedding_request_ids(context: Context, batch_size: int) -> list[str]:
    """Allocate one ID per embedding input with bounded span metadata.

    The prefix and batch size reconstruct every item ID (``<prefix>-<index>``)
    without recording a potentially large array on the span. Hyphens avoid
    SGLang's legacy underscore stripping when it exports the engine ID.
    """
    if batch_size <= 0:
        raise ValueError("Embedding batch size must be positive")
    prefix = str(uuid.uuid4())
    span = context.current_span()
    span.set_attribute("sglang.request_id_prefix", prefix)
    span.set_attribute("sglang.batch_size", batch_size)
    return [f"{prefix}-{index}" for index in range(batch_size)]
