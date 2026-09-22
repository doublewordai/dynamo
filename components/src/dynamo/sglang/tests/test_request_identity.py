# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only identity tests; no SGLang engine or compiled Dynamo bindings."""

from types import SimpleNamespace
from uuid import UUID

import pytest

from dynamo.sglang.request_identity import (
    new_embedding_request_ids,
    new_engine_request_id,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_reused_context_allocates_distinct_engine_ids():
    attributes = {}
    context = SimpleNamespace(
        id=lambda: "caller-controlled-id",
        trace_id="shared-trace",
        current_span=lambda: SimpleNamespace(set_attribute=attributes.__setitem__),
    )
    first = new_engine_request_id(context)
    assert attributes["sglang.request_id"] == first
    second = new_engine_request_id(context)
    assert attributes["sglang.request_id"] == second
    assert first != second
    assert UUID(first).version == UUID(second).version == 4


def test_contextless_engine_calls_still_get_unique_ids():
    first, second = new_engine_request_id(None), new_engine_request_id(None)
    assert first != second
    assert UUID(first).version == UUID(second).version == 4


@pytest.mark.parametrize("batch_size", [1, 2, 1024])
def test_embedding_ids_reconstruct_from_bounded_span_metadata(batch_size):
    attributes = {}
    context = SimpleNamespace(
        current_span=lambda: SimpleNamespace(set_attribute=attributes.__setitem__)
    )
    ids = new_embedding_request_ids(context, batch_size)
    prefix = attributes["sglang.request_id_prefix"]
    assert UUID(prefix).version == 4
    assert attributes == {
        "sglang.request_id_prefix": prefix,
        "sglang.batch_size": batch_size,
    }
    assert ids == [f"{prefix}-{i}" for i in range(batch_size)]
    assert len(set(ids)) == batch_size
    assert all("_" not in rid for rid in ids)
    assert set(ids).isdisjoint(new_embedding_request_ids(context, batch_size))


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_embedding_batch_has_no_engine_identity(batch_size):
    attributes = {}
    context = SimpleNamespace(
        current_span=lambda: SimpleNamespace(set_attribute=attributes.__setitem__)
    )
    with pytest.raises(ValueError, match="Embedding batch size must be positive"):
        new_embedding_request_ids(context, batch_size)
    assert attributes == {}
