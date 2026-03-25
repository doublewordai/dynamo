# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import terminate_process_tree

from .test_vllm import DynamoWorkerProcess
from .utils import (
    DynamoFrontendProcess,
    determine_request_receiving_worker,
    start_chat_completion_request,
    validate_response,
    verify_migration_metrics,
    verify_migration_occurred,
    wait_for_response,
)

logger = logging.getLogger(__name__)


def _structured_chat_payload() -> dict:
    return {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return a JSON object with exactly 24 animals. "
                    "Each animal must include name, habitat, and a detailed fact "
                    "with 2 sentences."
                ),
            }
        ],
        "stream": True,
        "temperature": 0,
        "max_tokens": 1600,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "animal_catalog",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "animals": {
                            "type": "array",
                            "minItems": 24,
                            "maxItems": 24,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "habitat": {"type": "string"},
                                    "fact": {"type": "string"},
                                },
                                "required": ["name", "habitat", "fact"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["animals"],
                    "additionalProperties": False,
                },
            },
        },
    }


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.post_merge
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(290)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_request_migration_vllm_aggregated_structured_output(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
):
    with DynamoFrontendProcess(request, migration_limit=3) as frontend:
        logger.info("Frontend started successfully")

        with DynamoWorkerProcess(request, "worker1", frontend.frontend_port) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(request, "worker2", frontend.frontend_port) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                request_thread, response_list = start_chat_completion_request(
                    frontend.frontend_port,
                    stream=True,
                    payload_override=_structured_chat_payload(),
                )

                worker, worker_name = determine_request_receiving_worker(
                    worker1,
                    worker2,
                    receiving_pattern="Decode Request ID: ",
                )
                wait_for_response(response_list, num_responses=20)

                logger.info(
                    "Killing %s with PID %s during structured-output stream",
                    worker_name,
                    worker.get_pid(),
                )
                terminate_process_tree(worker.get_pid(), immediate_kill=True, timeout=0)

                validate_response(request_thread, response_list, validate_delay=True)
                verify_migration_occurred(frontend)
                verify_migration_metrics(
                    frontend.frontend_port, expected_ongoing_request_count=1
                )

                response_text = "".join(
                    chunk
                    for chunk, _timestamp in response_list[1:]
                    if isinstance(chunk, str)
                )
                parsed = json.loads(response_text)
                assert isinstance(parsed, dict)
                assert "animals" in parsed
