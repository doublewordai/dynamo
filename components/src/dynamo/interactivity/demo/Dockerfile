# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libssl3 libnuma1 libzmq5 && rm -rf /var/lib/apt/lists/*
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl aiohttp prometheus-client tokenizers pyyaml && rm -rf /tmp/wheels
COPY components/src/dynamo/ /opt/components/dynamo/
ENV PYTHONPATH=/opt/components PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RUN python /opt/components/dynamo/interactivity/demo/make_model.py
