# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# syntax=docker/dockerfile:1.6
# Production image for the `dw-proxy-worker` binary
# (lib/spillover/proxy-worker).
#
# The binary links Dynamo's runtime, so the build must see the whole workspace:
# context is the repository root (see container/ for how the fork builds its
# other Rust images; this one stays minimal because it ships no Python).
#
#   docker build -f lib/spillover/proxy-worker/Dockerfile -t dw-proxy-worker .
#
# Docker is not needed on the developer machine to keep this file honest; the
# release binary it builds is exercised by `cargo build --release
# -p dw-proxy-worker`. See docs/spillover/images.md for the run contract.

# ---------------------------------------------------------------------------
# Builder: the `rust:slim` image ships rustup, which installs the toolchain
# pinned in rust-toolchain.toml (currently 1.96.1) on first cargo invocation and
# reads .cargo/config.toml (tokio_unstable rustflags, PCRE2_SYS_STATIC). All of
# registry/git, rustup and target are BuildKit cache mounts.
# ---------------------------------------------------------------------------
FROM rust:slim-bookworm AS builder

# C/C++ toolchain and libraries Dynamo's Rust crates build against:
#   build-essential   cc, for onig_sys/PCRE2/etc. vendored C sources
#   cmake             pcre2-sys/other -sys crates with CMake builds
#   clang/libclang-dev  bindgen for -sys crates
#   protobuf-compiler  dynamo runtime generated types
#   libssl-dev/pkg-config  TLS/openssl -sys crates
#   git                cargo fetch of the pinned ai-dynamo/dynamo git dependency
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libclang-dev \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# The whole workspace (rust-toolchain.toml, .cargo/, Cargo.toml/Cargo.lock,
# lib/). .dockerignore keeps the context to manifests and source.
COPY . .

# Install the pinned toolchain into the image layer (the base image's rustup lives in
# /usr/local/rustup; a cache mount there would hide it).
RUN rustup show active-toolchain || rustup toolchain install

# Cache mounts persist across builds and are not part of the image:
#   registry/git  downloaded crate sources and checkouts, safe to share
#   target        compiled artifacts, locked while writing
# Copy the binary out of the (non-image) cache in the same RUN.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=shared \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=shared \
    --mount=type=cache,target=/build/target,sharing=locked \
    mkdir -p /out \
    && cargo build --release -p dw-proxy-worker --locked \
    && cp /build/target/release/dw-proxy-worker /out/

# ---------------------------------------------------------------------------
# Runtime: slim base with only the binary's shared libraries, a non-root user
# and CA certificates for HTTPS to providers.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# ca-certificates is required for outbound HTTPS to OpenAI-compatible providers
# (reqwest/rustls). libstdc++6 is the one non-base C++ runtime the binary links;
# ldd shows libstdc++, libgcc_s, libm and libc only (docs/spillover/images.md).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. The proxy only reads its config and model card from
# mounted volumes, so no writable state is required.
RUN groupadd --system --gid 10001 dwproxy \
    && useradd --system --uid 10001 --gid dwproxy --create-home --home-dir /home/dwproxy dwproxy

COPY --from=builder /out/dw-proxy-worker /usr/local/bin/dw-proxy-worker

# Default config location; mount the deployment config there, or override the
# whole argument list: docker run ... dw-proxy-worker --config /elsewhere.yaml
RUN mkdir -p /etc/dw-proxy-worker && chown -R dwproxy:dwproxy /etc/dw-proxy-worker

USER dwproxy
WORKDIR /home/dwproxy

# The config carries the provider api_key_env name, the model_path whose
# tokenizer/config files must be mounted, and the Dynamo namespace/component/
# endpoint. Transport planes come from the environment; see docs/spillover/images.md.
ENTRYPOINT ["dw-proxy-worker"]
CMD ["--config", "/etc/dw-proxy-worker/proxy.yaml"]
