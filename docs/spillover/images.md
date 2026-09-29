# Proxy worker image

The production image for `dw-proxy-worker` (`crates/proxy-worker`): one Dynamo
worker that registers as a Tokens chat worker with the SGLang workers it joins
and serves every request from a third-party OpenAI-compatible provider.

- Dockerfile: `docker/proxy-worker.Dockerfile`
- Build context: repository root
- Runtime user: `proxy` (uid/gid 10001), non-root
- Entrypoint: `dw-proxy-worker`, default `--config /etc/dw-proxy-worker/proxy.yaml`
- Image packages added over the base: `ca-certificates`, `libstdc++6`

## Build

Docker is not required on developer machines; the same binary is produced by
`cargo build --release -p dw-proxy-worker --locked`.

```bash
# From the repository root. Needs BuildKit (Docker 23+ defaults to it).
docker build \
  -f docker/proxy-worker.Dockerfile \
  -t dw-proxy-worker:dev \
  .
```

The builder uses the toolchain pinned in `rust-toolchain.toml` (the `rust:slim`
image's rustup installs it) and the workspace `.cargo/config.toml`
(`tokio_unstable` rustflags, `PCRE2_SYS_STATIC=1`). Registry, git, rustup and
`target/` are BuildKit cache mounts, so repeated builds reuse compilation
artifacts. `.dockerignore` keeps the context to the Rust workspace.

## Runtime dependencies

Built locally with `CARGO_TARGET_DIR=/home/peter/.cache/dw-target-b2 cargo build
--release -p dw-proxy-worker --locked`:

```text
size: 80,420,904 bytes (~77 MiB), not stripped
file: ELF 64-bit LSB pie executable, x86-64, dynamically linked

$ ldd /home/peter/.cache/dw-target-b2/release/dw-proxy-worker
        linux-vdso.so.1
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6
        libgcc_s.so.1  => /lib/x86_64-linux-gnu/libgcc_s.so.1
        libm.so.6      => /lib/x86_64-linux-gnu/libm.so.6
        libc.so.6      => /lib/x86_64-linux-gnu/libc.so.6
        /lib64/ld-linux-x86-64.so.2
```

Everything else is linked statically (Rust std, rustls, vendored onig/PCRE2,
ZMQ). `debian:bookworm-slim` ships libc/libgcc; the image adds
`ca-certificates` for HTTPS to providers and `libstdc++6` for the binary. There
is no OpenSSL or libzmq dependency at runtime.

## Run

```bash
docker run --rm \
  -v "$PWD/proxy.yaml:/etc/dw-proxy-worker/proxy.yaml:ro" \
  -v /models/GLM-5.3:/models/GLM-5.3:ro \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e DYN_DISCOVERY_BACKEND=etcd \
  -e ETCD_ENDPOINTS=http://etcd:2379 \
  -e DYN_REQUEST_PLANE=tcp \
  -e DYN_EVENT_PLANE=zmq \
  dw-proxy-worker:dev
```

Override the config path by replacing the argument list after the image:

```bash
docker run --rm ... dw-proxy-worker:dev --config /configs/glm-interactive.yaml
```

The process fails fast (before registering) when the provider API key
environment variable or the tokenizer file is missing.

## Environment

### Provider API key

`provider.api_key_env` in the proxy config names the environment variable that
holds the bearer token sent to the provider. It must be set in the container;
the example config uses `OPENROUTER_API_KEY`. Nothing else about the provider
(base URL, model slug, headers, timeouts, body overrides) comes from the
environment.

### Dynamo discovery, request and event planes

The proxy registers as an ordinary worker, so it must reach the frontend's
discovery backend and must speak the deployment's communication planes. These
are the same variables every Dynamo process uses; the worker leaves runtime
transport overrides unset and reads them from the environment:

| Variable | Purpose |
|---|---|
| `DYN_DISCOVERY_BACKEND` | `etcd` (default), `kubernetes`, `file`, `mem` |
| `ETCD_ENDPOINTS` | Comma-separated etcd endpoints when `etcd` |
| `DYN_FILE_KV` | Directory for the `file` discovery backend |
| `NATS_SERVER` | NATS URL when discovery/request/event uses NATS |
| `DYN_REQUEST_PLANE` | `tcp` (default) or `nats` |
| `DYN_EVENT_PLANE` | `zmq` (default) or `nats` |
| `DYN_TCP_RPC_HOST` / `DYN_TCP_RPC_PORT` | Bind/advertise address for the TCP request plane |
| `DYN_EVENT_PLANE_HOST` | Advertised ZMQ event-plane address |
| `DYN_SYSTEM_PORT` | Health/metrics HTTP port; `-1` disables it |

The Dynamo `namespace`, `component` and `endpoint` are **not** environment
variables here: they come from the proxy YAML (`namespace`, `component`,
`endpoint`) and must match the SGLang workers the proxy joins. `dp_rank` is the
reserved rank that marks this proxy's tier to the `dw-spillover` policy; it must
fall in the tier's `dp_ranks` range in the router policy YAML.

See the full field list in
`docs/fern/pages/reference/components/runtime-configuration.mdx` upstream.

## Model files

`model_path` in the config is the same local path the SGLang workers mount. The
worker registers a model card that mirrors theirs: it reads the card's tokenizer
and chat template from this directory but **never reads or downloads model
weights**. If `model_path` is a local directory it is used directly; if it is
not a directory it is treated as the `tokenizer.json` path itself.

Mount the directory read-only at the same path inside the container:

```bash
-v /models/GLM-5.3:/models/GLM-5.3:ro
```

A model card directory from Hugging Face contains:

- `tokenizer.json` (required): loaded by the engine to retokenize provider output
- `config.json`
- `tokenizer_config.json` (chat template)
- `generation_config.json`
- `special_tokens_map.json`

Because the path exists inside the container, Dynamo's `LocalModel` uses it
instead of fetching from Hugging Face. A bare HF repo id (not present on disk)
would make the worker download weights, which defeats the point; always mount
the same local model directory as the SGLang workers.

## Verifying an image

```bash
docker run --rm dw-proxy-worker:dev --help
docker run --rm \
  -v "$PWD/proxy.yaml:/etc/dw-proxy-worker/proxy.yaml:ro" \
  -v /models/GLM-5.3:/models/GLM-5.3:ro \
  -e OPENROUTER_API_KEY=dummy \
  dw-proxy-worker:dev
# parse/config errors print before the runtime starts; check the card and
# planes once a real frontend and etcd/NATS are reachable.
```
