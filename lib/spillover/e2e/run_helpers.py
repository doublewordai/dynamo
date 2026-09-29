#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for ``run.sh`` that need Python semantics rather than shell/sed.

``render.sh``'s environment substitution, model-directory naming and port
pre-flight checks all have subtle rules (char-wise sanitization, literal
templating, distinct/free ports) that are hard to get right with ``sed`` and
``tr``. Keeping them here makes them unit-testable. Standard library only; see
``requirements.txt``.
"""

from __future__ import annotations

import argparse
import socket
import string
import sys


def sanitize_model_dir(name: str) -> str:
    """Mirror ``dw-spillover-deploy``'s ``sanitize``: one ``_`` per character.

    The generator substitutes one ``_`` per *char* outside
    ``[A-Za-z0-9._-]`` (``lib/spillover/deploy/src/lib.rs``), so a byte-wise
    ``tr`` would disagree on multi-byte characters.
    """
    return "".join(
        c if (c.isascii() and (c.isalnum() or c in "-._")) else "_" for c in name
    )


def render_template(template: str, values: dict[str, str]) -> str:
    """Substitute ``${NAME}`` placeholders literally, with no regex semantics."""
    return string.Template(template).substitute(values)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def check_ports(ports: list[int]) -> list[str]:
    """Return human-readable problems for a set of ports the run needs to bind.

    Detects duplicate ports (two services configured onto one port, e.g. a
    hosted mocker range that overlaps the proxy system ports) and ports already
    bound by another process, so a stale process cannot satisfy readiness and
    be scraped as if it were this run's.
    """
    problems: list[str] = []
    seen: set[int] = set()
    for port in ports:
        if port in seen:
            problems.append(f"port {port} is configured more than once")
        seen.add(port)
    for port in dict.fromkeys(ports):
        if not _port_free(port):
            problems.append(f"port {port} is already in use")
    return problems


def read_proxy_provider(path: str) -> str:
    """Return the top-level ``provider.name`` from a generated proxy config.

    The generated files are simple serde YAML; scanning for the top-level
    ``provider:`` key avoids depending on a YAML library (the e2e scripts are
    standard library only).
    """
    with open(path, encoding="utf-8") as handle:
        in_provider = False
        for line in handle:
            if not in_provider:
                if line.rstrip("\n") == "provider:":
                    in_provider = True
                continue
            stripped = line.strip()
            if stripped.startswith("name:"):
                return stripped.split(":", 1)[1].strip().strip("'\"")
            if stripped and not line.startswith((" ", "\t")):
                break
    raise SystemExit(f"no top-level provider.name found in {path}")


def _parse_set(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {value!r}")
        out[key] = val
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sanitize = sub.add_parser("sanitize", help="sanitize a model name for a dir name")
    sanitize.add_argument("--name", required=True)

    render = sub.add_parser("render", help="substitute ${NAME} placeholders in a file")
    render.add_argument("--template", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    ports = sub.add_parser("check-ports", help="fail if any required port is unusable")
    ports.add_argument("--ports", type=int, nargs="+", required=True)

    provider = sub.add_parser(
        "proxy-provider", help="read provider.name from a generated proxy config"
    )
    provider.add_argument("--config", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "sanitize":
        print(sanitize_model_dir(args.name))
        return 0
    if args.command == "render":
        with open(args.template, encoding="utf-8") as handle:
            template = handle.read()
        rendered = render_template(template, _parse_set(args.set))
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        return 0
    if args.command == "check-ports":
        problems = check_ports(args.ports)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    if args.command == "proxy-provider":
        print(read_proxy_provider(args.config))
        return 0
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
