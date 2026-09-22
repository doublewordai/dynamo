# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint the selected Cargo workspace and its local dependency manifests."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def cargo_metadata(manifest):
    # --no-deps avoids downloading registry crates before restoring their cache.
    return json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                str(manifest),
            ],
            cwd=manifest.parent,
        )
    )


def manifest_inputs(workspace, repository, metadata=cargo_metadata):
    workspace = workspace.resolve()
    repository = repository.resolve()
    packages = {}
    roots = {}

    def load(manifest):
        result = metadata(manifest)
        root = Path(result["workspace_root"])
        for package in result["packages"]:
            path = Path(package["manifest_path"])
            packages[path] = package
            roots[path] = root
        return [Path(p["manifest_path"]) for p in result["packages"]]

    pending = load(workspace / "Cargo.toml")
    inputs = {workspace / "Cargo.lock"}
    visited = set()
    while pending:
        manifest = pending.pop()
        if manifest in visited:
            continue
        manifest.relative_to(repository)  # Local dependencies must be in checkout.
        if manifest not in packages:
            load(manifest)
        visited.add(manifest)
        inputs.update((manifest, roots[manifest] / "Cargo.toml"))
        pending.extend(
            Path(dependency["path"]) / "Cargo.toml"
            for dependency in packages[manifest]["dependencies"]
            if "path" in dependency
        )

    # Cargo/rustup discover configuration from the invocation directory upward.
    for directory in (workspace, *workspace.parents):
        inputs.update(directory.glob("rust-toolchain*"))
        inputs.update((directory / ".cargo").glob("config*"))
        if directory == repository:
            break
    return sorted(path for path in inputs if path.is_file())


def fingerprint(paths, repository):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_state(repository):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repository).decode(
        "utf-8", errors="surrogateescape"
    )
    result = {}
    directories = set()
    for name in tracked.split("\0"):
        path = repository / name
        if not name or path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        result[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mode": stat.st_mode,
            "mtime_ns": stat.st_mtime_ns,
        }
        parent = path.parent
        while parent != repository:
            directories.add(parent)
            parent = parent.parent

    # Protobuf build scripts also watch include directories. Restore a
    # directory only when its entire tree consists of matching tracked files.
    # An untracked file or symlink makes it ineligible, preserving invalidation
    # for generated/new inputs that are absent from the Git file list.
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        children = []
        for child in sorted(directory.iterdir()):
            state = result.get(str(child.relative_to(repository)))
            if state is None:
                break
            children.append((child.name, state["sha256"], state["mode"]))
        else:
            stat = directory.stat()
            result[str(directory.relative_to(repository))] = {
                "sha256": hashlib.sha256(json.dumps(children).encode()).hexdigest(),
                "mode": stat.st_mode,
                "mtime_ns": stat.st_mtime_ns,
            }
    return result


def restore_mtimes(repository, saved, current):
    restored = 0
    for name, state in current.items():
        previous = saved.get(name, {})
        if (
            previous.get("sha256") == state["sha256"]
            and previous.get("mode") == state["mode"]
        ):
            path = repository / name
            os.utime(path, ns=(path.stat().st_atime_ns, previous["mtime_ns"]))
            restored += 1
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--restore-mtimes", action="store_true")
    mode.add_argument("--capture-mtimes", action="store_true")
    args = parser.parse_args()
    repository = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    state_file = args.workspace / "target" / ".ci-source-state.json"
    if args.restore_mtimes:
        if not state_file.exists():
            print("No source state in cache; Cargo will check the fresh checkout.")
            return
        try:
            saved = json.loads(state_file.read_text())
        except (ValueError, OSError) as error:
            print(f"Ignoring unreadable source state: {error}", file=sys.stderr)
            return
        count = restore_mtimes(repository, saved, source_state(repository))
        print(f"Restored timestamps for {count} content-identical files/directories.")
    elif args.capture_mtimes:
        # Run AFTER successful checks, before actions/cache saves target. This
        # records the actual source state associated with the built artifacts,
        # including any tracked files modified by generators during the job.
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(source_state(repository)))
    else:
        paths = manifest_inputs(args.workspace, repository)
        print(f"fingerprint={fingerprint(paths, repository)}")


if __name__ == "__main__":
    main()
