# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip the fork Rust suite only for known documentation and SBOM-only PRs."""

import argparse
import subprocess
import sys

# Keep this allowlist deliberately small. In particular, Markdown and other
# fixtures under lib/ can be included by Rust code and must run the tests.
NON_RUST_PREFIXES = ("docs/", "fern/", "container/compliance/base_sboms/")
NON_RUST_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "README.zh-CN.md",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "DCO.md",
}


def needs_rust(paths):
    return any(
        path not in NON_RUST_FILES and not path.startswith(NON_RUST_PREFIXES)
        for path in paths
    )


def changed_paths(base, head, cwd=None):
    # No rename detection: both the old and new paths must be considered.
    # NUL separation preserves spaces and newlines in filenames. This avoids
    # the PR-files API's truncation limit on large upstream merges.
    output = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"],
        cwd=cwd,
    )
    return output.decode("utf-8", errors="surrogateescape").split("\0")[:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD^1")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        selected = needs_rust(changed_paths(args.base, args.head))
    except subprocess.CalledProcessError:
        # Missing comparison history cannot turn the required check green
        # without testing. Run the complete narrow suite instead.
        print("Cannot determine changed files; running Rust tests.", file=sys.stderr)
        selected = True
    print(f"rust={str(selected).lower()}")


if __name__ == "__main__":
    main()
