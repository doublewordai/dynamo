# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rust_ci_scope import changed_paths, needs_rust


class RustScopeTests(unittest.TestCase):
    def test_documentation_and_sboms(self):
        self.assertFalse(
            needs_rust(
                [
                    "AGENTS.md",
                    "CLAUDE.md",
                    "README.md",
                    "README.zh-CN.md",
                    "CONTRIBUTING.md",
                    "CONTRIBUTORS.md",
                    "CODE_OF_CONDUCT.md",
                    "SECURITY.md",
                    "DCO.md",
                    "docs/page.md",
                    "fern/pages/page.mdx",
                    "container/compliance/base_sboms/new.json",
                ]
            )
        )

    def test_shared_inputs_and_unknown_paths_run(self):
        for path in (
            "lib/runtime/src/lib.rs",
            "lib/runtime/docs/rayon-tokio-strategy.md",
            "lib/llm/tests/data/prompt.txt",
            "lib/bindings/python/Cargo.lock",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            ".github/actions/rust-ci-cache/action.yml",
            ".github/workflows/fork-rust-tests.yml",
            ".github/scripts/rust_ci_scope.py",
            "container/compliance/policy/validate.py",
            "new-directory/unknown-input",
            "README.md.rs",
        ):
            with self.subTest(path=path):
                self.assertTrue(needs_rust(["docs/page.md", path]))

    def test_git_diff_keeps_deleted_and_renamed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", directory, *args], stderr=subprocess.DEVNULL
                )

            git("init", "-q")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci@example.invalid")
            (root / "lib").mkdir()
            (root / "docs").mkdir()
            old = "lib/fixture with\nnewline.txt"
            (root / old).write_text("a Rust fixture\n")
            (root / "lib/deleted.rs").write_text("// deleted\n")
            git("add", ".")
            git("commit", "-qm", "base")
            (root / old).rename(root / "docs/moved.txt")
            (root / "lib/deleted.rs").unlink()
            git("add", "-A")
            git("commit", "-qm", "move and delete")
            paths = changed_paths("HEAD^", "HEAD", cwd=directory)
            self.assertEqual(set(paths), {old, "lib/deleted.rs", "docs/moved.txt"})
            self.assertTrue(needs_rust(paths))

    def test_missing_base_runs_tests(self):
        script = Path(__file__).with_name("rust_ci_scope.py")
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "-q", directory], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "-c",
                    "user.name=CI test",
                    "-c",
                    "user.email=ci@example.invalid",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "base",
                ],
                check=True,
            )
            result = subprocess.run(
                [sys.executable, str(script), "--base", "missing-base"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(result.stdout, "rust=true\n")


if __name__ == "__main__":
    unittest.main()
