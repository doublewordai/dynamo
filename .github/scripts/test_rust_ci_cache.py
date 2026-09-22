# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from pathlib import Path

from rust_ci_cache import fingerprint, manifest_inputs, restore_mtimes


class RustCacheTests(unittest.TestCase):
    def test_only_identical_contents_and_modes_restore_the_cached_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = {}
            current = {}
            for name in ("same.rs", "changed.rs", "executable.rs", "new.rs"):
                path = root / name
                path.write_text(name)
                os.utime(path, ns=(2_000_000_000, 2_000_000_000))
                current[name] = {"sha256": name, "mode": 0o100644}
                if name != "new.rs":
                    saved[name] = {**current[name], "mtime_ns": 1_000_000_000}
            current["changed.rs"]["sha256"] = "different contents"
            current["executable.rs"]["mode"] = 0o100755
            self.assertEqual(restore_mtimes(root, saved, current), 1)
            self.assertEqual((root / "same.rs").stat().st_mtime_ns, 1_000_000_000)
            for name in ("changed.rs", "executable.rs", "new.rs"):
                self.assertEqual((root / name).stat().st_mtime_ns, 2_000_000_000)

    def test_only_selected_lockfile_and_transitive_local_manifests_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            selected = root / "binding"
            for name in (
                "Cargo.toml",
                "Cargo.lock",
                "binding/Cargo.toml",
                "binding/Cargo.lock",
                "shared/Cargo.toml",
                "transitive/Cargo.toml",
                "unrelated/Cargo.toml",
                "unrelated/Cargo.lock",
                ".cargo/config.toml",
                "rust-toolchain.toml",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)

            def package(name, dependencies):
                return {
                    "manifest_path": str(root / name / "Cargo.toml"),
                    "dependencies": [{"path": str(root / d)} for d in dependencies],
                }

            def metadata(manifest):
                if manifest.parent == selected:
                    return {
                        "workspace_root": str(selected),
                        "packages": [package("binding", ["shared"])],
                    }
                return {
                    "workspace_root": str(root),
                    "packages": [
                        package("shared", ["transitive"]),
                        package("transitive", []),
                        package("unrelated", []),
                    ],
                }

            paths = manifest_inputs(selected, root, metadata)
            expected = {
                "Cargo.toml",
                "binding/Cargo.toml",
                "binding/Cargo.lock",
                "shared/Cargo.toml",
                "transitive/Cargo.toml",
                ".cargo/config.toml",
                "rust-toolchain.toml",
            }
            self.assertEqual({str(p.relative_to(root)) for p in paths}, expected)
            before = fingerprint(paths, root)
            for name in ("Cargo.lock", "unrelated/Cargo.lock", "unrelated/Cargo.toml"):
                (root / name).write_text("unrelated change")
            self.assertEqual(fingerprint(paths, root), before)
            for name in expected:
                with self.subTest(name=name):
                    path = root / name
                    original = path.read_text()
                    path.write_text("dependency change")
                    self.assertNotEqual(fingerprint(paths, root), before)
                    path.write_text(original)


if __name__ == "__main__":
    unittest.main()
