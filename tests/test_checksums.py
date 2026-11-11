"""Integrity regression tests. Uses temporary files only; no research datasets."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("checksums", Path(__file__).resolve().parents[1] / "make_checksums.py")
checksums = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checksums)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        Path("outputs/nested").mkdir(parents=True)
        Path("outputs/result with spaces.txt").write_text("measured result\n")
        Path("outputs/nested/config.json").write_text('{"seed": 340}')

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    def test_roundtrip_and_determinism(self):
        self.assertEqual(checksums.generate(), 2)
        first = Path("CHECKSUMS.txt").read_bytes()
        self.assertEqual(checksums.verify("CHECKSUMS.txt"), 2)
        checksums.generate()
        self.assertEqual(first, Path("CHECKSUMS.txt").read_bytes())

    def test_tampering_is_detected(self):
        checksums.generate()
        Path("outputs/nested/config.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            checksums.verify("CHECKSUMS.txt")

    def test_missing_file_is_detected(self):
        checksums.generate()
        Path("outputs/nested/config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing artifact"):
            checksums.verify("CHECKSUMS.txt")

    def test_empty_outputs_do_not_overwrite_manifest(self):
        Path("empty").mkdir()
        Path("CHECKSUMS.txt").write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "no artifacts"):
            checksums.generate("empty")
        self.assertEqual(Path("CHECKSUMS.txt").read_text(), "preserve me")

    def test_missing_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            checksums.generate("absent")

    def test_self_and_cache_are_excluded(self):
        Path("outputs/__pycache__").mkdir()
        Path("outputs/__pycache__/cache.pyc").write_bytes(b"cache")
        self.assertEqual(checksums.generate("outputs", "outputs/sums.txt"), 2)
        self.assertEqual(checksums.generate("outputs", "outputs/sums.txt"), 2)
        self.assertEqual(checksums.verify("outputs/sums.txt"), 2)

    def test_unsafe_malformed_and_duplicate_entries_are_rejected(self):
        for text in ("0" * 64 + "  ../outside.txt\n", "invalid\n", ""):
            Path("bad.txt").write_text(text)
            with self.assertRaises(ValueError):
                checksums.verify("bad.txt")
        checksums.generate()
        text = Path("CHECKSUMS.txt").read_text()
        Path("bad.txt").write_text(text + text)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checksums.verify("bad.txt")


if __name__ == "__main__":
    unittest.main()
