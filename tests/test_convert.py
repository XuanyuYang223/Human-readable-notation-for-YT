"""Tests for the deterministic notation-conversion reference implementation."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from yt_transformer.convert import convert_notation, main


class ConvertNotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = "[YT start] 2 3 5 | 1 4 [YT end]"
        self.row = "[YT row start] 2 3 5 | 1 4 [YT row end]"
        self.col = "[YT col start] 2 1 | 3 4 | 5 [YT col end]"

    def test_screenshot_example_converts_between_every_surface_kind(self) -> None:
        expected = {"raw": self.raw, "row": self.row, "col": self.col}
        for source in expected.values():
            for target, target_text in expected.items():
                with self.subTest(source=source, target=target):
                    self.assertEqual(convert_notation(source, target), target_text)

    def test_cli_main_prints_one_canonical_conversion(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main([self.raw, "--to", "col"])
        self.assertEqual(output.getvalue(), self.col + "\n")

    def test_invalid_source_and_target_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            convert_notation("[YT start] 2  3 [YT end]", "row")
        with self.assertRaises(ValueError):
            convert_notation(self.raw, "diagonal")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
