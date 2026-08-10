"""Tests for deterministic permutation-to-RSK conversion."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from yt_transformer.rsk_convert import convert_permutation, main


class RSKConvertTests(unittest.TestCase):
    def test_known_conversion(self) -> None:
        self.assertEqual(
            convert_permutation("[perm start] 3 5 1 4 2 [perm end]"),
            "[YT start] 1 2 | 3 4 | 5 [YT end]",
        )

    def test_rejects_non_permutation(self) -> None:
        with self.assertRaises(ValueError):
            convert_permutation("[perm start] 1 1 [perm end]")

    def test_cli_prints_result(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            main(["[perm start] 3 1 4 2 [perm end]"])
        self.assertEqual(
            output.getvalue().strip(),
            "[YT start] 1 2 | 3 4 [YT end]",
        )


if __name__ == "__main__":
    unittest.main()
