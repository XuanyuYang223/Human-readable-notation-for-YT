"""Tests for canonical permutations and standard RSK row insertion."""

from __future__ import annotations

from itertools import permutations
import unittest

from yt_transformer.notation import Tableau
from yt_transformer.rsk import (
    format_permutation,
    parse_permutation,
    rsk_insertion_tableau,
    validate_permutation,
)


class PermutationNotationTests(unittest.TestCase):
    def test_validation_normalizes_finite_integer_iterables(self) -> None:
        self.assertEqual(validate_permutation([3, 1, 4, 2]), (3, 1, 4, 2))
        self.assertEqual(validate_permutation(value for value in (2, 1)), (2, 1))

    def test_format_and_parse_canonical_surface(self) -> None:
        permutation = (3, 1, 4, 2)
        text = "[perm start] 3 1 4 2 [perm end]"

        self.assertEqual(format_permutation(permutation), text)
        self.assertEqual(parse_permutation(text), permutation)
        self.assertEqual(format_permutation(parse_permutation(text)), text)

    def test_rejects_invalid_permutation_values(self) -> None:
        invalid = (
            (),
            (1, 1),
            (2,),
            (0,),
            (1, 3),
            (True,),
            (1.0,),
            tuple(range(1, 52)),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_permutation(values)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            validate_permutation("1 2")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            validate_permutation(123)  # type: ignore[arg-type]

    def test_rejects_noncanonical_or_invalid_surface_text(self) -> None:
        invalid = (
            "[perm start] [perm end]",
            "[perm start] 3  1 4 2 [perm end]",
            "[perm start] 03 1 4 2 [perm end]",
            "[perm start] 3 1 4 2  [perm end]",
            " [perm start] 3 1 4 2 [perm end]",
            "[perm start] 3 1 4 2 [YT end]",
            "[YT start] 3 1 4 2 [perm end]",
            "[perm start] 1 1 [perm end]",
            "[perm start] 1 3 [perm end]",
            "[perm start] 0 [perm end]",
            "[perm start] ١ [perm end]",
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_permutation(text)

        with self.assertRaises(TypeError):
            parse_permutation(123)  # type: ignore[arg-type]


class RSKInsertionTests(unittest.TestCase):
    def test_known_row_insertion_examples(self) -> None:
        self.assertEqual(
            rsk_insertion_tableau((1, 2, 3, 4)),
            Tableau(((1, 2, 3, 4),)),
        )
        self.assertEqual(
            rsk_insertion_tableau((4, 3, 2, 1)),
            Tableau(((1,), (2,), (3,), (4,))),
        )
        self.assertEqual(
            rsk_insertion_tableau((3, 1, 4, 2)),
            Tableau(((1, 2), (3, 4))),
        )

    @staticmethod
    def _longest_subsequence_length(
        values: tuple[int, ...], *, increasing: bool
    ) -> int:
        lengths = [1] * len(values)
        for right in range(len(values)):
            for left in range(right):
                ordered = values[left] < values[right]
                if ordered == increasing:
                    lengths[right] = max(lengths[right], lengths[left] + 1)
        return max(lengths)

    def test_exhaustive_small_outputs_are_standard_and_obey_schensted(self) -> None:
        for size in range(1, 7):
            for values in permutations(range(1, size + 1)):
                with self.subTest(permutation=values):
                    tableau = rsk_insertion_tableau(values)
                    flattened = tuple(value for row in tableau.rows for value in row)
                    self.assertEqual(set(flattened), set(range(1, size + 1)))
                    self.assertTrue(
                        all(
                            left < right
                            for row in tableau.rows
                            for left, right in zip(row, row[1:])
                        )
                    )
                    self.assertTrue(
                        all(
                            tableau.rows[row_index - 1][column_index]
                            < tableau.rows[row_index][column_index]
                            for row_index in range(1, len(tableau.rows))
                            for column_index in range(len(tableau.rows[row_index]))
                        )
                    )
                    self.assertEqual(
                        len(tableau.rows[0]),
                        self._longest_subsequence_length(values, increasing=True),
                    )
                    self.assertEqual(
                        len(tableau.rows),
                        self._longest_subsequence_length(values, increasing=False),
                    )


if __name__ == "__main__":
    unittest.main()
