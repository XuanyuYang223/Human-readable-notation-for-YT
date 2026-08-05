"""Tests for permutation-based out-of-distribution data generation."""

from __future__ import annotations

import unittest

from yt_transformer.ood import (
    compact_shape,
    generate_length_stress_tableaux,
    stress_shape,
)


class OODGenerationTests(unittest.TestCase):
    def test_compact_shapes_have_the_requested_size(self) -> None:
        self.assertEqual(compact_shape(21), (5, 5, 5, 5, 1))
        self.assertEqual(compact_shape(50), (10, 10, 10, 10, 10))
        self.assertEqual(compact_shape(54), (11, 11, 11, 11, 10))
        for entries in (1, 21, 30, 40, 50, 54):
            shape = compact_shape(entries)
            self.assertEqual(sum(shape), entries)
            self.assertLessEqual(len(shape), 5)
            self.assertEqual(tuple(sorted(shape, reverse=True)), shape)
        self.assertEqual(stress_shape(50), (10, 10, 10, 10, 10))
        self.assertEqual(stress_shape(54), (8, 8, 8, 8, 8, 8, 6))

    def test_up_to_50_entries_are_true_unique_value_permutations(self) -> None:
        tableaux = generate_length_stress_tableaux(4, entries=50, seed=7)
        self.assertEqual(len(set(tableau.rows for tableau in tableaux)), 4)
        for tableau in tableaux:
            values = [value for row in tableau.rows for value in row]
            self.assertEqual(len(values), 50)
            self.assertEqual(set(values), set(range(1, 51)))

    def test_above_50_entries_repeats_only_known_vocabulary_values(self) -> None:
        tableau = generate_length_stress_tableaux(1, entries=54, seed=8)[0]
        values = [value for row in tableau.rows for value in row]
        self.assertEqual(len(values), 54)
        self.assertEqual(set(values), set(range(1, 51)))
        self.assertLess(len(set(values)), len(values))


if __name__ == "__main__":
    unittest.main()
