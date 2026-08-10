"""Tests for permutation-based out-of-distribution data generation."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from yt_transformer.evaluate import EvaluationMetrics
from yt_transformer.ood import (
    compact_shape,
    evaluate_ood_lengths,
    generate_length_stress_tableaux,
    stress_shape,
)
from yt_transformer.tokenizer import HandmadeTokenizer


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

    def test_ood_evaluates_only_checkpoint_style_intersection(self) -> None:
        tokenizer = HandmadeTokenizer()
        forward_metadata = {
            "direction": "yt_to_human",
            "training_config": {"human_kinds": ["row", "coord"], "max_cells": 20},
        }
        reverse_metadata = {
            "direction": "human_to_yt",
            "training_config": {"human_kinds": ["col", "coord"], "max_cells": 20},
        }
        evaluated_styles: list[str] = []

        def fake_evaluate(model, tokenizer, examples, *, batch_size):
            del model, tokenizer, batch_size
            evaluated_styles.extend(example.human_kind for example in examples)
            return EvaluationMetrics(1, 0.0, 1.0, 1.0, 1.0, 0.0)

        with (
            patch(
                "yt_transformer.ood.load_checkpoint",
                side_effect=(
                    (object(), tokenizer, forward_metadata),
                    (object(), tokenizer, reverse_metadata),
                ),
            ),
            patch("yt_transformer.ood.evaluate_examples", side_effect=fake_evaluate),
        ):
            report = evaluate_ood_lengths(
                forward_checkpoint=Path("forward.pt"),
                reverse_checkpoint=Path("reverse.pt"),
                entries_values=(1,),
                samples=1,
                batch_size=1,
                seed=5,
                device_name="cpu",
            )

        metrics = report["cases"]["1"]["metrics"]
        self.assertEqual(set(metrics["yt_to_human"]), {"coord"})
        self.assertEqual(set(metrics["human_to_yt"]), {"coord"})
        self.assertEqual(evaluated_styles, ["coord", "coord"])


if __name__ == "__main__":
    unittest.main()
