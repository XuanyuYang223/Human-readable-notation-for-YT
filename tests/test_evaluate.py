"""Fast tests for held-out reconstruction and evaluation metrics."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn

from yt_transformer.data import (
    generate_translation_splits,
    make_translation_examples,
)
from yt_transformer.evaluate import (
    evaluate_examples,
    evaluate_round_trip,
    test_examples_from_metadata as rebuild_test_examples,
)
from yt_transformer.notation import Tableau
from yt_transformer.tokenizer import HandmadeTokenizer


class ControlledEvaluationModel(nn.Module):
    """Teacher-forces perfect logits while returning configured greedy IDs."""

    def __init__(
        self,
        *,
        target_ids: list[int],
        generated_ids: list[int],
        vocab_size: int,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.target_ids = torch.tensor(target_ids, dtype=torch.long)
        self.generated_ids = torch.tensor(generated_ids, dtype=torch.long)
        self.vocab_size = vocab_size

    def forward(self, source: Tensor, target_input: Tensor) -> Tensor:
        batch_size, sequence_length = target_input.shape
        expected_next = self.target_ids[1 : sequence_length + 1].to(source.device)
        if expected_next.numel() != sequence_length:
            raise AssertionError("controlled target does not match teacher-forced length")
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -20.0,
            dtype=torch.float32,
            device=source.device,
        )
        logits.scatter_(
            2,
            expected_next.view(1, -1, 1).expand(batch_size, -1, -1),
            20.0,
        )
        return logits

    def greedy_decode(
        self,
        source: Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int,
    ) -> Tensor:
        return self.generated_ids.to(source.device).unsqueeze(0).expand(source.size(0), -1)


class MetadataReconstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training_config = {
            "num_tableaux": 12,
            "seed": 29,
            "split_ratios": [0.5, 0.25, 0.25],
            "human_kinds": ["row", "col", "coord"],
            "max_rows": 3,
            "max_columns": 4,
            "max_cells": 8,
        }
        self.metadata = {"training_config": self.training_config}

    def test_rebuilds_exact_held_out_split_for_each_direction(self) -> None:
        for direction in ("yt_to_human", "human_to_yt"):
            with self.subTest(direction=direction):
                rebuilt = rebuild_test_examples(self.metadata, direction)
                expected = generate_translation_splits(
                    self.training_config["num_tableaux"],
                    seed=self.training_config["seed"],
                    split_seed=self.training_config["seed"] + 1,
                    split_ratios=tuple(self.training_config["split_ratios"]),
                    directions=(direction,),
                    human_kinds=tuple(self.training_config["human_kinds"]),
                    max_rows=self.training_config["max_rows"],
                    max_columns=self.training_config["max_columns"],
                    max_cells=self.training_config["max_cells"],
                )["test"]

                self.assertEqual(rebuilt, expected)
                self.assertTrue(rebuilt)
                self.assertTrue(all(example.direction == direction for example in rebuilt))
                self.assertEqual(
                    {example.human_kind for example in rebuilt},
                    {"row", "col", "coord"},
                )

    def test_rejects_incomplete_or_invalid_training_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_config"):
            rebuild_test_examples({}, "yt_to_human")

        invalid = {"training_config": {**self.training_config, "human_kinds": ["grid"]}}
        with self.assertRaisesRegex(ValueError, "unknown human notation kind"):
            rebuild_test_examples(invalid, "yt_to_human")

        boolean_seed = {"training_config": {**self.training_config, "seed": True}}
        with self.assertRaisesRegex(ValueError, "invalid 'seed'"):
            rebuild_test_examples(boolean_seed, "yt_to_human")

        with self.assertRaisesRegex(ValueError, "yt-rsk-evaluate"):
            rebuild_test_examples(self.metadata, "perm_to_yt")


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HandmadeTokenizer()
        self.tableau = Tableau(((2, 3, 5), (1, 4)))
        self.example = make_translation_examples(
            self.tableau,
            directions=("yt_to_human",),
            human_kinds=("coord",),
        )[0]
        self.target_ids = self.tokenizer.encode(self.example.target)

    def _model(self, generated_ids: list[int]) -> ControlledEvaluationModel:
        return ControlledEvaluationModel(
            target_ids=self.target_ids,
            generated_ids=generated_ids,
            vocab_size=self.tokenizer.vocab_size,
        )

    def test_perfect_controlled_output_has_perfect_metrics(self) -> None:
        metrics = evaluate_examples(
            self._model(self.target_ids),  # type: ignore[arg-type]
            self.tokenizer,
            (self.example,),
            batch_size=1,
        )

        self.assertEqual(metrics.examples, 1)
        self.assertLess(metrics.loss, 1e-6)
        self.assertEqual(metrics.token_accuracy, 1.0)
        self.assertEqual(metrics.exact_match, 1.0)
        self.assertEqual(metrics.semantic_accuracy, 1.0)
        self.assertEqual(metrics.invalid_output_rate, 0.0)

    def test_missing_eos_and_invalid_notation_count_as_invalid(self) -> None:
        invalid_generations = (
            self.target_ids[:-1],
            [
                self.tokenizer.bos_id,
                self.tokenizer.token_id("n1"),
                self.tokenizer.eos_id,
            ],
        )
        for generated_ids in invalid_generations:
            with self.subTest(generated_ids=generated_ids):
                metrics = evaluate_examples(
                    self._model(generated_ids),  # type: ignore[arg-type]
                    self.tokenizer,
                    (self.example,),
                    batch_size=1,
                )
                # Teacher-forced metrics remain perfect; greedy validity metrics
                # independently expose the malformed generated sequence.
                self.assertEqual(metrics.token_accuracy, 1.0)
                self.assertEqual(metrics.exact_match, 0.0)
                self.assertEqual(metrics.semantic_accuracy, 0.0)
                self.assertEqual(metrics.invalid_output_rate, 1.0)

    def test_valid_wrong_kind_is_not_invalid_but_is_semantically_wrong(self) -> None:
        col_example = make_translation_examples(
            self.tableau,
            directions=("yt_to_human",),
            human_kinds=("col",),
        )[0]
        generated = self.tokenizer.encode(col_example.target)
        metrics = evaluate_examples(
            self._model(generated),  # type: ignore[arg-type]
            self.tokenizer,
            (self.example,),
            batch_size=1,
        )

        self.assertEqual(metrics.exact_match, 0.0)
        self.assertEqual(metrics.semantic_accuracy, 0.0)
        self.assertEqual(metrics.invalid_output_rate, 0.0)

    def test_rejects_empty_examples_and_nonpositive_batch_size(self) -> None:
        model = self._model(self.target_ids)
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_examples(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                (),
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            evaluate_examples(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                (self.example,),
                batch_size=0,
            )

    def test_coordinate_only_round_trip_uses_requested_common_style(self) -> None:
        raw = "[YT start] 2 3 5 | 1 4 [YT end]"
        coord_example = self.example
        forward = self._model(self.tokenizer.encode(coord_example.target))
        reverse = self._model(self.tokenizer.encode(raw))

        metrics = evaluate_round_trip(
            forward,  # type: ignore[arg-type]
            reverse,  # type: ignore[arg-type]
            self.tokenizer,
            (coord_example,),
            limit_tableaux=1,
            human_kinds=("coord",),
        )

        self.assertEqual(metrics["attempts"], 1)
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["invalid_pipeline_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
