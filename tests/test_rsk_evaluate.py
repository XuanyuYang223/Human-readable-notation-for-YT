"""Controlled tests for held-out RSK checkpoint evaluation."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn

from yt_transformer.notation import Tableau, format_notation
from yt_transformer.rsk import format_permutation, rsk_insertion_tableau
from yt_transformer.rsk_data import RSKExample, generate_rsk_splits
from yt_transformer.rsk_evaluate import (
    RSKEvaluationMetrics,
    RSKEvaluationReport,
    evaluate_rsk_examples,
    main,
    rsk_test_examples_from_metadata,
)
from yt_transformer.tokenizer import HandmadeTokenizer, RSK_VOCAB


class ControlledRSKEvaluationModel(nn.Module):
    """Return perfect teacher logits and one configured greedy sequence."""

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
            raise AssertionError("controlled target length does not match the batch")
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
        del bos_id, eos_id, pad_id, max_new_tokens
        return self.generated_ids.to(source.device).unsqueeze(0).expand(
            source.size(0), -1
        )


class RSKMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)
        permutation = (3, 1, 4, 2)
        tableau = rsk_insertion_tableau(permutation)
        self.example = RSKExample(
            permutation=permutation,
            tableau=tableau,
            source=format_permutation(permutation),
            target=format_notation(tableau, "raw"),
        )
        self.target_ids = self.tokenizer.encode(self.example.target)

    def _evaluate(self, generated_text: str | None) -> RSKEvaluationReport:
        generated_ids = (
            self.target_ids[:-1]
            if generated_text is None
            else self.tokenizer.encode(generated_text)
        )
        model = ControlledRSKEvaluationModel(
            target_ids=self.target_ids,
            generated_ids=generated_ids,
            vocab_size=self.tokenizer.vocab_size,
        )
        return evaluate_rsk_examples(
            model,  # type: ignore[arg-type]
            self.tokenizer,
            (self.example,),
            batch_size=1,
        )

    def test_perfect_output_has_perfect_overall_and_length_metrics(self) -> None:
        report = self._evaluate(self.example.target)

        self.assertEqual(set(report.by_length), {4})
        for metrics in (report.overall, report.by_length[4]):
            self.assertEqual(metrics.examples, 1)
            self.assertLess(metrics.loss, 1e-6)
            self.assertEqual(metrics.token_accuracy, 1.0)
            self.assertEqual(metrics.exact_match, 1.0)
            self.assertEqual(metrics.semantic_accuracy, 1.0)
            self.assertEqual(metrics.invalid_output_rate, 0.0)
            self.assertEqual(metrics.shape_exact_match, 1.0)
            self.assertEqual(metrics.content_preservation, 1.0)

    def test_structural_metrics_distinguish_shape_content_and_semantics(self) -> None:
        cases = (
            # A different valid tableau with the same shape and content.
            (Tableau(((1, 3), (2, 4))), 1.0, 1.0),
            # The same content in a different shape.
            (Tableau(((1, 2, 3, 4),)), 0.0, 1.0),
            # The expected shape but altered content.
            (Tableau(((1, 2), (3, 3))), 1.0, 0.0),
        )
        for tableau, expected_shape, expected_content in cases:
            with self.subTest(tableau=tableau.rows):
                metrics = self._evaluate(format_notation(tableau, "raw")).overall
                self.assertEqual(metrics.exact_match, 0.0)
                self.assertEqual(metrics.semantic_accuracy, 0.0)
                self.assertEqual(metrics.invalid_output_rate, 0.0)
                self.assertEqual(metrics.shape_exact_match, expected_shape)
                self.assertEqual(metrics.content_preservation, expected_content)

        # Surface markers do not change the parsed tableau semantics.
        row_metrics = self._evaluate(
            format_notation(self.example.tableau, "row")
        ).overall
        self.assertEqual(row_metrics.exact_match, 0.0)
        self.assertEqual(row_metrics.semantic_accuracy, 1.0)
        self.assertEqual(row_metrics.invalid_output_rate, 0.0)

    def test_missing_eos_is_invalid_without_affecting_teacher_forced_metrics(self) -> None:
        metrics = self._evaluate(None).overall

        self.assertEqual(metrics.token_accuracy, 1.0)
        self.assertEqual(metrics.exact_match, 0.0)
        self.assertEqual(metrics.semantic_accuracy, 0.0)
        self.assertEqual(metrics.invalid_output_rate, 1.0)
        self.assertEqual(metrics.shape_exact_match, 0.0)
        self.assertEqual(metrics.content_preservation, 0.0)

    def test_rejects_empty_examples_and_invalid_batch_size(self) -> None:
        model = ControlledRSKEvaluationModel(
            target_ids=self.target_ids,
            generated_ids=self.target_ids,
            vocab_size=self.tokenizer.vocab_size,
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_rsk_examples(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                (),
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            evaluate_rsk_examples(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                (self.example,),
                batch_size=0,
            )


class RSKMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "num_permutations": 40,
            "min_length": 3,
            "max_length": 6,
            "split_ratios": [0.6, 0.2, 0.2],
            "seed": 17,
            "split_seed": 23,
        }
        self.metadata = {"training_config": self.config}

    def test_reconstructs_exact_group_disjoint_test_split(self) -> None:
        rebuilt = rsk_test_examples_from_metadata(self.metadata)
        expected = generate_rsk_splits(
            self.config["num_permutations"],
            seed=self.config["seed"],
            split_seed=self.config["split_seed"],
            min_length=self.config["min_length"],
            max_length=self.config["max_length"],
            split_ratios=tuple(self.config["split_ratios"]),
        )

        self.assertEqual(rebuilt, expected["test"])
        train_groups = {example.tableau_key for example in expected["train"]}
        val_groups = {example.tableau_key for example in expected["val"]}
        test_groups = {example.tableau_key for example in expected["test"]}
        self.assertFalse(train_groups & val_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(val_groups & test_groups)

    def test_rejects_incomplete_and_invalid_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_config"):
            rsk_test_examples_from_metadata({})

        for key in (
            "num_permutations",
            "min_length",
            "max_length",
            "split_ratios",
            "seed",
            "split_seed",
        ):
            invalid_config = dict(self.config)
            del invalid_config[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                rsk_test_examples_from_metadata(
                    {"training_config": invalid_config}
                )

        boolean_seed = {"training_config": {**self.config, "seed": True}}
        with self.assertRaisesRegex(ValueError, "'seed'"):
            rsk_test_examples_from_metadata(boolean_seed)
        too_few = {"training_config": {**self.config, "num_permutations": 2}}
        with self.assertRaisesRegex(ValueError, "num_permutations"):
            rsk_test_examples_from_metadata(too_few)
        invalid_ratios = {
            "training_config": {**self.config, "split_ratios": [0.8, 0.2, 0.0]}
        }
        with self.assertRaisesRegex(ValueError, "split_ratios"):
            rsk_test_examples_from_metadata(invalid_ratios)


class RSKCLItests(unittest.TestCase):
    def test_cli_rejects_checkpoint_from_another_direction(self) -> None:
        tokenizer = HandmadeTokenizer()
        with patch(
            "yt_transformer.rsk_evaluate.load_checkpoint",
            return_value=(object(), tokenizer, {"direction": "yt_to_human"}),
        ):
            with self.assertRaisesRegex(SystemExit, "perm_to_yt"):
                main(["--checkpoint", "wrong.pt", "--device", "cpu"])

    def test_cli_prints_json_report(self) -> None:
        tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)
        metrics = RSKEvaluationMetrics(1, 0.1, 0.9, 0.8, 0.7, 0.2, 0.6, 0.5)
        report = RSKEvaluationReport(metrics, {4: metrics})
        example = RSKExample(
            permutation=(1,),
            tableau=Tableau(((1,),)),
            source="[perm start] 1 [perm end]",
            target="[YT start] 1 [YT end]",
        )
        output = StringIO()
        with (
            patch(
                "yt_transformer.rsk_evaluate.load_checkpoint",
                return_value=(object(), tokenizer, {"direction": "perm_to_yt"}),
            ),
            patch(
                "yt_transformer.rsk_evaluate.rsk_test_examples_from_metadata",
                return_value=(example,),
            ),
            patch(
                "yt_transformer.rsk_evaluate.evaluate_rsk_examples",
                return_value=report,
            ),
            redirect_stdout(output),
        ):
            main(
                [
                    "--checkpoint",
                    "model.pt",
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--limit",
                    "1",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["direction"], "perm_to_yt")
        self.assertEqual(payload["overall"]["examples"], 1)
        self.assertEqual(payload["by_length"]["4"]["shape_exact_match"], 0.6)


if __name__ == "__main__":
    unittest.main()
