"""Fast CPU tests for the training loop (stdlib unittest only)."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from yt_transformer.checkpoint import checkpoint_direction, load_checkpoint
from yt_transformer.data import (
    TranslationDataset,
    make_collate_fn,
    make_translation_examples,
)
from yt_transformer.model import ModelConfig, Seq2SeqTransformer
from yt_transformer.notation import Tableau
from yt_transformer.tokenizer import HandmadeTokenizer
from yt_transformer.train import (
    TrainingConfig,
    build_parser,
    run_training_epoch,
    train_direction,
)


class TrainingConfigTest(unittest.TestCase):
    def test_defaults_include_coordinate_notation_and_longer_sequences(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.human_kinds, ("row", "col", "coord"))
        self.assertEqual(config.max_seq_len, 256)

        args = build_parser().parse_args([])
        self.assertEqual(tuple(args.human_kinds), ("row", "col", "coord"))
        self.assertEqual(args.max_seq_len, 256)

    def test_rejects_invalid_core_settings(self) -> None:
        invalid_overrides = (
            {"num_tableaux": 2},
            {"num_tableaux": True},
            {"split_ratios": (0.0, 0.0, 0.0)},
            {"split_ratios": (0.8, float("nan"), 0.2)},
            {"human_kinds": ()},
            {"human_kinds": ("row", "row")},
            {
                "human_kinds": ("coord",),
                "max_rows": 51,
                "max_columns": 1,
                "max_cells": 51,
            },
            {
                "human_kinds": ("coord",),
                "max_rows": 1,
                "max_columns": 51,
                "max_cells": 51,
            },
            {"max_rows": 2, "max_columns": 2, "max_cells": 5},
            {"max_rows": True},
            {"epochs": 0},
            {"batch_size": 0},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
            {"weight_decay": -0.1},
            {"grad_clip": 0.0},
            {"patience": -1},
            {"val_exact_limit": 0},
            {"d_model": 7, "nhead": 2},
            {"num_layers": 0},
            {"dim_feedforward": 0},
            {"dropout": 1.0},
            {"max_seq_len": 3},
            {"seed": -1},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    TrainingConfig(**overrides)

    def test_checkpoint_dict_contains_only_portable_collections(self) -> None:
        config = TrainingConfig(
            num_tableaux=4,
            split_ratios=(0.5, 0.25, 0.25),
            human_kinds=("coord",),
        )

        values = config.checkpoint_dict()

        self.assertEqual(values["split_ratios"], [0.5, 0.25, 0.25])
        self.assertEqual(values["human_kinds"], ["coord"])
        self.assertIsInstance(values["split_ratios"], list)
        self.assertIsInstance(values["human_kinds"], list)


class TrainingLoopTest(unittest.TestCase):
    @staticmethod
    def _tiny_model(tokenizer: HandmadeTokenizer) -> Seq2SeqTransformer:
        return Seq2SeqTransformer(
            ModelConfig(
                src_vocab_size=tokenizer.vocab_size,
                tgt_vocab_size=tokenizer.vocab_size,
                d_model=8,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=16,
                dropout=0.0,
                max_seq_len=24,
                pad_id=tokenizer.pad_id,
            )
        )

    def test_single_training_epoch_returns_finite_loss(self) -> None:
        torch.manual_seed(11)
        tokenizer = HandmadeTokenizer()
        tableaux = (
            Tableau(((1, 2), (3,))),
            Tableau(((4,),)),
            Tableau(((5, 6, 7),)),
            Tableau(((8, 9), (10, 11))),
        )
        examples = tuple(
            example
            for tableau in tableaux
            for example in make_translation_examples(
                tableau,
                directions=("yt_to_human",),
                human_kinds=("row",),
            )
        )
        dataset = TranslationDataset(examples, tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(tokenizer.pad_id),
        )
        model = self._tiny_model(tokenizer)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        metrics = run_training_epoch(
            model,
            loader,
            optimizer,
            device=torch.device("cpu"),
            pad_id=tokenizer.pad_id,
            grad_clip=1.0,
        )

        self.assertTrue(math.isfinite(metrics.loss))
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreaterEqual(metrics.token_accuracy, 0.0)
        self.assertLessEqual(metrics.token_accuracy, 1.0)
        self.assertIsNone(metrics.exact_match)

    def test_train_direction_saves_a_reloadable_checkpoint(self) -> None:
        config = TrainingConfig(
            num_tableaux=4,
            split_ratios=(0.5, 0.25, 0.25),
            human_kinds=("row",),
            max_rows=1,
            max_columns=1,
            max_cells=1,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            patience=0,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            max_seq_len=12,
            seed=17,
        )

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output_dir = Path(directory)
            result = train_direction(
                "yt_to_human",
                output_dir=output_dir,
                device=torch.device("cpu"),
                config=config,
            )

            self.assertEqual(result.direction, "yt_to_human")
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(result.checkpoint_path, output_dir / "yt_to_human.pt")
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertTrue(math.isfinite(result.best_metrics.loss))
            self.assertIsNotNone(result.best_metrics.exact_match)

            model, tokenizer, metadata = load_checkpoint(
                result.checkpoint_path,
                device="cpu",
            )
            self.assertFalse(model.training)
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            self.assertEqual(model.config.d_model, config.d_model)
            self.assertEqual(model.config.pad_id, tokenizer.pad_id)
            self.assertEqual(model.supported_human_kinds, ("row",))
            self.assertEqual(checkpoint_direction(metadata), "yt_to_human")
            self.assertEqual(metadata["epoch"], 1)
            self.assertNotIn("model_state_dict", metadata)
            self.assertEqual(
                metadata["training_config"]["human_kinds"],
                ["row"],
            )

    def test_rsk_direction_uses_its_dedicated_training_pipeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "yt-rsk-train"):
            train_direction(
                "perm_to_yt",
                output_dir=Path("unused"),
                device=torch.device("cpu"),
                config=TrainingConfig(),
            )


if __name__ == "__main__":
    unittest.main()
