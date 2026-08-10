"""Fast CPU tests for the independent RSK training pipeline."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from yt_transformer.checkpoint import checkpoint_direction, load_checkpoint
from yt_transformer.rsk_train import (
    RSKTrainingConfig,
    build_parser,
    build_rsk_loaders,
    train_rsk,
)
from yt_transformer.tokenizer import HandmadeTokenizer, RSK_VOCAB


class RSKTrainingConfigTests(unittest.TestCase):
    def test_recommended_defaults_and_explicit_split_seed(self) -> None:
        config = RSKTrainingConfig()
        self.assertEqual(config.num_permutations, 40_000)
        self.assertEqual((config.min_length, config.max_length), (1, 20))
        self.assertEqual(config.epochs, 40)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.nhead, 8)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.dim_feedforward, 1_024)
        self.assertEqual(config.max_seq_len, 128)
        self.assertEqual(config.split_seed, config.seed + 1)

        args = build_parser().parse_args([])
        self.assertEqual(args.num_permutations, 40_000)
        self.assertEqual((args.min_length, args.max_length), (1, 20))
        self.assertEqual(args.epochs, 40)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.d_model, 256)
        self.assertEqual(args.nhead, 8)
        self.assertEqual(args.num_layers, 4)
        self.assertEqual(args.dim_feedforward, 1_024)
        self.assertEqual(args.max_seq_len, 128)

    def test_checkpoint_metadata_reconstructs_the_exact_split(self) -> None:
        config = RSKTrainingConfig(
            num_permutations=6,
            min_length=3,
            max_length=3,
            split_ratios=(0.5, 0.25, 0.25),
            seed=11,
            split_seed=29,
            max_seq_len=15,
        )

        values = config.checkpoint_dict()

        self.assertEqual(values["num_permutations"], 6)
        self.assertEqual(values["min_length"], 3)
        self.assertEqual(values["max_length"], 3)
        self.assertEqual(values["split_ratios"], [0.5, 0.25, 0.25])
        self.assertEqual(values["seed"], 11)
        self.assertEqual(values["split_seed"], 29)
        self.assertIsInstance(values["split_ratios"], list)

    def test_rejects_invalid_core_settings(self) -> None:
        invalid_overrides = (
            {"num_permutations": 2},
            {"num_permutations": True},
            {"min_length": 0},
            {"min_length": 4, "max_length": 3},
            {"max_length": 51, "max_seq_len": 256},
            {"split_ratios": (0.8, 0.2)},
            {"split_ratios": (0.8, 0.0, 0.2)},
            {"split_ratios": (0.8, float("nan"), 0.2)},
            {"split_seed": True},
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
            {"max_seq_len": 82},
            {"tie_embeddings": 1},
            {"seed": -1},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                RSKTrainingConfig(**overrides)  # type: ignore[arg-type]


class RSKLoaderAndTrainingTests(unittest.TestCase):
    @staticmethod
    def _tiny_config(*, epochs: int = 1) -> RSKTrainingConfig:
        return RSKTrainingConfig(
            num_permutations=6,
            min_length=3,
            max_length=3,
            split_ratios=(0.5, 0.25, 0.25),
            split_seed=31,
            epochs=epochs,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            patience=0,
            val_exact_limit=8,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            max_seq_len=15,
            seed=17,
        )

    def test_loaders_use_all_examples_and_the_rsk_vocabulary(self) -> None:
        config = self._tiny_config()
        tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)

        train_loader, val_loader, max_new_tokens, counts = build_rsk_loaders(
            config, tokenizer=tokenizer
        )

        self.assertEqual(sum(counts.values()), config.num_permutations)
        self.assertEqual(len(train_loader.dataset), counts["train"])
        self.assertEqual(len(val_loader.dataset), counts["val"])
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertGreater(max_new_tokens, 0)
        self.assertLessEqual(max_new_tokens, config.max_seq_len - 1)
        batch = next(iter(train_loader))
        source_ids = batch["source_ids"]
        self.assertIsInstance(source_ids, torch.Tensor)
        self.assertTrue(torch.all(source_ids[:, 1].eq(tokenizer.token_id("x9"))).item())

        val_examples = val_loader.dataset.examples
        available_lengths = sorted({example.length for example in val_examples})
        self.assertEqual(
            [example.length for example in val_examples[: len(available_lengths)]],
            available_lengths,
        )

        with self.assertRaisesRegex(ValueError, "72-token RSK vocabulary"):
            build_rsk_loaders(config, tokenizer=HandmadeTokenizer())

    def test_one_epoch_saves_a_reloadable_best_checkpoint(self) -> None:
        config = self._tiny_config()
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output_dir = Path(directory)
            result = train_rsk(
                output_dir=output_dir,
                device=torch.device("cpu"),
                config=config,
            )

            self.assertEqual(result.direction, "perm_to_yt")
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(result.checkpoint_path, output_dir / "perm_to_yt.pt")
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertTrue(math.isfinite(result.best_metrics.loss))
            self.assertIsNotNone(result.best_metrics.exact_match)

            model, tokenizer, metadata = load_checkpoint(result.checkpoint_path)
            self.assertFalse(model.training)
            self.assertEqual(checkpoint_direction(metadata), "perm_to_yt")
            self.assertEqual(tokenizer.vocab, RSK_VOCAB)
            self.assertEqual(model.config.src_vocab_size, 72)
            self.assertEqual(model.config.d_model, config.d_model)
            self.assertEqual(metadata["epoch"], 1)
            training_config = metadata["training_config"]
            for key in (
                "num_permutations",
                "min_length",
                "max_length",
                "split_ratios",
                "seed",
                "split_seed",
            ):
                self.assertIn(key, training_config)
            self.assertEqual(training_config["split_seed"], config.split_seed)


if __name__ == "__main__":
    unittest.main()
