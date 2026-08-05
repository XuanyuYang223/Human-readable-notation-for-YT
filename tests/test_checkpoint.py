"""Tests for versioned, self-describing model checkpoints."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from yt_transformer.checkpoint import (
    checkpoint_direction,
    load_checkpoint,
    save_checkpoint,
)
from yt_transformer.model import ModelConfig, Seq2SeqTransformer
from yt_transformer.tokenizer import HandmadeTokenizer


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.tokenizer = HandmadeTokenizer()
        self.config = ModelConfig(
            src_vocab_size=self.tokenizer.vocab_size,
            tgt_vocab_size=self.tokenizer.vocab_size,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            max_seq_len=32,
            pad_id=self.tokenizer.pad_id,
        )
        self.model = Seq2SeqTransformer(self.config)

    def _save(self, path: Path, *, direction: str = "yt_to_human") -> None:
        save_checkpoint(
            path,
            model=self.model,
            tokenizer=self.tokenizer,
            direction=direction,  # type: ignore[arg-type]
            epoch=3,
            metrics={"loss": 0.125, "exact_match": 0.75},
            training_config={"seed": 17, "batch_size": 8},
        )

    def test_save_and_reload_preserves_model_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "forward.pt"
            self._save(path)

            loaded_model, loaded_tokenizer, payload = load_checkpoint(path)

            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            self.assertFalse(loaded_model.training)
            self.assertEqual(loaded_model.config, self.config)
            self.assertEqual(loaded_tokenizer.vocab, self.tokenizer.vocab)
            self.assertEqual(checkpoint_direction(payload), "yt_to_human")
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["metrics"]["loss"], 0.125)
            self.assertEqual(payload["training_config"]["seed"], 17)

            expected_state = self.model.state_dict()
            actual_state = loaded_model.state_dict()
            self.assertEqual(expected_state.keys(), actual_state.keys())
            for name in expected_state:
                with self.subTest(parameter=name):
                    self.assertTrue(torch.equal(expected_state[name], actual_state[name]))

    def test_both_checkpoint_directions_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for direction in ("yt_to_human", "human_to_yt"):
                with self.subTest(direction=direction):
                    path = Path(directory) / f"{direction}.pt"
                    self._save(path, direction=direction)
                    _, _, payload = load_checkpoint(path, device="cpu")
                    self.assertEqual(checkpoint_direction(payload), direction)

    def test_invalid_direction_is_rejected_when_saving_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direction.pt"
            with self.assertRaisesRegex(ValueError, "invalid checkpoint direction"):
                self._save(path, direction="sideways")

            self._save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["direction"] = "sideways"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "invalid checkpoint direction"):
                load_checkpoint(path)

        with self.assertRaisesRegex(ValueError, "invalid checkpoint direction"):
            checkpoint_direction({"direction": "sideways"})

    def test_mismatched_tokenizer_vocabulary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-vocab.pt"
            self._save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["tokenizer_vocab"] = [*payload["tokenizer_vocab"][:-1], "n51"]
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "tokenizer vocabulary"):
                load_checkpoint(path)

    def test_nonportable_or_nonfinite_metadata_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.pt"
            with self.assertRaisesRegex(TypeError, "JSON-like"):
                save_checkpoint(
                    path,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    direction="yt_to_human",
                    epoch=0,
                    metrics={"loss": 1.0},
                    training_config={"output": Path("checkpoints")},
                )
            self.assertFalse(path.exists())

            with self.assertRaisesRegex(ValueError, "finite"):
                save_checkpoint(
                    path,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    direction="yt_to_human",
                    epoch=0,
                    metrics={"loss": float("nan")},
                    training_config={},
                )
            self.assertFalse(path.exists())

    def test_loaded_metadata_does_not_retain_a_second_copy_of_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lean.pt"
            self._save(path)
            _, _, metadata = load_checkpoint(path)
            self.assertNotIn("model_state_dict", metadata)


if __name__ == "__main__":
    unittest.main()
