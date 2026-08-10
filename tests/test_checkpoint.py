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
from yt_transformer.tokenizer import HandmadeTokenizer, LEGACY_VOCAB, RSK_VOCAB


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

    def test_rsk_checkpoint_uses_and_preserves_the_72_token_vocabulary(self) -> None:
        tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)
        model = Seq2SeqTransformer(
            ModelConfig(
                src_vocab_size=tokenizer.vocab_size,
                tgt_vocab_size=tokenizer.vocab_size,
                d_model=8,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=16,
                dropout=0.0,
                max_seq_len=32,
                pad_id=tokenizer.pad_id,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perm_to_yt.pt"
            save_checkpoint(
                path,
                model=model,
                tokenizer=tokenizer,
                direction="perm_to_yt",
                epoch=2,
                metrics={"loss": 0.25},
                training_config={"seed": 19},
            )

            loaded_model, loaded_tokenizer, payload = load_checkpoint(path)

            self.assertEqual(checkpoint_direction(payload), "perm_to_yt")
            self.assertEqual(loaded_tokenizer.vocab, RSK_VOCAB)
            self.assertEqual(loaded_model.config.src_vocab_size, 72)
            self.assertEqual(loaded_tokenizer.token_id("x9"), 70)
            self.assertEqual(loaded_tokenizer.token_id("x10"), 71)

    def test_rsk_direction_requires_x9_and_x10_when_saving_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "save.pt"
            with self.assertRaisesRegex(ValueError, "RSK-aware tokenizer with x9/x10"):
                save_checkpoint(
                    save_path,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    direction="perm_to_yt",
                    epoch=1,
                    metrics={"loss": 1.0},
                    training_config={},
                )
            self.assertFalse(save_path.exists())

            load_path = Path(directory) / "load.pt"
            self._save(load_path)
            payload = torch.load(load_path, map_location="cpu", weights_only=True)
            payload["direction"] = "perm_to_yt"
            torch.save(payload, load_path)
            with self.assertRaisesRegex(ValueError, "RSK-aware tokenizer with x9/x10"):
                load_checkpoint(load_path)

        self.assertEqual(
            checkpoint_direction({"direction": "perm_to_yt"}), "perm_to_yt"
        )

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

    def test_save_rejects_model_tokenizer_vocabulary_mismatch(self) -> None:
        legacy_tokenizer = HandmadeTokenizer(vocab=LEGACY_VOCAB)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.pt"
            with self.assertRaisesRegex(ValueError, "source vocabulary size"):
                save_checkpoint(
                    path,
                    model=self.model,
                    tokenizer=legacy_tokenizer,
                    direction="yt_to_human",
                    epoch=1,
                    metrics={"loss": 1.0},
                    training_config={},
                )
            self.assertFalse(path.exists())

    def test_save_rejects_invalid_or_unsupported_human_kinds(self) -> None:
        invalid_configs = (
            {"human_kinds": ["grid"]},
            {"human_kinds": ["row", "row"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, training_config in enumerate(invalid_configs):
                path = Path(directory) / f"invalid-{index}.pt"
                with self.subTest(training_config=training_config):
                    with self.assertRaisesRegex(
                        ValueError, "human notation kind"
                    ):
                        save_checkpoint(
                            path,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            direction="yt_to_human",
                            epoch=1,
                            metrics={"loss": 1.0},
                            training_config=training_config,
                        )
                    self.assertFalse(path.exists())

        legacy_tokenizer = HandmadeTokenizer(vocab=LEGACY_VOCAB)
        legacy_model = Seq2SeqTransformer(
            ModelConfig(
                src_vocab_size=legacy_tokenizer.vocab_size,
                tgt_vocab_size=legacy_tokenizer.vocab_size,
                d_model=8,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=16,
                dropout=0.0,
                max_seq_len=32,
                pad_id=legacy_tokenizer.pad_id,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-coord.pt"
            with self.assertRaisesRegex(ValueError, "coordinate-aware tokenizer"):
                save_checkpoint(
                    path,
                    model=legacy_model,
                    tokenizer=legacy_tokenizer,
                    direction="yt_to_human",
                    epoch=1,
                    metrics={"loss": 1.0},
                    training_config={"human_kinds": ["coord"]},
                )
            self.assertFalse(path.exists())

    def test_original_63_token_checkpoints_remain_loadable(self) -> None:
        legacy_tokenizer = HandmadeTokenizer(vocab=LEGACY_VOCAB)
        legacy_model = Seq2SeqTransformer(
            ModelConfig(
                src_vocab_size=legacy_tokenizer.vocab_size,
                tgt_vocab_size=legacy_tokenizer.vocab_size,
                d_model=8,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=16,
                dropout=0.0,
                max_seq_len=32,
                pad_id=legacy_tokenizer.pad_id,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(
                path,
                model=legacy_model,
                tokenizer=legacy_tokenizer,
                direction="yt_to_human",
                epoch=1,
                metrics={"loss": 1.0},
                training_config={"human_kinds": ["row", "col"]},
            )

            loaded_model, loaded_tokenizer, _ = load_checkpoint(path)

            self.assertEqual(loaded_tokenizer.vocab, LEGACY_VOCAB)
            self.assertEqual(loaded_model.config.src_vocab_size, 63)
            self.assertEqual(
                loaded_model.supported_human_kinds,
                ("row", "col"),
            )
            self.assertEqual(
                loaded_tokenizer.encode(
                    "[YT start] 1 [YT end]", task="row"
                )[1],
                loaded_tokenizer.to_row_id,
            )
            with self.assertRaisesRegex(ValueError, "unknown token"):
                loaded_tokenizer.encode(
                    "[YT start] 1 [YT end]", task="coord"
                )

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
