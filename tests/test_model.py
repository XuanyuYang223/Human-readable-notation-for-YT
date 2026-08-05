"""Tests for the small encoder-decoder Transformer (stdlib unittest only)."""

from __future__ import annotations

import json
import unittest

import torch

from yt_transformer.model import ModelConfig, Seq2SeqTransformer


class Seq2SeqTransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig(
            src_vocab_size=23,
            tgt_vocab_size=19,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_seq_len=12,
            pad_id=0,
        )

    def test_config_json_round_trip(self) -> None:
        serialized = json.loads(json.dumps(self.config.to_dict()))
        self.assertEqual(ModelConfig.from_dict(serialized), self.config)

    def test_optional_token_weight_tying_shares_one_parameter(self) -> None:
        tied_config = ModelConfig(
            **{**self.config.to_dict(), "src_vocab_size": 19, "tie_embeddings": True}
        )
        model = Seq2SeqTransformer(tied_config)
        self.assertIs(model.src_embedding.weight, model.tgt_embedding.weight)
        self.assertIs(model.src_embedding.weight, model.output_projection.weight)

        with self.assertRaisesRegex(ValueError, "equal source and target"):
            ModelConfig(**{**self.config.to_dict(), "tie_embeddings": True})

    def test_forward_shape_and_finite_logits_with_padding(self) -> None:
        model = Seq2SeqTransformer(self.config).eval()
        src = torch.tensor(
            [[4, 5, 6, 0, 0], [7, 8, 9, 10, 11]], dtype=torch.long
        )
        tgt_input = torch.tensor(
            [[1, 4, 5, 0], [1, 6, 7, 8]], dtype=torch.long
        )

        logits = model(src, tgt_input)

        self.assertEqual(logits.shape, (2, 4, self.config.tgt_vocab_size))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_greedy_decode_stops_when_all_items_emit_eos(self) -> None:
        model = Seq2SeqTransformer(self.config)
        eos_id = 2
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.fill_(-1.0)
            model.output_projection.bias[eos_id] = 1.0

        src = torch.tensor([[4, 5, 0], [6, 7, 8]], dtype=torch.long)
        decoded = model.greedy_decode(
            src,
            bos_id=1,
            eos_id=eos_id,
            pad_id=0,
            max_new_tokens=6,
        )

        self.assertEqual(decoded.shape, (2, 2))
        self.assertTrue(torch.equal(decoded[:, 0], torch.tensor([1, 1])))
        self.assertTrue(torch.equal(decoded[:, 1], torch.tensor([2, 2])))
        # greedy_decode temporarily enters eval mode but restores its caller's mode.
        self.assertTrue(model.training)

    def test_greedy_decode_respects_max_new_tokens_without_eos(self) -> None:
        model = Seq2SeqTransformer(self.config).eval()
        repeated_token_id = 3
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.fill_(-1.0)
            model.output_projection.bias[repeated_token_id] = 1.0

        src = torch.tensor([[4, 5, 0], [6, 7, 8]], dtype=torch.long)
        decoded = model.greedy_decode(
            src,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            max_new_tokens=3,
        )

        self.assertEqual(decoded.shape, (2, 4))
        self.assertTrue(torch.all(decoded[:, 0].eq(1)).item())
        self.assertTrue(torch.all(decoded[:, 1:].eq(repeated_token_id)).item())
        self.assertFalse(model.training)

    def test_zero_new_tokens_returns_only_bos(self) -> None:
        model = Seq2SeqTransformer(self.config)
        src = torch.tensor([[4, 5]], dtype=torch.long)

        decoded = model.greedy_decode(
            src,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            max_new_tokens=0,
        )

        self.assertTrue(torch.equal(decoded, torch.tensor([[1]])))


if __name__ == "__main__":
    unittest.main()
