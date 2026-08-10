"""Controlled tests for permutation-to-RSK-tableau inference."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn

from yt_transformer.rsk_infer import infer_rsk_tableau
from yt_transformer.tokenizer import HandmadeTokenizer, RSK_VOCAB


class ControlledRSKModel(nn.Module):
    def __init__(self, generated_ids: list[int], *, max_seq_len: int = 128) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.generated_ids = torch.tensor(generated_ids, dtype=torch.long)
        self.last_source: Tensor | None = None

    def greedy_decode(
        self,
        source: Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int,
    ) -> Tensor:
        del bos_id, eos_id, pad_id, max_new_tokens
        self.last_source = source.detach().cpu()
        return self.generated_ids.to(source.device).unsqueeze(0)


class RSKInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)
        self.permutation = "[perm start] 3 1 4 2 [perm end]"
        self.tableau = "[YT start] 1 2 | 3 4 [YT end]"

    def test_valid_prediction_uses_permutation_markers(self) -> None:
        model = ControlledRSKModel(self.tokenizer.encode(self.tableau))

        result = infer_rsk_tableau(
            model,  # type: ignore[arg-type]
            self.tokenizer,
            self.permutation,
        )

        self.assertEqual(result, self.tableau)
        assert model.last_source is not None
        self.assertEqual(model.last_source[0, 1].item(), self.tokenizer.token_id("x9"))

    def test_rejects_non_permutation_input_and_non_raw_output(self) -> None:
        raw_model = ControlledRSKModel(self.tokenizer.encode(self.tableau))
        with self.assertRaises(ValueError):
            infer_rsk_tableau(
                raw_model,  # type: ignore[arg-type]
                self.tokenizer,
                "[YT start] 3 1 4 2 [YT end]",
            )

        row = "[YT row start] 1 2 | 3 4 [YT row end]"
        row_model = ControlledRSKModel(self.tokenizer.encode(row))
        with self.assertRaisesRegex(ValueError, "expected 'raw'"):
            infer_rsk_tableau(
                row_model,  # type: ignore[arg-type]
                self.tokenizer,
                self.permutation,
            )

    def test_missing_eos_is_reported(self) -> None:
        model = ControlledRSKModel(self.tokenizer.encode(self.tableau)[:-1])
        with self.assertRaisesRegex(ValueError, "did not emit EOS"):
            infer_rsk_tableau(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                self.permutation,
            )


if __name__ == "__main__":
    unittest.main()
