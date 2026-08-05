"""Fast, controlled tests for single-example inference."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn

from yt_transformer.infer import translate
from yt_transformer.tokenizer import HandmadeTokenizer


class ControlledDecodeModel(nn.Module):
    """Minimal model surface used by ``translate``, with fixed generation."""

    def __init__(self, generated_ids: list[int], *, max_seq_len: int = 64) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.generated_ids = torch.tensor(generated_ids, dtype=torch.long)
        self.last_source: Tensor | None = None
        self.last_decode_arguments: tuple[int, int, int, int] | None = None

    def greedy_decode(
        self,
        source: Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int,
    ) -> Tensor:
        self.last_source = source.detach().cpu()
        self.last_decode_arguments = (bos_id, eos_id, pad_id, max_new_tokens)
        return self.generated_ids.to(source.device).unsqueeze(0).expand(source.size(0), -1)


class TranslateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HandmadeTokenizer()
        self.raw = "[YT start] 2 3 5 | 1 4 [YT end]"
        self.row = "[YT row start] 2 3 5 | 1 4 [YT row end]"
        self.col = "[YT col start] 2 1 | 3 4 | 5 [YT col end]"

    def _model_for(self, output: str, *, max_seq_len: int = 64) -> ControlledDecodeModel:
        return ControlledDecodeModel(
            self.tokenizer.encode(output), max_seq_len=max_seq_len
        )

    def test_valid_forward_translation_uses_requested_task_token(self) -> None:
        model = self._model_for(self.col)

        result = translate(
            model,  # type: ignore[arg-type]
            self.tokenizer,
            "yt_to_human",
            self.raw,
            style="col",
            max_new_tokens=31,
        )

        self.assertEqual(result, self.col)
        self.assertIsNotNone(model.last_source)
        assert model.last_source is not None
        self.assertEqual(model.last_source[0, 0].item(), self.tokenizer.bos_id)
        self.assertEqual(model.last_source[0, 1].item(), self.tokenizer.to_col_id)
        self.assertEqual(
            model.last_decode_arguments,
            (
                self.tokenizer.bos_id,
                self.tokenizer.eos_id,
                self.tokenizer.pad_id,
                31,
            ),
        )

    def test_valid_reverse_translation_accepts_both_human_markers(self) -> None:
        for human in (self.row, self.col):
            with self.subTest(human=human):
                model = self._model_for(self.raw)
                self.assertEqual(
                    translate(
                        model,  # type: ignore[arg-type]
                        self.tokenizer,
                        "human_to_yt",
                        human,
                    ),
                    self.raw,
                )
                assert model.last_source is not None
                self.assertNotIn(
                    model.last_source[0, 1].item(),
                    (self.tokenizer.to_row_id, self.tokenizer.to_col_id),
                )

    def test_input_wrapper_must_match_checkpoint_direction(self) -> None:
        forward = self._model_for(self.row)
        with self.assertRaisesRegex(ValueError, "yt_to_human expects"):
            translate(
                forward,  # type: ignore[arg-type]
                self.tokenizer,
                "yt_to_human",
                self.row,
            )

        reverse = self._model_for(self.raw)
        with self.assertRaisesRegex(ValueError, "human_to_yt expects"):
            translate(
                reverse,  # type: ignore[arg-type]
                self.tokenizer,
                "human_to_yt",
                self.raw,
            )

        with self.assertRaisesRegex(ValueError, "style must be row or col"):
            translate(
                forward,  # type: ignore[arg-type]
                self.tokenizer,
                "yt_to_human",
                self.raw,
                style="diagonal",  # type: ignore[arg-type]
            )

    def test_missing_eos_reports_generated_symbolic_tokens(self) -> None:
        generated_without_eos = self.tokenizer.encode(self.row)[:-1]
        model = ControlledDecodeModel(generated_without_eos)

        with self.assertRaisesRegex(ValueError, "did not emit EOS") as caught:
            translate(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                "yt_to_human",
                self.raw,
            )
        self.assertIn("x1", str(caught.exception))

    def test_eos_terminated_but_invalid_notation_is_rejected(self) -> None:
        model = ControlledDecodeModel(
            [
                self.tokenizer.bos_id,
                self.tokenizer.token_id("n1"),
                self.tokenizer.eos_id,
            ]
        )

        with self.assertRaisesRegex(ValueError, "emitted invalid notation"):
            translate(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                "yt_to_human",
                self.raw,
            )

    def test_valid_notation_of_wrong_output_kind_is_rejected(self) -> None:
        model = self._model_for(self.col)
        with self.assertRaisesRegex(ValueError, "expected 'row'"):
            translate(
                model,  # type: ignore[arg-type]
                self.tokenizer,
                "yt_to_human",
                self.raw,
                style="row",
            )


if __name__ == "__main__":
    unittest.main()
