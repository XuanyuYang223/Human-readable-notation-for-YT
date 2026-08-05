import unittest

from yt_transformer.tokenizer import HandmadeTokenizer, VOCAB


class HandmadeTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HandmadeTokenizer()
        self.raw = "[YT start] 2 3 5 | 1 4 [YT end]"
        self.row = "[YT row start] 2 3 5 | 1 4 [YT row end]"
        self.col = "[YT col start] 2 1 | 3 4 | 5 [YT col end]"

    def test_fixed_vocabulary_and_special_ids(self) -> None:
        self.assertEqual(self.tokenizer.vocab, VOCAB)
        self.assertEqual(self.tokenizer.vocab_size, 63)
        self.assertEqual(self.tokenizer.pad_id, 0)
        self.assertEqual(self.tokenizer.bos_id, 1)
        self.assertEqual(self.tokenizer.eos_id, 2)
        self.assertEqual(self.tokenizer.to_row_id, 3)
        self.assertEqual(self.tokenizer.to_col_id, 4)
        self.assertEqual(self.tokenizer.token_id("x1"), 5)
        self.assertEqual(self.tokenizer.token_id("x6"), 10)
        self.assertEqual(self.tokenizer.token_id("s"), 11)
        self.assertEqual(self.tokenizer.token_id("x"), 12)
        self.assertEqual(self.tokenizer.token_id("n1"), 13)
        self.assertEqual(self.tokenizer.token_id("n50"), 62)

    def test_surface_tokenization_uses_markers_spaces_bars_and_atomic_numbers(self) -> None:
        self.assertEqual(
            self.tokenizer.tokenize(self.raw),
            [
                "x5",
                "s",
                "n2",
                "s",
                "n3",
                "s",
                "n5",
                "s",
                "x",
                "s",
                "n1",
                "s",
                "n4",
                "s",
                "x6",
            ],
        )
        self.assertEqual(self.tokenizer.detokenize(self.tokenizer.tokenize(self.col)), self.col)

    def test_encode_decode_round_trip_for_every_notation_kind(self) -> None:
        for text in (self.raw, self.row, self.col):
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                self.assertEqual(encoded[0], self.tokenizer.bos_id)
                self.assertEqual(encoded[-1], self.tokenizer.eos_id)
                self.assertEqual(self.tokenizer.decode(encoded), text)

    def test_encode_inserts_task_after_bos(self) -> None:
        row_ids = self.tokenizer.encode(self.raw, task="row")
        col_ids = self.tokenizer.encode(self.raw, task="col")
        self.assertEqual(row_ids[:2], [self.tokenizer.bos_id, self.tokenizer.to_row_id])
        self.assertEqual(col_ids[:2], [self.tokenizer.bos_id, self.tokenizer.to_col_id])
        self.assertEqual(self.tokenizer.decode(row_ids), self.raw)
        self.assertEqual(self.tokenizer.decode(col_ids), self.raw)

        unwrapped = self.tokenizer.encode(self.raw, add_special_tokens=False, task="row")
        self.assertEqual(unwrapped[0], self.tokenizer.to_row_id)
        self.assertEqual(self.tokenizer.decode(unwrapped), self.raw)

    def test_decode_ignores_padding_and_control_tokens_by_default(self) -> None:
        ids = [self.tokenizer.pad_id, *self.tokenizer.encode(self.row), self.tokenizer.pad_id]
        self.assertEqual(self.tokenizer.decode(ids), self.row)
        with self.assertRaises(ValueError):
            self.tokenizer.decode(ids, skip_special_tokens=False)

    def test_conversion_helpers_round_trip(self) -> None:
        tokens = ["BOS", "TO_COL", "x5", "s", "n50", "s", "x6", "EOS"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(ids), tokens)

    def test_rejects_unknown_and_non_canonical_input(self) -> None:
        invalid_texts = (
            "[YT start] 2  3 [YT end]",
            "[YT start] 0 [YT end]",
            "[YT start] 51 [YT end]",
            "[YT start] two [YT end]",
        )
        for text in invalid_texts:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.tokenizer.tokenize(text)

        with self.assertRaises(ValueError):
            self.tokenizer.detokenize(["x5", "s", "n1"])
        with self.assertRaises(ValueError):
            self.tokenizer.detokenize(["BOS", "x5", "s", "x6"])
        with self.assertRaises(ValueError):
            self.tokenizer.detokenize(["x5", "s", "n2", "n3", "s", "x6"])
        with self.assertRaises(ValueError):
            self.tokenizer.token_id("UNKNOWN")
        with self.assertRaises(ValueError):
            self.tokenizer.id_token(999)
        with self.assertRaises(ValueError):
            self.tokenizer.encode(self.raw, task="raw")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
