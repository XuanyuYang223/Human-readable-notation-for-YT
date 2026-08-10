"""Tests for reproducible, leakage-safe synthetic translation data."""

from __future__ import annotations

import unittest

import torch

from yt_transformer.data import (
    DEFAULT_HUMAN_KINDS,
    SPLIT_NAMES,
    TranslationDataset,
    build_translation_splits,
    collate_translation_batch,
    generate_tableaux,
    generate_translation_splits,
    make_translation_examples,
    split_tableaux,
)
from yt_transformer.notation import Tableau
from yt_transformer.tokenizer import HandmadeTokenizer


class TableauGenerationTests(unittest.TestCase):
    def test_generation_is_reproducible_unique_and_valid(self) -> None:
        first = generate_tableaux(80, seed=1729)
        second = generate_tableaux(80, seed=1729)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 80)
        self.assertEqual(len({tableau.rows for tableau in first}), 80)
        for tableau in first:
            lengths = [len(row) for row in tableau.rows]
            self.assertEqual(lengths, sorted(lengths, reverse=True))
            self.assertLessEqual(len(lengths), 5)
            self.assertLessEqual(sum(lengths), 20)
            self.assertTrue(
                all(1 <= value <= 50 for row in tableau.rows for value in row)
            )

    def test_generation_does_not_sort_entries_to_force_monotonicity(self) -> None:
        tableaux = generate_tableaux(80, seed=7)
        self.assertTrue(
            any(
                any(left > right for left, right in zip(row, row[1:]))
                for tableau in tableaux
                for row in tableau.rows
            )
        )

    def test_generation_validates_bounds(self) -> None:
        with self.assertRaises(ValueError):
            generate_tableaux(-1)
        with self.assertRaises(ValueError):
            generate_tableaux(1, max_rows=2, max_columns=2, max_cells=5)
        with self.assertRaises(ValueError):
            generate_tableaux(1, min_value=0)


class TranslationExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tableau = Tableau(((2, 3, 5), (1, 4)))

    def test_forward_and_reverse_all_human_notation_pairs(self) -> None:
        examples = make_translation_examples(self.tableau)
        self.assertEqual(DEFAULT_HUMAN_KINDS, ("row", "col", "coord"))
        self.assertEqual(len(examples), 6)
        by_key = {(example.direction, example.human_kind): example for example in examples}

        raw = "[YT start] 2 3 5 | 1 4 [YT end]"
        row = "[YT row start] 2 3 5 | 1 4 [YT row end]"
        col = "[YT col start] 2 1 | 3 4 | 5 [YT col end]"
        coord = (
            "[YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | "
            "(2,1) : 1 | (2,2) : 4 [YT coord end]"
        )

        row_forward = by_key[("yt_to_human", "row")]
        self.assertEqual((row_forward.source, row_forward.target), (raw, row))
        self.assertEqual(row_forward.source_task, "row")
        col_forward = by_key[("yt_to_human", "col")]
        self.assertEqual((col_forward.source, col_forward.target), (raw, col))
        self.assertEqual(col_forward.source_task, "col")
        coord_forward = by_key[("yt_to_human", "coord")]
        self.assertEqual((coord_forward.source, coord_forward.target), (raw, coord))
        self.assertEqual(coord_forward.source_task, "coord")

        row_reverse = by_key[("human_to_yt", "row")]
        self.assertEqual((row_reverse.source, row_reverse.target), (row, raw))
        self.assertIsNone(row_reverse.source_task)
        col_reverse = by_key[("human_to_yt", "col")]
        self.assertEqual((col_reverse.source, col_reverse.target), (col, raw))
        self.assertIsNone(col_reverse.source_task)
        coord_reverse = by_key[("human_to_yt", "coord")]
        self.assertEqual((coord_reverse.source, coord_reverse.target), (coord, raw))
        self.assertIsNone(coord_reverse.source_task)

    def test_forward_control_token_and_reverse_marker_disambiguation(self) -> None:
        tokenizer = HandmadeTokenizer()
        examples = make_translation_examples(self.tableau)
        dataset = TranslationDataset(examples, tokenizer)
        items = {
            (example.direction, example.human_kind): dataset[index]
            for index, example in enumerate(examples)
        }

        row_forward_ids = items[("yt_to_human", "row")]["source_ids"]
        col_forward_ids = items[("yt_to_human", "col")]["source_ids"]
        coord_forward_ids = items[("yt_to_human", "coord")]["source_ids"]
        self.assertIsInstance(row_forward_ids, torch.Tensor)
        self.assertEqual(row_forward_ids[1].item(), tokenizer.to_row_id)
        self.assertEqual(col_forward_ids[1].item(), tokenizer.to_col_id)
        self.assertEqual(coord_forward_ids[1].item(), tokenizer.to_coord_id)

        row_reverse_ids = items[("human_to_yt", "row")]["source_ids"]
        col_reverse_ids = items[("human_to_yt", "col")]["source_ids"]
        coord_reverse_ids = items[("human_to_yt", "coord")]["source_ids"]
        self.assertEqual(row_reverse_ids[1].item(), tokenizer.token_id("x1"))
        self.assertEqual(col_reverse_ids[1].item(), tokenizer.token_id("x3"))
        self.assertEqual(coord_reverse_ids[1].item(), tokenizer.token_id("x7"))
        self.assertNotIn(tokenizer.to_row_id, row_reverse_ids.tolist())
        self.assertNotIn(tokenizer.to_col_id, col_reverse_ids.tolist())
        self.assertNotIn(tokenizer.to_coord_id, coord_reverse_ids.tolist())


class SplitTests(unittest.TestCase):
    def test_duplicates_are_removed_before_splitting(self) -> None:
        tableaux = generate_tableaux(9, seed=8)
        split = split_tableaux((*tableaux, tableaux[0], tableaux[1]), seed=1)
        flattened = tuple(tableau for name in SPLIT_NAMES for tableau in split[name])
        self.assertEqual(len(flattened), 9)
        self.assertEqual(len({tableau.rows for tableau in flattened}), 9)

    def test_splits_are_disjoint_and_all_variants_stay_together(self) -> None:
        tableaux = generate_tableaux(60, seed=23)
        splits = build_translation_splits(tableaux, seed=31)
        key_sets = {
            name: {example.tableau_key for example in splits[name]} for name in SPLIT_NAMES
        }
        self.assertTrue(key_sets["train"].isdisjoint(key_sets["val"]))
        self.assertTrue(key_sets["train"].isdisjoint(key_sets["test"]))
        self.assertTrue(key_sets["val"].isdisjoint(key_sets["test"]))
        self.assertEqual(set.union(*key_sets.values()), {tableau.rows for tableau in tableaux})

        for name in SPLIT_NAMES:
            counts: dict[tuple[tuple[int, ...], ...], int] = {}
            for example in splits[name]:
                counts[example.tableau_key] = counts.get(example.tableau_key, 0) + 1
            self.assertTrue(all(count == 6 for count in counts.values()))

    def test_generated_splits_are_reproducible(self) -> None:
        first = generate_translation_splits(30, seed=99, split_seed=100)
        second = generate_translation_splits(30, seed=99, split_seed=100)
        self.assertEqual(first, second)


class DatasetAndCollateTests(unittest.TestCase):
    def test_dataset_returns_long_tensors_and_collate_pads(self) -> None:
        tokenizer = HandmadeTokenizer()
        short = Tableau(((1,),))
        long = Tableau(((2, 3, 5), (1, 4)))
        examples = (
            make_translation_examples(short, human_kinds=("row",))[0],
            make_translation_examples(long, human_kinds=("coord",))[0],
        )
        dataset = TranslationDataset(examples, tokenizer)
        batch = collate_translation_batch(
            [dataset[0], dataset[1]],
            pad_id=tokenizer.pad_id,
        )

        source_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        source_mask = batch["source_padding_mask"]
        target_mask = batch["target_padding_mask"]
        self.assertIsInstance(source_ids, torch.Tensor)
        self.assertIsInstance(target_ids, torch.Tensor)
        self.assertEqual(source_ids.dtype, torch.long)
        self.assertEqual(target_ids.dtype, torch.long)
        self.assertEqual(source_ids.ndim, 2)
        self.assertEqual(target_ids.ndim, 2)
        self.assertEqual(tuple(source_mask.shape), tuple(source_ids.shape))
        self.assertEqual(tuple(target_mask.shape), tuple(target_ids.shape))
        self.assertEqual(source_mask.dtype, torch.bool)
        self.assertEqual(target_mask.dtype, torch.bool)
        self.assertEqual(len(batch["examples"]), 2)
        self.assertTrue(source_mask[0].any().item())
        self.assertTrue(target_mask[0].any().item())

    def test_empty_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_translation_batch([], pad_id=0)


if __name__ == "__main__":
    unittest.main()
