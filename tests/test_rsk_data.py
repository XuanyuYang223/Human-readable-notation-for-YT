"""Tests for reproducible and Knuth-class-safe RSK training data."""

from __future__ import annotations

from collections import Counter
import math
import unittest

import torch

from yt_transformer.rsk_data import (
    RSKDataset,
    SPLIT_NAMES,
    build_rsk_splits,
    collate_rsk_batch,
    generate_permutations,
    generate_rsk_splits,
    make_rsk_example,
    split_rsk_examples,
)
from yt_transformer.tokenizer import HandmadeTokenizer, RSK_VOCAB


class PermutationGenerationTests(unittest.TestCase):
    def test_generation_is_reproducible_unique_and_length_balanced(self) -> None:
        first = generate_permutations(100, seed=1729, min_length=1, max_length=8)
        second = generate_permutations(100, seed=1729, min_length=1, max_length=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        counts = Counter(map(len, first))
        self.assertEqual(set(counts), set(range(1, 9)))
        for permutation in first:
            self.assertEqual(set(permutation), set(range(1, len(permutation) + 1)))

        unsaturated = [
            count
            for length, count in counts.items()
            if count < math.factorial(length)
        ]
        self.assertLessEqual(max(unsaturated) - min(unsaturated), 1)

    def test_each_length_quota_includes_identity_and_reverse(self) -> None:
        permutations = generate_permutations(100, seed=19, min_length=1, max_length=8)
        by_length = {
            length: {value for value in permutations if len(value) == length}
            for length in range(1, 9)
        }
        for length, values in by_length.items():
            identity = tuple(range(1, length + 1))
            reverse = tuple(range(length, 0, -1))
            self.assertIn(identity, values)
            self.assertIn(reverse, values)

    def test_full_small_capacity_generates_every_permutation(self) -> None:
        capacity = sum(math.factorial(length) for length in range(1, 5))
        permutations = generate_permutations(capacity, seed=3, max_length=4)
        self.assertEqual(len(permutations), capacity)
        counts = Counter(map(len, permutations))
        self.assertEqual(
            counts,
            Counter({length: math.factorial(length) for length in range(1, 5)}),
        )

    def test_large_factorial_range_does_not_overflow(self) -> None:
        permutations = generate_permutations(
            20,
            seed=8,
            min_length=50,
            max_length=50,
        )
        self.assertEqual(len(permutations), 20)
        self.assertEqual(len(set(permutations)), 20)
        self.assertTrue(all(len(permutation) == 50 for permutation in permutations))

    def test_generation_checks_capacity_and_bounds(self) -> None:
        capacity = sum(math.factorial(length) for length in range(1, 4))
        with self.assertRaises(ValueError):
            generate_permutations(capacity + 1, max_length=3)
        with self.assertRaises(ValueError):
            generate_permutations(-1)
        with self.assertRaises(ValueError):
            generate_permutations(True)
        with self.assertRaises(ValueError):
            generate_permutations(1, min_length=0)
        with self.assertRaises(ValueError):
            generate_permutations(1, min_length=5, max_length=4)
        with self.assertRaises(ValueError):
            generate_permutations(1, max_length=51)

    def test_zero_count_is_supported(self) -> None:
        self.assertEqual(generate_permutations(0), ())


class RSKExampleTests(unittest.TestCase):
    def test_example_uses_canonical_surfaces_and_exact_oracle(self) -> None:
        example = make_rsk_example((3, 5, 1, 4, 2))
        self.assertEqual(example.permutation, (3, 5, 1, 4, 2))
        self.assertEqual(example.source, "[perm start] 3 5 1 4 2 [perm end]")
        self.assertEqual(example.tableau.rows, ((1, 2), (3, 4), (5,)))
        self.assertEqual(example.target, "[YT start] 1 2 | 3 4 | 5 [YT end]")
        self.assertEqual(example.permutation_key, example.permutation)
        self.assertEqual(example.tableau_key, example.tableau.rows)
        self.assertEqual(example.length, 5)

    def test_invalid_permutation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_rsk_example((1, 1))


class RSKSplitTests(unittest.TestCase):
    def test_same_insertion_tableau_never_crosses_splits(self) -> None:
        permutations = generate_permutations(240, seed=23, min_length=1, max_length=9)
        splits = build_rsk_splits(permutations, seed=31)
        key_sets = {
            name: {example.tableau_key for example in splits[name]}
            for name in SPLIT_NAMES
        }
        self.assertTrue(key_sets["train"].isdisjoint(key_sets["val"]))
        self.assertTrue(key_sets["train"].isdisjoint(key_sets["test"]))
        self.assertTrue(key_sets["val"].isdisjoint(key_sets["test"]))
        self.assertEqual(sum(map(len, splits.values())), len(permutations))

    def test_known_knuth_equivalent_sources_stay_together(self) -> None:
        permutations = (
            (3, 1, 4, 2),
            (3, 4, 1, 2),
            *generate_permutations(30, seed=10, min_length=5, max_length=7),
        )
        splits = build_rsk_splits(permutations, seed=4)
        locations = {
            permutation: split_name
            for split_name, examples in splits.items()
            for example in examples
            for permutation in (example.permutation,)
        }
        self.assertEqual(locations[(3, 1, 4, 2)], locations[(3, 4, 1, 2)])

    def test_split_deduplicates_sources_balances_counts_and_populates_all(self) -> None:
        permutations = generate_permutations(200, seed=17, min_length=1, max_length=9)
        splits = build_rsk_splits((*permutations, permutations[0]), seed=18)
        counts = {name: len(splits[name]) for name in SPLIT_NAMES}
        self.assertEqual(sum(counts.values()), 200)
        self.assertTrue(all(count > 0 for count in counts.values()))

        largest_group = max(
            Counter(
                example.tableau_key
                for examples in splits.values()
                for example in examples
            ).values()
        )
        targets = {"train": 160, "val": 20, "test": 20}
        self.assertTrue(
            all(abs(counts[name] - targets[name]) <= largest_group for name in SPLIT_NAMES)
        )

    def test_each_sufficiently_diverse_length_reaches_every_split(self) -> None:
        permutations = generate_permutations(
            240,
            seed=61,
            min_length=1,
            max_length=9,
        )
        splits = build_rsk_splits(permutations, seed=62)
        lengths_by_split = {
            split_name: {example.length for example in examples}
            for split_name, examples in splits.items()
        }

        # At n >= 3 there are at least three distinct insertion tableaux, and
        # this generated pool contains enough sources to represent them.
        for length in range(3, 10):
            self.assertTrue(
                all(length in lengths_by_split[split_name] for split_name in SPLIT_NAMES)
            )

    def test_lengths_one_and_two_cannot_fill_three_p_disjoint_splits(self) -> None:
        splits = build_rsk_splits(((1,), (1, 2), (2, 1)), seed=70)
        occupied = {
            length: sum(
                any(example.length == length for example in examples)
                for examples in splits.values()
            )
            for length in (1, 2)
        }
        self.assertEqual(occupied[1], 1)
        self.assertLessEqual(occupied[2], 2)

        # The limitation comes from group cardinality, not leakage: n=1 has
        # one P and n=2 has two, and each remains wholly in one split.
        for length, expected_groups in ((1, 1), (2, 2)):
            actual_groups = {
                example.tableau_key
                for examples in splits.values()
                for example in examples
                if example.length == length
            }
            self.assertEqual(len(actual_groups), expected_groups)

    def test_split_is_deterministic_and_input_order_independent(self) -> None:
        examples = tuple(
            make_rsk_example(permutation)
            for permutation in generate_permutations(120, seed=41, max_length=8)
        )
        first = split_rsk_examples(examples, seed=42)
        second = split_rsk_examples(reversed(examples), seed=42)
        self.assertEqual(first, second)

    def test_metadata_arguments_reconstruct_identical_splits(self) -> None:
        metadata = {
            "count": 160,
            "seed": 51,
            "split_seed": 52,
            "min_length": 2,
            "max_length": 9,
            "split_ratios": [0.8, 0.1, 0.1],
        }
        first = generate_rsk_splits(**metadata)
        second = generate_rsk_splits(**metadata)
        self.assertEqual(first, second)

    def test_invalid_split_ratios_are_rejected(self) -> None:
        examples = (make_rsk_example((1,)),)
        with self.assertRaises(ValueError):
            split_rsk_examples(examples, split_ratios=(1.0, 0.0))
        with self.assertRaises(ValueError):
            split_rsk_examples(examples, split_ratios=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            split_rsk_examples(examples, split_ratios=(0.8, float("nan"), 0.2))


class RSKDatasetAndCollateTests(unittest.TestCase):
    def test_dataset_encodes_and_collate_pads(self) -> None:
        tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)
        examples = (
            make_rsk_example((1,)),
            make_rsk_example((3, 5, 1, 4, 2)),
        )
        dataset = RSKDataset(examples, tokenizer)
        batch = collate_rsk_batch([dataset[0], dataset[1]], pad_id=tokenizer.pad_id)

        source_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        self.assertIsInstance(source_ids, torch.Tensor)
        self.assertIsInstance(target_ids, torch.Tensor)
        self.assertEqual(source_ids.dtype, torch.long)
        self.assertEqual(target_ids.dtype, torch.long)
        self.assertEqual(source_ids.ndim, 2)
        self.assertEqual(target_ids.ndim, 2)
        self.assertEqual(
            tuple(batch["source_padding_mask"].shape), tuple(source_ids.shape)
        )
        self.assertEqual(
            tuple(batch["target_padding_mask"].shape), tuple(target_ids.shape)
        )
        self.assertEqual(len(batch["examples"]), 2)
        self.assertTrue(batch["source_padding_mask"][0].any().item())
        self.assertTrue(batch["target_padding_mask"][0].any().item())

    def test_empty_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_rsk_batch([], pad_id=0)


if __name__ == "__main__":
    unittest.main()
