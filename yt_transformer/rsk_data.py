"""Synthetic, leakage-safe data for permutation-to-RSK-tableau training.

Permutations are generated without rejection by sampling their lexicographic
ranks and unranking them.  Length quotas are water-filled across the requested
range, subject to the finite ``n!`` capacity at each length.  Splitting is
length-stratified and done by insertion tableau rather than by source
permutation: all members of one Knuth class therefore remain in exactly one
split, while every sufficiently diverse length reaches each requested split.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from random import Random
from typing import Callable, Iterable, Literal, Mapping, Protocol, Sequence, TypeAlias, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .notation import Tableau, format_notation
from .rsk import (
    Permutation,
    format_permutation,
    rsk_insertion_tableau,
    validate_permutation,
)


SplitName: TypeAlias = Literal["train", "val", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")


class EncodesRSK(Protocol):
    """The tokenizer operation needed by :class:`RSKDataset`."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class RSKExample:
    """One canonical permutation and its Robinson--Schensted tableau ``P``."""

    permutation: Permutation
    tableau: Tableau
    source: str
    target: str

    @property
    def permutation_key(self) -> Permutation:
        return self.permutation

    @property
    def tableau_key(self) -> tuple[tuple[int, ...], ...]:
        return self.tableau.rows

    @property
    def length(self) -> int:
        return len(self.permutation)


def _checked_int(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be an integer that is {qualifier}")
    return value


def _validate_generation_bounds(
    count: int,
    min_length: int,
    max_length: int,
) -> tuple[int, int, int]:
    checked_count = _checked_int("count", count)
    checked_min = _checked_int("min_length", min_length, minimum=1)
    checked_max = _checked_int("max_length", max_length, minimum=1)
    if checked_min > checked_max:
        raise ValueError("min_length must not exceed max_length")
    if checked_max > 50:
        raise ValueError("max_length must not exceed the 50-token number range")
    capacity = sum(math.factorial(length) for length in range(checked_min, checked_max + 1))
    if checked_count > capacity:
        raise ValueError(
            f"requested {checked_count} unique permutations, but lengths "
            f"{checked_min}..{checked_max} contain only {capacity}"
        )
    return checked_count, checked_min, checked_max


def _length_quotas(
    count: int,
    *,
    min_length: int,
    max_length: int,
) -> dict[int, int]:
    """Water-fill ``count`` across lengths while respecting ``n!`` capacities."""

    capacities = {
        length: math.factorial(length)
        for length in range(min_length, max_length + 1)
    }
    quotas = {length: 0 for length in capacities}
    active = list(capacities)
    level = 0
    remaining = count

    while remaining and active:
        next_level = min(capacities[length] for length in active)
        layer_size = (next_level - level) * len(active)
        if layer_size <= remaining:
            for length in active:
                quotas[length] = next_level
            remaining -= layer_size
            level = next_level
            active = [
                length for length in active if capacities[length] > next_level
            ]
            continue

        full_layers, extra = divmod(remaining, len(active))
        for length in active:
            quotas[length] = level + full_layers
        # Deterministically give the at-most-one remainder to shorter lengths.
        # Counts among all unsaturated lengths still differ by at most one.
        for length in active[:extra]:
            quotas[length] += 1
        remaining = 0

    if remaining:  # pragma: no cover - guarded by the total capacity check
        raise RuntimeError("permutation quota allocation exhausted its capacity")
    return quotas


def _sample_unique_offsets(rng: Random, population: int, count: int) -> list[int]:
    """Sample offsets from ``range(population)`` even when it exceeds sys.maxsize.

    Floyd's algorithm uses memory and time proportional to ``count`` and avoids
    calling ``len`` on enormous ranges such as ``range(50!)``.
    """

    if not 0 <= count <= population:
        raise ValueError("sample count is outside the finite population")
    chosen: set[int] = set()
    for upper in range(population - count, population):
        candidate = rng.randrange(upper + 1)
        chosen.add(upper if candidate in chosen else candidate)
    result = sorted(chosen)
    rng.shuffle(result)
    return result


def _unrank_permutation(length: int, rank: int) -> Permutation:
    """Return the zero-based lexicographic permutation with the given rank."""

    available = list(range(1, length + 1))
    result: list[int] = []
    remainder = rank
    for width in range(length, 0, -1):
        block = math.factorial(width - 1)
        index, remainder = divmod(remainder, block)
        result.append(available.pop(index))
    return tuple(result)


def generate_permutations(
    count: int,
    *,
    seed: int = 0,
    min_length: int = 1,
    max_length: int = 20,
) -> tuple[Permutation, ...]:
    """Generate unique reproducible permutations with balanced length quotas.

    For every represented length, the identity is included.  If its quota is
    at least two, the reverse permutation is included as well.  Remaining
    permutations are sampled uniformly by rank without rejection.
    """

    checked_count, checked_min, checked_max = _validate_generation_bounds(
        count, min_length, max_length
    )
    _checked_int("seed", seed)
    if checked_count == 0:
        return ()

    quotas = _length_quotas(
        checked_count,
        min_length=checked_min,
        max_length=checked_max,
    )
    rng = Random(seed)
    generated: list[Permutation] = []
    for length in range(checked_min, checked_max + 1):
        quota = quotas[length]
        if quota == 0:
            continue
        capacity = math.factorial(length)
        ranks = [0]
        if quota >= 2:
            ranks.append(capacity - 1)
        interior_count = quota - len(ranks)
        if interior_count:
            offsets = _sample_unique_offsets(rng, capacity - 2, interior_count)
            ranks.extend(offset + 1 for offset in offsets)
        rng.shuffle(ranks)
        generated.extend(_unrank_permutation(length, rank) for rank in ranks)

    if len(generated) != checked_count:  # pragma: no cover - internal invariant
        raise RuntimeError("permutation generator returned the wrong count")
    rng.shuffle(generated)
    return tuple(generated)


def make_rsk_example(values: Iterable[int]) -> RSKExample:
    """Build one canonical source/target example using the exact RSK oracle."""

    permutation = validate_permutation(values)
    tableau = rsk_insertion_tableau(permutation)
    return RSKExample(
        permutation=permutation,
        tableau=tableau,
        source=format_permutation(permutation),
        target=format_notation(tableau, "raw"),
    )


def make_rsk_examples(permutations: Iterable[Iterable[int]]) -> tuple[RSKExample, ...]:
    """Build RSK examples, retaining input order and any exact duplicates."""

    return tuple(make_rsk_example(permutation) for permutation in permutations)


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError("split_ratios must contain train, val, and test weights")
    checked: list[float] = []
    for ratio in ratios:
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or ratio < 0
        ):
            raise ValueError("split ratios must be non-negative finite numbers")
        checked.append(float(ratio))
    total = sum(checked)
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    return cast(
        tuple[float, float, float],
        tuple(ratio / total for ratio in checked),
    )


def _assignment_error(
    counts: Sequence[int],
    targets: Sequence[float],
) -> float:
    return sum((count - target) ** 2 for count, target in zip(counts, targets, strict=True))


_RSKGroup: TypeAlias = tuple[
    tuple[tuple[int, ...], ...],
    list[RSKExample],
]


def _assign_group_stratum(
    groups: list[_RSKGroup],
    *,
    ratios: tuple[float, float, float],
    rng: Random,
) -> list[list[_RSKGroup]]:
    """Balance whole groups within one permutation-length stratum."""

    # Canonical sorting makes the seeded assignment independent of caller
    # iteration order.  Shuffling before the stable size sort randomizes only
    # ties, while largest-first placement gives the greedy balancer less chance
    # to be trapped by a late large Knuth class.
    ordered = sorted(groups, key=lambda item: item[0])
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)

    total = sum(len(group_examples) for _, group_examples in ordered)
    targets = tuple(total * ratio for ratio in ratios)
    assignments: list[list[_RSKGroup]] = [[], [], []]
    counts = [0, 0, 0]
    positive_indices = [index for index, ratio in enumerate(ratios) if ratio > 0]

    for group in ordered:
        size = len(group[1])
        candidates: list[tuple[float, int]] = []
        for split_index in positive_indices:
            proposed = counts.copy()
            proposed[split_index] += size
            candidates.append((_assignment_error(proposed, targets), split_index))
        _, selected = min(candidates)
        assignments[selected].append(group)
        counts[selected] += size

    # A pure squared-error optimum can leave a small-ratio split empty for a
    # short stratum.  If the stratum has at least one whole group per requested
    # split, move the group that causes the smallest loss of count balance.
    if len(ordered) >= len(positive_indices):
        for empty_index in positive_indices:
            if assignments[empty_index]:
                continue
            moves: list[tuple[float, int, int]] = []
            for source_index in positive_indices:
                if len(assignments[source_index]) <= 1:
                    continue
                for group_index, group in enumerate(assignments[source_index]):
                    size = len(group[1])
                    proposed = counts.copy()
                    proposed[source_index] -= size
                    proposed[empty_index] += size
                    moves.append(
                        (
                            _assignment_error(proposed, targets),
                            source_index,
                            group_index,
                        )
                    )
            if not moves:  # pragma: no cover - group-count condition implies a move
                raise RuntimeError("could not populate every requested split")
            _, source_index, group_index = min(moves)
            group = assignments[source_index].pop(group_index)
            size = len(group[1])
            assignments[empty_index].append(group)
            counts[source_index] -= size
            counts[empty_index] += size
    return assignments


def split_rsk_examples(
    examples: Iterable[RSKExample],
    *,
    seed: int = 0,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[SplitName, tuple[RSKExample, ...]]:
    """Deduplicate and assign whole insertion-tableau groups to splits.

    Each permutation length is balanced independently.  Within a length,
    groups are processed largest first and placed to minimize squared error
    from the desired *example* counts.  When a length has enough groups, a
    deterministic repair guarantees that every positive-ratio split contains
    that length.  Lengths one and two cannot occupy three group-disjoint splits.
    """

    _checked_int("seed", seed)
    ratios = _validate_ratios(split_ratios)

    by_permutation: dict[Permutation, RSKExample] = {}
    for example in examples:
        if not isinstance(example, RSKExample):
            raise TypeError("examples must contain RSKExample instances")
        previous = by_permutation.get(example.permutation_key)
        if previous is not None and previous != example:
            raise ValueError("one permutation has conflicting RSK examples")
        by_permutation.setdefault(example.permutation_key, example)

    groups_by_tableau: dict[
        tuple[tuple[int, ...], ...], list[RSKExample]
    ] = {}
    for example in by_permutation.values():
        groups_by_tableau.setdefault(example.tableau_key, []).append(example)
    for group in groups_by_tableau.values():
        group.sort(key=lambda example: example.permutation_key)

    groups_by_length: dict[int, list[_RSKGroup]] = {}
    for tableau_key, group_examples in groups_by_tableau.items():
        lengths = {example.length for example in group_examples}
        if len(lengths) != 1:  # pragma: no cover - equal P has equal cell count
            raise RuntimeError("one insertion-tableau group spans multiple lengths")
        length = next(iter(lengths))
        groups_by_length.setdefault(length, []).append((tableau_key, group_examples))

    assignments: list[list[_RSKGroup]] = [[], [], []]
    for length in sorted(groups_by_length):
        # Mixing the integer seed with the length keeps every stratum stable if
        # callers later add or remove examples at a different length.
        stratum_rng = Random(seed ^ (length * 0x9E3779B97F4A7C15))
        stratum_assignments = _assign_group_stratum(
            groups_by_length[length],
            ratios=ratios,
            rng=stratum_rng,
        )
        for split_index in range(len(SPLIT_NAMES)):
            assignments[split_index].extend(stratum_assignments[split_index])

    result: dict[SplitName, tuple[RSKExample, ...]] = {}
    for split_index, split_name in enumerate(SPLIT_NAMES):
        flattened = [
            example
            for _, group_examples in assignments[split_index]
            for example in group_examples
        ]
        flattened.sort(key=lambda example: example.permutation_key)
        result[split_name] = tuple(flattened)
    return result


def build_rsk_splits(
    permutations: Iterable[Iterable[int]],
    *,
    seed: int = 0,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[SplitName, tuple[RSKExample, ...]]:
    """Build exact labels and return insertion-tableau-grouped splits."""

    return split_rsk_examples(
        make_rsk_examples(permutations),
        seed=seed,
        split_ratios=split_ratios,
    )


def generate_rsk_splits(
    count: int,
    *,
    seed: int = 0,
    split_seed: int | None = None,
    min_length: int = 1,
    max_length: int = 20,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[SplitName, tuple[RSKExample, ...]]:
    """Generate and split RSK data from metadata-reconstructible arguments."""

    permutations = generate_permutations(
        count,
        seed=seed,
        min_length=min_length,
        max_length=max_length,
    )
    return build_rsk_splits(
        permutations,
        seed=seed if split_seed is None else split_seed,
        split_ratios=split_ratios,
    )


class RSKDataset(Dataset[Mapping[str, object]]):
    """Torch dataset encoding permutation sources and raw-tableau targets."""

    def __init__(self, examples: Sequence[RSKExample], tokenizer: EncodesRSK) -> None:
        self.examples = tuple(examples)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _long_tensor(token_ids: Sequence[int], *, field: str) -> Tensor:
        ids = list(token_ids)
        if not ids:
            raise ValueError(f"tokenizer returned an empty {field} sequence")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise TypeError(f"tokenizer {field} IDs must be integers")
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        example = self.examples[index]
        source_ids = self.tokenizer.encode(example.source)
        target_ids = self.tokenizer.encode(example.target)
        return {
            "source_ids": self._long_tensor(source_ids, field="source"),
            "target_ids": self._long_tensor(target_ids, field="target"),
            "example": example,
        }


def collate_rsk_batch(
    batch: Sequence[Mapping[str, object]],
    *,
    pad_id: int,
) -> dict[str, object]:
    """Pad an RSK batch and return tensors, masks, and example metadata."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int):
        raise TypeError("pad_id must be an integer")

    sources: list[Tensor] = []
    targets: list[Tensor] = []
    examples: list[RSKExample] = []
    for item in batch:
        source = item.get("source_ids")
        target = item.get("target_ids")
        example = item.get("example")
        if not isinstance(source, Tensor) or source.ndim != 1 or source.dtype != torch.long:
            raise TypeError("each source_ids item must be a one-dimensional LongTensor")
        if not isinstance(target, Tensor) or target.ndim != 1 or target.dtype != torch.long:
            raise TypeError("each target_ids item must be a one-dimensional LongTensor")
        if not isinstance(example, RSKExample):
            raise TypeError("each item must retain its RSKExample metadata")
        sources.append(source)
        targets.append(target)
        examples.append(example)

    source_ids = pad_sequence(sources, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    return {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "source_padding_mask": source_ids.eq(pad_id),
        "target_padding_mask": target_ids.eq(pad_id),
        "examples": tuple(examples),
    }


def make_rsk_collate_fn(
    pad_id: int,
) -> Callable[[Sequence[Mapping[str, object]]], dict[str, object]]:
    """Return a DataLoader-compatible RSK collator bound to ``pad_id``."""

    if isinstance(pad_id, bool) or not isinstance(pad_id, int):
        raise TypeError("pad_id must be an integer")

    def collate(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return collate_rsk_batch(batch, pad_id=pad_id)

    return collate


__all__ = [
    "RSKDataset",
    "RSKExample",
    "SPLIT_NAMES",
    "SplitName",
    "build_rsk_splits",
    "collate_rsk_batch",
    "generate_permutations",
    "generate_rsk_splits",
    "make_rsk_collate_fn",
    "make_rsk_example",
    "make_rsk_examples",
    "split_rsk_examples",
]
