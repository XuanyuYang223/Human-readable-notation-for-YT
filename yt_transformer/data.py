"""Synthetic paired data for translating Young-tableau notation.

The split is performed on tableaux *before* row/column and direction variants
are expanded.  This prevents the same mathematical object from leaking across
train, validation, and test data under a different surface notation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from random import Random
from typing import Callable, Iterable, Literal, Mapping, Protocol, Sequence, TypeAlias, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .notation import Tableau, format_notation


Direction: TypeAlias = Literal["yt_to_human", "human_to_yt"]
HumanKind: TypeAlias = Literal["row", "col"]
SplitName: TypeAlias = Literal["train", "val", "test"]

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
DEFAULT_DIRECTIONS: tuple[Direction, ...] = ("yt_to_human", "human_to_yt")
DEFAULT_HUMAN_KINDS: tuple[HumanKind, ...] = ("row", "col")


class EncodesNotation(Protocol):
    """The small part of :class:`Tokenizer` required by this module."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        task: HumanKind | None = None,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class TranslationExample:
    """One text-to-text example, plus its leakage-safe group identity.

    ``source_task`` is set only for ``yt_to_human`` examples.  The tokenizer
    turns it into a ``TO_ROW`` or ``TO_COL`` token immediately after ``BOS``.
    Reverse examples need no control token because their row/column surface
    marker already identifies the input notation.
    """

    tableau: Tableau
    direction: Direction
    human_kind: HumanKind
    source: str
    target: str
    source_task: HumanKind | None

    @property
    def tableau_key(self) -> tuple[tuple[int, ...], ...]:
        return self.tableau.rows


def tableau_key(tableau: Tableau) -> tuple[tuple[int, ...], ...]:
    """Return a stable, surface-independent key for a tableau."""

    if not isinstance(tableau, Tableau):
        raise TypeError("tableau must be a Tableau")
    return tableau.rows


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sample_partition(
    rng: Random,
    *,
    cells: int,
    max_rows: int,
    max_columns: int,
) -> tuple[int, ...]:
    """Sample a non-increasing positive partition of exactly ``cells``."""

    minimum_rows = (cells + max_columns - 1) // max_columns
    maximum_rows = min(max_rows, cells)
    if minimum_rows > maximum_rows:
        raise ValueError("cell count cannot fit within max_rows and max_columns")

    row_count = rng.randint(minimum_rows, maximum_rows)
    remaining = cells
    previous = max_columns
    lengths: list[int] = []

    for row_index in range(row_count):
        rows_after = row_count - row_index - 1
        upper = min(previous, max_columns, remaining - rows_after)
        candidates = [
            length
            for length in range(1, upper + 1)
            if rows_after <= remaining - length <= rows_after * length
        ]
        # Feasibility of row_count guarantees at least one candidate.  Keep a
        # defensive error here because it is much easier to diagnose than a
        # malformed synthetic shape later in the pipeline.
        if not candidates:  # pragma: no cover - defensive invariant
            raise RuntimeError("failed to sample a feasible Young shape")
        length = rng.choice(candidates)
        lengths.append(length)
        remaining -= length
        previous = length

    if remaining != 0:  # pragma: no cover - defensive invariant
        raise RuntimeError("partition sampler did not consume all cells")
    return tuple(lengths)


def generate_tableaux(
    count: int,
    *,
    seed: int = 0,
    max_rows: int = 5,
    max_columns: int = 8,
    max_cells: int = 20,
    min_value: int = 1,
    max_value: int = 50,
) -> tuple[Tableau, ...]:
    """Generate ``count`` unique, reproducible, randomly filled tableaux.

    Shapes are left-aligned partitions (row lengths are non-increasing).
    Entries are sampled independently, so no row/column monotonicity is
    imposed.  The default value range follows the fixed tokenizer vocabulary.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    _positive_int("max_rows", max_rows)
    _positive_int("max_columns", max_columns)
    _positive_int("max_cells", max_cells)
    if max_cells > max_rows * max_columns:
        raise ValueError("max_cells cannot exceed max_rows * max_columns")
    if (
        isinstance(min_value, bool)
        or isinstance(max_value, bool)
        or not isinstance(min_value, int)
        or not isinstance(max_value, int)
        or not 1 <= min_value <= max_value <= 50
    ):
        raise ValueError("values must satisfy 1 <= min_value <= max_value <= 50")
    if count == 0:
        return ()

    rng = Random(seed)
    generated: list[Tableau] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    # Rejection is negligible under the defaults.  The cap gives callers a
    # clear failure for deliberately tiny finite spaces (for example, more
    # than 50 one-cell tableaux) rather than an infinite loop.
    attempt_limit = max(1_000, count * 200)

    for _ in range(attempt_limit):
        cells = rng.randint(1, max_cells)
        shape = _sample_partition(
            rng,
            cells=cells,
            max_rows=max_rows,
            max_columns=max_columns,
        )
        rows = tuple(
            tuple(rng.randint(min_value, max_value) for _ in range(length))
            for length in shape
        )
        tableau = Tableau(rows)
        key = tableau.rows
        if key in seen:
            continue
        seen.add(key)
        generated.append(tableau)
        if len(generated) == count:
            return tuple(generated)

    raise ValueError(
        "could not generate the requested number of unique tableaux; "
        "increase the shape/value range or request fewer examples"
    )


def make_translation_examples(
    tableau: Tableau,
    *,
    directions: Sequence[Direction] = DEFAULT_DIRECTIONS,
    human_kinds: Sequence[HumanKind] = DEFAULT_HUMAN_KINDS,
) -> tuple[TranslationExample, ...]:
    """Expand one tableau into the requested direction/notation variants."""

    if not isinstance(tableau, Tableau):
        raise TypeError("tableau must be a Tableau")
    checked_directions: list[Direction] = []
    for direction in directions:
        if direction not in DEFAULT_DIRECTIONS:
            raise ValueError(f"unknown direction: {direction!r}")
        checked_directions.append(cast(Direction, direction))
    checked_kinds: list[HumanKind] = []
    for human_kind in human_kinds:
        if human_kind not in DEFAULT_HUMAN_KINDS:
            raise ValueError(f"unknown human notation kind: {human_kind!r}")
        checked_kinds.append(cast(HumanKind, human_kind))

    raw = format_notation(tableau, "raw")
    examples: list[TranslationExample] = []
    for human_kind in checked_kinds:
        human = format_notation(tableau, human_kind)
        for direction in checked_directions:
            if direction == "yt_to_human":
                examples.append(
                    TranslationExample(
                        tableau=tableau,
                        direction=direction,
                        human_kind=human_kind,
                        source=raw,
                        target=human,
                        source_task=human_kind,
                    )
                )
            else:
                examples.append(
                    TranslationExample(
                        tableau=tableau,
                        direction=direction,
                        human_kind=human_kind,
                        source=human,
                        target=raw,
                        source_task=None,
                    )
                )
    return tuple(examples)


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError("split_ratios must contain train, val, and test weights")
    checked: list[float] = []
    for ratio in ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio < 0:
            raise ValueError("split ratios must be non-negative numbers")
        checked.append(float(ratio))
    total = sum(checked)
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    return cast(tuple[float, float, float], tuple(value / total for value in checked))


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [total * ratio for ratio in ratios]
    counts = [floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return cast(tuple[int, int, int], tuple(counts))


def split_tableaux(
    tableaux: Iterable[Tableau],
    *,
    seed: int = 0,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[SplitName, tuple[Tableau, ...]]:
    """Deduplicate and split tableaux without allowing surface-form leakage."""

    unique: list[Tableau] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for tableau in tableaux:
        key = tableau_key(tableau)
        if key not in seen:
            seen.add(key)
            unique.append(tableau)

    Random(seed).shuffle(unique)
    ratios = _validate_ratios(split_ratios)
    train_count, val_count, test_count = _split_counts(len(unique), ratios)
    train_end = train_count
    val_end = train_end + val_count
    if val_end + test_count != len(unique):  # pragma: no cover - invariant
        raise RuntimeError("split sizes do not cover the input")
    return {
        "train": tuple(unique[:train_end]),
        "val": tuple(unique[train_end:val_end]),
        "test": tuple(unique[val_end:]),
    }


def build_translation_splits(
    tableaux: Iterable[Tableau],
    *,
    seed: int = 0,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    directions: Sequence[Direction] = DEFAULT_DIRECTIONS,
    human_kinds: Sequence[HumanKind] = DEFAULT_HUMAN_KINDS,
) -> dict[SplitName, tuple[TranslationExample, ...]]:
    """Split by tableau, then expand each split into paired text examples."""

    grouped = split_tableaux(tableaux, seed=seed, split_ratios=split_ratios)
    return {
        split_name: tuple(
            example
            for tableau in grouped[split_name]
            for example in make_translation_examples(
                tableau,
                directions=directions,
                human_kinds=human_kinds,
            )
        )
        for split_name in SPLIT_NAMES
    }


def generate_translation_splits(
    count: int,
    *,
    seed: int = 0,
    split_seed: int | None = None,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    directions: Sequence[Direction] = DEFAULT_DIRECTIONS,
    human_kinds: Sequence[HumanKind] = DEFAULT_HUMAN_KINDS,
    max_rows: int = 5,
    max_columns: int = 8,
    max_cells: int = 20,
    min_value: int = 1,
    max_value: int = 50,
) -> dict[SplitName, tuple[TranslationExample, ...]]:
    """Generate unique tableaux and return leakage-safe translation splits."""

    tableaux = generate_tableaux(
        count,
        seed=seed,
        max_rows=max_rows,
        max_columns=max_columns,
        max_cells=max_cells,
        min_value=min_value,
        max_value=max_value,
    )
    return build_translation_splits(
        tableaux,
        seed=seed if split_seed is None else split_seed,
        split_ratios=split_ratios,
        directions=directions,
        human_kinds=human_kinds,
    )


class TranslationDataset(Dataset[Mapping[str, object]]):
    """Torch dataset that encodes paired examples with the hand-made tokenizer."""

    def __init__(
        self,
        examples: Sequence[TranslationExample],
        tokenizer: EncodesNotation,
    ) -> None:
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
        source_ids = self.tokenizer.encode(example.source, task=example.source_task)
        target_ids = self.tokenizer.encode(example.target)
        return {
            "source_ids": self._long_tensor(source_ids, field="source"),
            "target_ids": self._long_tensor(target_ids, field="target"),
            "example": example,
        }


def collate_translation_batch(
    batch: Sequence[Mapping[str, object]],
    *,
    pad_id: int,
) -> dict[str, object]:
    """Pad a batch into ``[batch, sequence]`` tensors and padding masks."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int):
        raise TypeError("pad_id must be an integer")

    source_items: list[Tensor] = []
    target_items: list[Tensor] = []
    examples: list[TranslationExample] = []
    for item in batch:
        source = item.get("source_ids")
        target = item.get("target_ids")
        example = item.get("example")
        if not isinstance(source, Tensor) or source.ndim != 1 or source.dtype != torch.long:
            raise TypeError("each source_ids item must be a one-dimensional LongTensor")
        if not isinstance(target, Tensor) or target.ndim != 1 or target.dtype != torch.long:
            raise TypeError("each target_ids item must be a one-dimensional LongTensor")
        if not isinstance(example, TranslationExample):
            raise TypeError("each item must retain its TranslationExample metadata")
        source_items.append(source)
        target_items.append(target)
        examples.append(example)

    source_ids = pad_sequence(source_items, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(target_items, batch_first=True, padding_value=pad_id)
    return {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "source_padding_mask": source_ids.eq(pad_id),
        "target_padding_mask": target_ids.eq(pad_id),
        "examples": tuple(examples),
    }


def make_collate_fn(pad_id: int) -> Callable[[Sequence[Mapping[str, object]]], dict[str, object]]:
    """Return a DataLoader-compatible collator bound to ``pad_id``."""

    if isinstance(pad_id, bool) or not isinstance(pad_id, int):
        raise TypeError("pad_id must be an integer")

    def collate(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return collate_translation_batch(batch, pad_id=pad_id)

    return collate


__all__ = [
    "DEFAULT_DIRECTIONS",
    "DEFAULT_HUMAN_KINDS",
    "Direction",
    "HumanKind",
    "SPLIT_NAMES",
    "SplitName",
    "TranslationDataset",
    "TranslationExample",
    "build_translation_splits",
    "collate_translation_batch",
    "generate_tableaux",
    "generate_translation_splits",
    "make_collate_fn",
    "make_translation_examples",
    "split_tableaux",
    "tableau_key",
]
