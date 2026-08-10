"""Out-of-distribution length tests using random entry permutations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from random import Random
from typing import Any, cast

from .checkpoint import checkpoint_direction, checkpoint_human_kinds, load_checkpoint
from .data import Direction, make_translation_examples
from .evaluate import evaluate_examples
from .notation import Tableau
from .runtime import resolve_device


def compact_shape(entries: int, *, max_rows: int = 5) -> tuple[int, ...]:
    """Return a compact non-increasing shape containing exactly ``entries`` cells."""

    if isinstance(entries, bool) or not isinstance(entries, int) or entries <= 0:
        raise ValueError("entries must be a positive integer")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    width = math.ceil(entries / max_rows)
    full_rows, remainder = divmod(entries, width)
    lengths = (width,) * full_rows
    if remainder:
        lengths += (remainder,)
    return lengths


def stress_shape(entries: int) -> tuple[int, ...]:
    """Choose a comparable 5-row shape, balancing only above the 50-token limit."""

    if entries <= 50:
        return compact_shape(entries, max_rows=5)
    return compact_shape(entries, max_rows=max(5, round(math.sqrt(entries))))


def generate_length_stress_tableaux(
    samples: int,
    *,
    entries: int,
    seed: int,
) -> tuple[Tableau, ...]:
    """Generate unique-value permutations, or repeated fills above 50 cells.

    For ``entries <= 50``, every tableau contains distinct values sampled from
    ``1..50``. Above 50 cells, uniqueness is impossible under the fixed
    tokenizer vocabulary, so ``1..50`` is repeated as needed and shuffled.
    """

    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    shape = stress_shape(entries)
    rng = Random(seed)
    result: list[Tableau] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    attempts = 0
    while len(result) < samples:
        attempts += 1
        if attempts > samples * 100:
            raise RuntimeError("could not generate enough distinct stress tableaux")
        if entries <= 50:
            values = rng.sample(range(1, 51), entries)
        else:
            values = [index % 50 + 1 for index in range(entries)]
            rng.shuffle(values)
        offset = 0
        rows = []
        for length in shape:
            rows.append(tuple(values[offset : offset + length]))
            offset += length
        tableau = Tableau(rows)
        if tableau.rows in seen:
            continue
        seen.add(tableau.rows)
        result.append(tableau)
    return tuple(result)


def evaluate_ood_lengths(
    *,
    forward_checkpoint: Path,
    reverse_checkpoint: Path,
    entries_values: tuple[int, ...],
    samples: int,
    batch_size: int,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate both directions and their common human styles at several cell counts."""

    device = resolve_device(device_name)
    forward, forward_tokenizer, forward_metadata = load_checkpoint(
        forward_checkpoint, device=device
    )
    reverse, reverse_tokenizer, reverse_metadata = load_checkpoint(
        reverse_checkpoint, device=device
    )
    if checkpoint_direction(forward_metadata) != "yt_to_human":
        raise ValueError("--yt-to-human checkpoint has the wrong direction")
    if checkpoint_direction(reverse_metadata) != "human_to_yt":
        raise ValueError("--human-to-yt checkpoint has the wrong direction")
    if forward_tokenizer.vocab != reverse_tokenizer.vocab:
        raise ValueError("checkpoint tokenizers do not match")

    reverse_styles = set(checkpoint_human_kinds(reverse_metadata))
    common_styles = tuple(
        style
        for style in checkpoint_human_kinds(forward_metadata)
        if style in reverse_styles
    )
    if not common_styles:
        raise ValueError("forward and reverse checkpoints have no human styles in common")

    training_config = forward_metadata.get("training_config")
    trained_max_cells = (
        training_config.get("max_cells") if isinstance(training_config, dict) else None
    )
    report: dict[str, Any] = {
        "device": str(device),
        "samples_per_case": samples,
        "training_max_cells": trained_max_cells,
        "cases": {},
    }
    for case_index, entries in enumerate(entries_values):
        tableaux = generate_length_stress_tableaux(
            samples,
            entries=entries,
            seed=seed + case_index,
        )
        case: dict[str, Any] = {
            "entries": entries,
            "shape": list(stress_shape(entries)),
            "all_entries_unique": entries <= 50,
            "metrics": {},
        }
        for direction, model in (
            (cast(Direction, "yt_to_human"), forward),
            (cast(Direction, "human_to_yt"), reverse),
        ):
            direction_metrics: dict[str, Any] = {}
            for style in common_styles:
                examples = tuple(
                    example
                    for tableau in tableaux
                    for example in make_translation_examples(
                        tableau,
                        directions=(direction,),
                        human_kinds=(style,),
                    )
                )
                try:
                    metrics = evaluate_examples(
                        model,
                        forward_tokenizer,
                        examples,
                        batch_size=batch_size,
                    )
                except ValueError as exc:
                    direction_metrics[style] = {"error": str(exc)}
                else:
                    direction_metrics[style] = asdict(metrics)
            case["metrics"][direction] = direction_metrics
        report["cases"][str(entries)] = case
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test length extrapolation beyond the training maximum cells."
    )
    parser.add_argument("--yt-to-human", type=Path, required=True)
    parser.add_argument("--human-to-yt", type=Path, required=True)
    parser.add_argument(
        "--entries",
        type=int,
        nargs="+",
        default=(20, 21, 30, 40, 50, 54),
        help="cell counts to test; counts above 50 must repeat vocabulary values",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(set(args.entries)) != len(args.entries):
        raise SystemExit("--entries values must be unique")
    report = evaluate_ood_lengths(
        forward_checkpoint=args.yt_to_human,
        reverse_checkpoint=args.human_to_yt,
        entries_values=tuple(args.entries),
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "compact_shape",
    "evaluate_ood_lengths",
    "generate_length_stress_tableaux",
    "main",
    "stress_shape",
]
