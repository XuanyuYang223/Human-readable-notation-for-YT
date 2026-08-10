"""Deterministic Robinson--Schensted conversion from a permutation to ``P``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .notation import format_notation
from .rsk import parse_permutation, rsk_insertion_tableau


def convert_permutation(text: str) -> str:
    """Return the raw YT notation for the insertion tableau of ``text``."""

    permutation = parse_permutation(text)
    return format_notation(rsk_insertion_tableau(permutation), "raw")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the standard RSK insertion tableau P exactly."
    )
    parser.add_argument(
        "text",
        help="canonical input such as '[perm start] 3 1 4 2 [perm end]'",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(convert_permutation(args.text))


if __name__ == "__main__":
    main()


__all__ = ["convert_permutation", "main"]
