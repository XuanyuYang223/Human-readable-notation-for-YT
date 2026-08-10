"""Deterministic reference conversion for the four notation forms."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .notation import NotationKind, format_notation, parse_notation


def convert_notation(text: str, target: NotationKind) -> str:
    """Convert canonical notation without a neural model.

    This function is the label-generating oracle used to check the learned
    models; it is also useful while debugging a dataset or tokenizer.
    """

    tableau, _ = parse_notation(text)
    return format_notation(tableau, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw/row/column/coordinate YT notation deterministically."
    )
    parser.add_argument("text", help="canonical notation string (quote it in a shell)")
    parser.add_argument("--to", choices=("raw", "row", "col", "coord"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(convert_notation(args.text, args.to))


if __name__ == "__main__":
    main()


__all__ = ["convert_notation", "main"]
