"""Canonical permutations and the standard RSK insertion tableau.

This module implements the permutation-to-Young-tableau task used by the
training pipeline.  It deliberately computes only the insertion tableau
``P``: recovering a permutation would require the recording tableau ``Q`` as
well, so there is no inverse operation from a lone :class:`Tableau`.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, TypeAlias

from .notation import Tableau


Permutation: TypeAlias = tuple[int, ...]

_PERM_START = "[perm start]"
_PERM_END = "[perm end]"
_MAX_PERMUTATION_SIZE = 50


def validate_permutation(values: Iterable[int]) -> Permutation:
    """Normalize and validate a non-empty permutation of ``1..n``.

    The fixed tokenizer has atomic number tokens only for ``1..50``, so this
    public representation intentionally limits ``n`` to 50 as well.
    """

    if isinstance(values, (str, bytes)):
        raise TypeError("permutation must be an iterable of integers")
    try:
        permutation = tuple(values)
    except TypeError as exc:
        raise TypeError("permutation must be an iterable of integers") from exc

    if not permutation:
        raise ValueError("permutation must not be empty")
    if len(permutation) > _MAX_PERMUTATION_SIZE:
        raise ValueError("permutation length must be in the range 1..50")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in permutation):
        raise ValueError("permutation entries must be integers")
    if set(permutation) != set(range(1, len(permutation) + 1)):
        raise ValueError("permutation entries must be exactly 1..n")
    return permutation


def format_permutation(values: Iterable[int]) -> str:
    """Serialize a permutation using its canonical plain-text surface form."""

    permutation = validate_permutation(values)
    body = " ".join(str(value) for value in permutation)
    return f"{_PERM_START} {body} {_PERM_END}"


def parse_permutation(text: str) -> Permutation:
    """Parse strict ``[perm start] ... [perm end]`` permutation text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    prefix = f"{_PERM_START} "
    suffix = f" {_PERM_END}"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("text does not use the canonical permutation marker pair")
    body = text[len(prefix) : -len(suffix)]
    if not body:
        raise ValueError("permutation must not be empty")

    fields = body.split(" ")
    if any(not field or not field.isascii() or not field.isdecimal() for field in fields):
        raise ValueError("permutation must contain canonical decimal integers")
    if any(len(field) > 1 and field.startswith("0") for field in fields):
        raise ValueError("permutation integers must not contain leading zeroes")

    permutation = validate_permutation(int(field) for field in fields)
    if format_permutation(permutation) != text:
        raise ValueError("text is not in canonical permutation notation")
    return permutation


def rsk_insertion_tableau(values: Iterable[int]) -> Tableau:
    """Return the standard row-insertion tableau ``P`` of a permutation.

    Values are read from left to right.  At each row, the first entry strictly
    greater than the value being inserted is replaced and bumped to the next
    row; if there is no such entry, the value is appended and insertion ends.
    """

    permutation = validate_permutation(values)
    rows: list[list[int]] = []
    for value in permutation:
        bumped = value
        for row in rows:
            insertion_index = bisect_right(row, bumped)
            if insertion_index == len(row):
                row.append(bumped)
                break
            row[insertion_index], bumped = bumped, row[insertion_index]
        else:
            rows.append([bumped])
    return Tableau(rows)


__all__ = [
    "Permutation",
    "format_permutation",
    "parse_permutation",
    "rsk_insertion_tableau",
    "validate_permutation",
]
