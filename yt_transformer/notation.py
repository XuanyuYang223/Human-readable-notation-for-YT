"""Canonical, human-readable notation for small Young tableaux.

The surface forms implemented here follow the conventions in the project notes::

    [YT start] 2 3 5 | 1 4 [YT end]
    [YT row start] 2 3 5 | 1 4 [YT row end]
    [YT col start] 2 1 | 3 4 | 5 [YT col end]
    [YT coord start] (1,1) : 2 | (1,2) : 3 | (1,3) : 5 | (2,1) : 1 | (2,2) : 4 [YT coord end]

``raw`` and ``row`` serialize rows.  ``col`` serializes the transpose, so it is
losslessly convertible back to the same :class:`Tableau` even for ragged rows.
``coord`` emits every cell in one-based matrix order as ``(row,column) : value``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal, TypeAlias, cast


NotationKind: TypeAlias = Literal["raw", "row", "col", "coord"]

_MARKERS: dict[NotationKind, tuple[str, str]] = {
    "raw": ("[YT start]", "[YT end]"),
    "row": ("[YT row start]", "[YT row end]"),
    "col": ("[YT col start]", "[YT col end]"),
    "coord": ("[YT coord start]", "[YT coord end]"),
}

_COORD_ENTRY = re.compile(
    r"\(([1-9][0-9]*),([1-9][0-9]*)\) : ([1-9][0-9]*)",
    re.ASCII,
)


@dataclass(frozen=True, slots=True, init=False)
class Tableau:
    """An immutable tableau whose row lengths form a partition.

    The constructor accepts any finite iterable of finite row iterables and
    stores a fully immutable tuple-of-tuples.  Rows must be non-empty, row
    lengths must be non-increasing, and every entry must be an integer in
    ``1..50``.  The empty tableau is supported as ``Tableau(())``.
    """

    rows: tuple[tuple[int, ...], ...]

    def __init__(self, rows: Iterable[Iterable[int]]) -> None:
        try:
            normalized = tuple(tuple(row) for row in rows)
        except TypeError as exc:
            raise ValueError("rows must be an iterable of row iterables") from exc

        previous_length: int | None = None
        for row_index, row in enumerate(normalized):
            if not row:
                raise ValueError(f"row {row_index} must not be empty")
            if previous_length is not None and len(row) > previous_length:
                raise ValueError("row lengths must be non-increasing")
            previous_length = len(row)

            for value in row:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tableau entries must be integers")
                if not 1 <= value <= 50:
                    raise ValueError("tableau entries must be in the range 1..50")

        object.__setattr__(self, "rows", normalized)

    def transpose(self) -> Tableau:
        """Return the Ferrers-diagram transpose of this tableau."""

        if not self.rows:
            return Tableau(())

        columns = tuple(
            tuple(row[column_index] for row in self.rows if len(row) > column_index)
            for column_index in range(len(self.rows[0]))
        )
        return Tableau(columns)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):  # type annotation kept compatible with Python 3.9+
        return iter(self.rows)


def _validate_kind(kind: str) -> NotationKind:
    if kind not in _MARKERS:
        allowed = ", ".join(_MARKERS)
        raise ValueError(f"unknown notation kind {kind!r}; expected one of: {allowed}")
    return cast(NotationKind, kind)


def _format_rows(rows: tuple[tuple[int, ...], ...]) -> str:
    return " | ".join(" ".join(str(value) for value in row) for row in rows)


def _format_coordinates(tableau: Tableau) -> str:
    if len(tableau.rows) > 50 or (
        tableau.rows and len(tableau.rows[0]) > 50
    ):
        raise ValueError("coordinate indices must be in the range 1..50")
    return " | ".join(
        f"({row_index},{column_index}) : {value}"
        for row_index, row in enumerate(tableau.rows, start=1)
        for column_index, value in enumerate(row, start=1)
    )


def format_notation(tableau: Tableau, kind: NotationKind) -> str:
    """Serialize ``tableau`` using the canonical surface form for ``kind``.

    Exactly one ASCII space separates adjacent surface tokens.  ``col`` groups
    columns of the original tableau.  ``coord`` lists one-based ``(row,column)``
    entries in row-major matrix order.
    """

    if not isinstance(tableau, Tableau):
        raise TypeError("tableau must be a Tableau")
    checked_kind = _validate_kind(kind)
    start, end = _MARKERS[checked_kind]
    if checked_kind == "coord":
        body = _format_coordinates(tableau)
    else:
        displayed = tableau.transpose() if checked_kind == "col" else tableau
        body = _format_rows(displayed.rows)
    if not body:
        return f"{start} {end}"
    return f"{start} {body} {end}"


def _parse_body(body: str) -> Tableau:
    if not body:
        return Tableau(())

    row_texts = body.split(" | ")
    rows: list[tuple[int, ...]] = []
    for row_text in row_texts:
        if not row_text:
            raise ValueError("notation contains an empty row")
        fields = row_text.split(" ")
        if any(not field or not field.isascii() or not field.isdecimal() for field in fields):
            raise ValueError("rows must contain canonical decimal integers")
        if any(len(field) > 1 and field.startswith("0") for field in fields):
            raise ValueError("integers must not contain leading zeroes")
        rows.append(tuple(int(field) for field in fields))
    return Tableau(rows)


def _parse_coordinate_body(body: str) -> Tableau:
    if not body:
        return Tableau(())

    rows: list[tuple[int, ...]] = []
    current_values: list[int] = []
    current_row = 1
    expected_column = 1
    for entry_text in body.split(" | "):
        match = _COORD_ENTRY.fullmatch(entry_text)
        if match is None:
            raise ValueError(
                "coordinates must use canonical '(row,column) : value' entries"
            )
        row, column, value = (int(field) for field in match.groups())
        if row > 50 or column > 50:
            raise ValueError("coordinate indices must be in the range 1..50")

        if row == current_row and column == expected_column:
            current_values.append(value)
            expected_column += 1
            continue
        if row == current_row + 1 and column == 1 and current_values:
            rows.append(tuple(current_values))
            current_values = [value]
            current_row += 1
            expected_column = 2
            continue
        raise ValueError(
            "coordinates must be complete one-based row-major matrix entries"
        )

    if not current_values:  # pragma: no cover - non-empty body always enters the loop
        raise ValueError("coordinate notation contains no entries")
    rows.append(tuple(current_values))
    return Tableau(rows)


def parse_notation(text: str) -> tuple[Tableau, NotationKind]:
    """Parse a canonical ``raw``, ``row``, ``col``, or ``coord`` string.

    Non-canonical whitespace, unknown markers, invalid shapes, and values
    outside ``1..50`` raise :class:`ValueError` rather than being normalized.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matched_kind: NotationKind | None = None
    body: str | None = None
    for kind, (start, end) in _MARKERS.items():
        empty_form = f"{start} {end}"
        if text == empty_form:
            matched_kind = kind
            body = ""
            break

        prefix = f"{start} "
        suffix = f" {end}"
        if text.startswith(prefix) and text.endswith(suffix):
            matched_kind = kind
            body = text[len(prefix) : -len(suffix)]
            break

    if matched_kind is None or body is None:
        raise ValueError("text does not use a recognized canonical YT marker pair")

    if matched_kind == "coord":
        tableau = _parse_coordinate_body(body)
    else:
        displayed = _parse_body(body)
        tableau = displayed.transpose() if matched_kind == "col" else displayed

    # This single equality check also rejects subtle non-canonical cases such as
    # alternate Unicode digits or malformed column encodings.
    if format_notation(tableau, matched_kind) != text:
        raise ValueError("text is not in canonical YT notation")
    return tableau, matched_kind


__all__ = ["NotationKind", "Tableau", "format_notation", "parse_notation"]
