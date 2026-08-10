"""A fixed, hand-made tokenizer for canonical YT and permutation notation.

This module intentionally has no dependency on a tokenizer library.  Surface
markers and punctuation are mapped to the symbolic tokens from the project
notes, while each integer in ``1..50`` is represented atomically.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence, TypeAlias

from .notation import format_notation, parse_notation
from .rsk import format_permutation, parse_permutation


TaskKind: TypeAlias = Literal["row", "col", "coord"]

_LEGACY_SPECIAL_TOKENS: tuple[str, ...] = ("PAD", "BOS", "EOS", "TO_ROW", "TO_COL")
_LEGACY_MARKER_TOKENS: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")
_LEGACY_STRUCTURE_TOKENS: tuple[str, ...] = ("s", "x")

SPECIAL_TOKENS: tuple[str, ...] = _LEGACY_SPECIAL_TOKENS + ("TO_COORD",)
MARKER_TOKENS: tuple[str, ...] = _LEGACY_MARKER_TOKENS + ("x7", "x8")
STRUCTURE_TOKENS: tuple[str, ...] = _LEGACY_STRUCTURE_TOKENS + (
    "lparen",
    "comma",
    "rparen",
    "colon",
)
NUMBER_TOKENS: tuple[str, ...] = tuple(f"n{value}" for value in range(1, 51))

# Preserve every ID from the original 63-token tokenizer.  Coordinate-specific
# control, marker, and punctuation tokens are appended so legacy token IDs stay
# stable and the checkpoint loader can select either exact vocabulary.
LEGACY_VOCAB: tuple[str, ...] = (
    _LEGACY_SPECIAL_TOKENS
    + _LEGACY_MARKER_TOKENS
    + _LEGACY_STRUCTURE_TOKENS
    + NUMBER_TOKENS
)
VOCAB: tuple[str, ...] = LEGACY_VOCAB + (
    "TO_COORD",
    "x7",
    "x8",
    "lparen",
    "comma",
    "rparen",
    "colon",
)
# Keep the coordinate-aware 70-token vocabulary as the default.  RSK models
# opt into this append-only superset, preserving every established token ID.
RSK_VOCAB: tuple[str, ...] = VOCAB + ("x9", "x10")

_SURFACE_TO_TOKEN: Mapping[str, str] = MappingProxyType(
    {
        "[YT row start]": "x1",
        "[YT row end]": "x2",
        "[YT col start]": "x3",
        "[YT col end]": "x4",
        "[YT start]": "x5",
        "[YT end]": "x6",
        "[YT coord start]": "x7",
        "[YT coord end]": "x8",
        "[perm start]": "x9",
        "[perm end]": "x10",
        " ": "s",
        "|": "x",
        "(": "lparen",
        ",": "comma",
        ")": "rparen",
        ":": "colon",
        **{str(value): f"n{value}" for value in range(1, 51)},
    }
)
_TOKEN_TO_SURFACE: Mapping[str, str] = MappingProxyType(
    {token: surface for surface, token in _SURFACE_TO_TOKEN.items()}
)
_MARKER_SURFACES: tuple[str, ...] = tuple(
    surface for surface in _SURFACE_TO_TOKEN if surface.startswith("[")
)
_CONTROL_TOKENS = frozenset(SPECIAL_TOKENS)


def _validate_surface(text: str) -> str:
    """Validate one supported canonical surface and return its kind."""

    if isinstance(text, str) and text.startswith("[perm"):
        permutation = parse_permutation(text)
        if format_permutation(permutation) != text:  # defensive and explicit
            raise ValueError("text is not canonical permutation notation")
        return "perm"

    tableau, kind = parse_notation(text)
    if format_notation(tableau, kind) != text:  # defensive and explicit
        raise ValueError("text is not canonical YT notation")
    return kind


class HandmadeTokenizer:
    """Tokenizer with a stable, fixed vocabulary and no learned state.

    Vocabulary IDs are stable by construction: ``PAD`` is 0, ``BOS`` is 1,
    ``EOS`` is 2, ``TO_ROW`` is 3, and ``TO_COL`` is 4.  The original marker,
    structure, and number IDs remain unchanged; coordinate tokens occupy IDs
    63 through 69.
    """

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        """Create the current tokenizer, or a supported legacy variant.

        Passing a vocabulary selects a task-specific/checkpoint-compatible
        variant.  Only the original 63-token vocabulary, the coordinate-aware
        70-token vocabulary, and its 72-token RSK superset are accepted, so
        arbitrary checkpoint token layouts cannot silently acquire the wrong
        surface semantics.
        """

        if vocab is None:
            selected_vocab = VOCAB
        else:
            if isinstance(vocab, (str, bytes)):
                raise TypeError("vocab must be a sequence of token strings")
            selected_vocab = tuple(vocab)
            if selected_vocab not in (LEGACY_VOCAB, VOCAB, RSK_VOCAB):
                raise ValueError("unsupported tokenizer vocabulary")

        self._vocab = selected_vocab
        token_to_id = {token: index for index, token in enumerate(self._vocab)}
        self._token_to_id: Mapping[str, int] = MappingProxyType(token_to_id)
        self._id_to_token: Mapping[int, str] = MappingProxyType(
            {index: token for token, index in token_to_id.items()}
        )

    @property
    def vocab(self) -> tuple[str, ...]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def token_to_id(self) -> Mapping[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> Mapping[int, str]:
        return self._id_to_token

    @property
    def pad_id(self) -> int:
        return self._token_to_id["PAD"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["BOS"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["EOS"]

    @property
    def to_row_id(self) -> int:
        return self._token_to_id["TO_ROW"]

    @property
    def to_col_id(self) -> int:
        return self._token_to_id["TO_COL"]

    @property
    def to_coord_id(self) -> int:
        return self.token_id("TO_COORD")

    # Conventional aliases make the tokenizer convenient in model/data code.
    pad_token_id = pad_id
    bos_token_id = bos_id
    eos_token_id = eos_id

    def token_id(self, token: str) -> int:
        """Return the ID for one symbolic token, rejecting unknown tokens."""

        try:
            return self._token_to_id[token]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown token: {token!r}") from exc

    def id_token(self, token_id: int) -> str:
        """Return the symbolic token for one ID, rejecting unknown IDs."""

        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"token ID must be an integer, got {token_id!r}")
        try:
            return self._id_to_token[token_id]
        except KeyError as exc:
            raise ValueError(f"unknown token ID: {token_id}") from exc

    def tokenize(self, text: str) -> list[str]:
        """Convert canonical notation text to symbolic tokens.

        Validation happens before scanning, so alternate whitespace, unknown
        numbers, and mismatched marker pairs are never silently normalized.
        """

        kind = _validate_surface(text)

        tokens: list[str] = []
        position = 0
        while position < len(text):
            marker = next(
                (surface for surface in _MARKER_SURFACES if text.startswith(surface, position)),
                None,
            )
            if marker is not None:
                tokens.append(_SURFACE_TO_TOKEN[marker])
                position += len(marker)
                continue

            character = text[position]
            if character in {" ", "|", "(", ",", ")", ":"}:
                tokens.append(_SURFACE_TO_TOKEN[character])
                position += 1
                continue

            if character.isascii() and character.isdecimal():
                end = position + 1
                while end < len(text) and text[end].isascii() and text[end].isdecimal():
                    end += 1
                number = text[position:end]
                try:
                    tokens.append(_SURFACE_TO_TOKEN[number])
                except KeyError as exc:
                    raise ValueError(f"number has no fixed token: {number!r}") from exc
                position = end
                continue

            raise ValueError(f"unknown surface text at character {position}")
        if any(token not in self._token_to_id for token in tokens):
            raise ValueError(
                f"tokenizer vocabulary does not support {kind!r} notation"
            )
        return tokens

    def detokenize(self, tokens: Iterable[str]) -> str:
        """Convert surface tokens back to canonical notation text.

        Control tokens such as ``BOS`` have no surface representation and must
        be removed first (``decode`` does this by default).
        """

        if isinstance(tokens, str):
            raise TypeError("tokens must be an iterable of token strings, not one string")

        token_list = list(tokens)
        surfaces: list[str] = []
        for token in token_list:
            if not isinstance(token, str):
                raise ValueError(f"token must be a string, got {token!r}")
            try:
                surfaces.append(_TOKEN_TO_SURFACE[token])
            except KeyError as exc:
                if token in _CONTROL_TOKENS:
                    raise ValueError(
                        f"control token {token!r} has no surface representation"
                    ) from exc
                raise ValueError(f"unknown token: {token!r}") from exc
        text = "".join(surfaces)
        _validate_surface(text)
        if self.tokenize(text) != token_list:
            raise ValueError("tokens do not form canonical notation")
        return text

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> list[int]:
        if isinstance(tokens, str):
            raise TypeError("tokens must be an iterable of token strings, not one string")
        return [self.token_id(token) for token in tokens]

    def convert_ids_to_tokens(self, token_ids: Iterable[int]) -> list[str]:
        if isinstance(token_ids, (str, bytes)):
            raise TypeError("token_ids must be an iterable of integers")
        return [self.id_token(token_id) for token_id in token_ids]

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        task: TaskKind | None = None,
    ) -> list[int]:
        """Encode notation into IDs, optionally prefixing an output task.

        ``task="row"`` inserts ``TO_ROW``, ``task="col"`` inserts ``TO_COL``,
        and ``task="coord"`` inserts ``TO_COORD``.  With wrappers, the order is
        ``BOS, TO_*, <surface tokens>, EOS``.
        """

        surface_tokens = self.tokenize(text)
        task_token: str | None
        if task is None:
            task_token = None
        elif task == "row":
            task_token = "TO_ROW"
        elif task == "col":
            task_token = "TO_COL"
        elif task == "coord":
            task_token = "TO_COORD"
        else:
            raise ValueError("task must be 'row', 'col', 'coord', or None")

        tokens: list[str] = []
        if add_special_tokens:
            tokens.append("BOS")
        if task_token is not None:
            tokens.append(task_token)
        tokens.extend(surface_tokens)
        if add_special_tokens:
            tokens.append("EOS")
        return self.convert_tokens_to_ids(tokens)

    def decode(
        self,
        token_ids: Sequence[int] | Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode IDs into one canonical notation string.

        By default ``PAD``, ``BOS``, ``EOS``, and direction tokens are ignored.
        All remaining IDs must form exactly one canonical notation string.
        """

        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in _CONTROL_TOKENS]
        return self.detokenize(tokens)


# Friendly aliases for callers that prefer a domain-specific or concise name.
NotationTokenizer = HandmadeTokenizer
Tokenizer = HandmadeTokenizer


__all__ = [
    "HandmadeTokenizer",
    "LEGACY_VOCAB",
    "MARKER_TOKENS",
    "NUMBER_TOKENS",
    "NotationTokenizer",
    "RSK_VOCAB",
    "SPECIAL_TOKENS",
    "STRUCTURE_TOKENS",
    "TaskKind",
    "Tokenizer",
    "VOCAB",
]
