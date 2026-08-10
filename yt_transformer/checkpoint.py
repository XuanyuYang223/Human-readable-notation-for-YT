"""Versioned, self-describing checkpoints for the supported sequence tasks."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, TypeAlias, cast

import torch

from .model import ModelConfig, Seq2SeqTransformer
from .runtime import resolve_device
from .tokenizer import HandmadeTokenizer


Direction: TypeAlias = Literal["yt_to_human", "human_to_yt", "perm_to_yt"]
HumanKind: TypeAlias = Literal["row", "col", "coord"]
CHECKPOINT_VERSION = 1


def _checked_direction(value: object) -> Direction:
    if value not in ("yt_to_human", "human_to_yt", "perm_to_yt"):
        raise ValueError(f"invalid checkpoint direction: {value!r}")
    return cast(Direction, value)


def _validate_direction_tokenizer(
    direction: Direction, tokenizer: HandmadeTokenizer
) -> None:
    """Reject task metadata that the stored tokenizer cannot represent."""

    if direction == "perm_to_yt" and not {"x9", "x10"}.issubset(
        tokenizer.token_to_id
    ):
        raise ValueError(
            "perm_to_yt checkpoints require an RSK-aware tokenizer with x9/x10"
        )


def checkpoint_human_kinds(payload: Mapping[str, Any]) -> tuple[HumanKind, ...]:
    """Return the validated human styles recorded in checkpoint metadata."""

    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("checkpoint does not contain a training_config mapping")
    values = training_config.get("human_kinds")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("checkpoint has invalid human_kinds")
    if any(value not in ("row", "col", "coord") for value in values):
        raise ValueError("checkpoint has an unknown human notation kind")
    if len(set(values)) != len(values):
        raise ValueError("checkpoint has duplicate human notation kinds")
    return cast(tuple[HumanKind, ...], tuple(values))


def _portable_value(value: object, *, field: str) -> object:
    """Normalize metadata to values accepted by weights-only torch loading."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _portable_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings")
            result[key] = _portable_value(item, field=f"{field}.{key}")
        return result
    raise TypeError(
        f"{field} contains unsupported metadata type {type(value).__name__}; "
        "use JSON-like values"
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: Seq2SeqTransformer,
    tokenizer: HandmadeTokenizer,
    direction: Direction,
    epoch: int,
    metrics: Mapping[str, float],
    training_config: Mapping[str, Any],
) -> None:
    """Atomically save a model and all metadata needed for inference."""

    checked_direction = _checked_direction(direction)
    _validate_direction_tokenizer(checked_direction, tokenizer)
    if model.config.src_vocab_size != tokenizer.vocab_size:
        raise ValueError("source vocabulary size does not match the tokenizer")
    if model.config.tgt_vocab_size != tokenizer.vocab_size:
        raise ValueError("target vocabulary size does not match the tokenizer")
    if model.config.pad_id != tokenizer.pad_id:
        raise ValueError("model padding ID does not match the tokenizer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    checked_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("metric names must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {key!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"metric {key!r} must be finite")
        checked_metrics[key] = numeric
    checked_training_config = _portable_value(
        training_config, field="training_config"
    )
    if not isinstance(checked_training_config, dict):  # pragma: no cover - Mapping input
        raise TypeError("training_config must be a mapping")
    if "human_kinds" in checked_training_config:
        supported_kinds = checkpoint_human_kinds(
            {"training_config": checked_training_config}
        )
        if "coord" in supported_kinds and "TO_COORD" not in tokenizer.token_to_id:
            raise ValueError(
                "coordinate training metadata requires a coordinate-aware tokenizer"
            )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "direction": checked_direction,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "tokenizer_vocab": list(tokenizer.vocab),
        "epoch": epoch,
        "metrics": checked_metrics,
        "training_config": checked_training_config,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Seq2SeqTransformer, HandmadeTokenizer, dict[str, Any]]:
    """Load and validate a checkpoint, returning an eval-mode model."""

    checkpoint_path = Path(path)
    resolved_device = resolve_device(str(device))
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except FileNotFoundError:
        raise
    except Exception as exc:  # torch's serialization errors vary by version
        raise ValueError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dictionary")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('checkpoint_version')!r}"
        )
    checked_direction = _checked_direction(payload.get("direction"))

    stored_vocab = payload.get("tokenizer_vocab")
    try:
        tokenizer = HandmadeTokenizer(vocab=stored_vocab)
    except (TypeError, ValueError):
        raise ValueError(
            "checkpoint tokenizer vocabulary does not match this code"
        ) from None
    if stored_vocab != list(tokenizer.vocab):
        raise ValueError("checkpoint tokenizer vocabulary does not match this code")
    _validate_direction_tokenizer(checked_direction, tokenizer)
    supported_kinds: tuple[HumanKind, ...] | None = None
    training_config = payload.get("training_config")
    if isinstance(training_config, Mapping) and "human_kinds" in training_config:
        supported_kinds = checkpoint_human_kinds(payload)
        if "coord" in supported_kinds and "TO_COORD" not in tokenizer.token_to_id:
            raise ValueError(
                "coordinate training metadata requires a coordinate-aware tokenizer"
            )

    config_data = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(config_data, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing model configuration or weights")
    try:
        config = ModelConfig.from_dict(config_data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint has an invalid model configuration: {exc}") from exc
    if config.src_vocab_size != tokenizer.vocab_size:
        raise ValueError("source vocabulary size does not match the tokenizer")
    if config.tgt_vocab_size != tokenizer.vocab_size:
        raise ValueError("target vocabulary size does not match the tokenizer")
    if config.pad_id != tokenizer.pad_id:
        raise ValueError("checkpoint padding ID does not match the tokenizer")

    model = Seq2SeqTransformer(config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"checkpoint weights do not match its configuration: {exc}") from exc
    model.to(resolved_device)
    model.eval()
    if supported_kinds is not None:
        model.supported_human_kinds = supported_kinds
    metadata = dict(payload)
    del metadata["model_state_dict"]
    return model, tokenizer, metadata


def checkpoint_direction(payload: Mapping[str, Any]) -> Direction:
    """Return a validated direction from an already loaded payload."""

    return _checked_direction(payload.get("direction"))


__all__ = [
    "CHECKPOINT_VERSION",
    "Direction",
    "HumanKind",
    "checkpoint_direction",
    "checkpoint_human_kinds",
    "load_checkpoint",
    "save_checkpoint",
]
