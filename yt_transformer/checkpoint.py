"""Versioned, self-describing checkpoints for the two translation models."""

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


Direction: TypeAlias = Literal["yt_to_human", "human_to_yt"]
CHECKPOINT_VERSION = 1


def _checked_direction(value: object) -> Direction:
    if value not in ("yt_to_human", "human_to_yt"):
        raise ValueError(f"invalid checkpoint direction: {value!r}")
    return cast(Direction, value)


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
    _checked_direction(payload.get("direction"))

    tokenizer = HandmadeTokenizer()
    if payload.get("tokenizer_vocab") != list(tokenizer.vocab):
        raise ValueError("checkpoint tokenizer vocabulary does not match this code")

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
    metadata = dict(payload)
    del metadata["model_state_dict"]
    return model, tokenizer, metadata


def checkpoint_direction(payload: Mapping[str, Any]) -> Direction:
    """Return a validated direction from an already loaded payload."""

    return _checked_direction(payload.get("direction"))


__all__ = [
    "CHECKPOINT_VERSION",
    "Direction",
    "checkpoint_direction",
    "load_checkpoint",
    "save_checkpoint",
]
