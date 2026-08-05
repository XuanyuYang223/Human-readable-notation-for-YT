"""Small runtime helpers shared by training, inference, and evaluation."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch without requiring NumPy."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` or validate a supported execution device."""

    if not isinstance(requested, str) or not requested:
        raise ValueError("device must be a non-empty string")
    if requested != "auto":
        try:
            device = torch.device(requested)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid torch device {requested!r}") from exc
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("only cpu, cuda, and mps execution devices are supported")
        if device.type == "cpu" and device.index not in (None, 0):
            raise ValueError("CPU device indices other than 0 are not supported")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available")
            if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
                raise ValueError(f"CUDA device index {device.index} is not available")
        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS was requested but is not available")
            if device.index not in (None, 0):
                raise ValueError("MPS device indices other than 0 are not supported")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["resolve_device", "seed_everything"]
