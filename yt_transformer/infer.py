"""Greedy inference for a trained YT-notation Transformer checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

from .checkpoint import Direction, checkpoint_direction, load_checkpoint
from .data import HumanKind
from .model import Seq2SeqTransformer
from .notation import parse_notation
from .runtime import resolve_device
from .tokenizer import HandmadeTokenizer


@torch.no_grad()
def translate(
    model: Seq2SeqTransformer,
    tokenizer: HandmadeTokenizer,
    direction: Direction,
    text: str,
    *,
    style: HumanKind = "row",
    max_new_tokens: int | None = None,
) -> str:
    """Translate one canonical notation string with a trained model."""

    if direction == "perm_to_yt":
        raise ValueError(
            "perm_to_yt uses infer_rsk_tableau (or the yt-rsk-infer command)"
        )
    _, source_kind = parse_notation(text)
    supported_kinds = getattr(model, "supported_human_kinds", None)
    if direction == "yt_to_human":
        if source_kind != "raw":
            raise ValueError("yt_to_human expects input wrapped by [YT start]/[YT end]")
        if style not in ("row", "col", "coord"):
            raise ValueError("style must be row, col, or coord")
        if supported_kinds is not None and style not in supported_kinds:
            raise ValueError(
                f"checkpoint was not trained to generate {style!r} notation"
            )
        task: HumanKind | None = style
        expected_kind = style
    elif direction == "human_to_yt":
        if source_kind not in ("row", "col", "coord"):
            raise ValueError(
                "human_to_yt expects row, column, or coordinate notation"
            )
        if supported_kinds is not None and source_kind not in supported_kinds:
            raise ValueError(
                f"checkpoint was not trained to read {source_kind!r} notation"
            )
        task = None
        expected_kind = "raw"
    else:  # pragma: no cover - Direction plus checkpoint validation
        raise ValueError(f"unknown direction {direction!r}")

    source_ids = tokenizer.encode(text, task=task)
    if len(source_ids) > model.config.max_seq_len:
        raise ValueError(
            f"encoded input has {len(source_ids)} tokens but the model limit is "
            f"{model.config.max_seq_len}"
        )
    generation_limit = (
        model.config.max_seq_len - 1 if max_new_tokens is None else max_new_tokens
    )
    if generation_limit <= 0 or generation_limit > model.config.max_seq_len - 1:
        raise ValueError(
            f"max_new_tokens must be in 1..{model.config.max_seq_len - 1}"
        )
    device = next(model.parameters()).device
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    generated = model.greedy_decode(
        source,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.pad_id,
        generation_limit,
    )[0]
    generated_ids = [int(value) for value in generated.tolist()]
    if tokenizer.eos_id not in generated_ids[1:]:
        symbolic = tokenizer.convert_ids_to_tokens(generated_ids)
        raise ValueError(
            "model did not emit EOS before the generation limit; "
            f"generated tokens: {' '.join(symbolic)}"
        )
    try:
        prediction = tokenizer.decode(generated_ids)
        _, prediction_kind = parse_notation(prediction)
    except ValueError as exc:
        symbolic = tokenizer.convert_ids_to_tokens(generated_ids)
        raise ValueError(
            "model emitted invalid notation; "
            f"generated tokens: {' '.join(symbolic)}"
        ) from exc
    if prediction_kind != expected_kind:
        raise ValueError(
            f"model emitted {prediction_kind!r} notation, expected {expected_kind!r}"
        )
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate one YT notation string.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True, help="canonical input notation")
    parser.add_argument(
        "--style",
        choices=("row", "col", "coord"),
        default="row",
        help="output style for a yt_to_human checkpoint",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device=device)
    direction = checkpoint_direction(metadata)
    result = translate(
        model,
        tokenizer,
        direction,
        args.text,
        style=cast(HumanKind, args.style),
        max_new_tokens=args.max_new_tokens,
    )
    print(result)


if __name__ == "__main__":
    main()


__all__ = ["main", "translate"]
