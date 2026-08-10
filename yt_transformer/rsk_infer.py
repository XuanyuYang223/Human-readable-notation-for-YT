"""Greedy permutation-to-tableau inference for an RSK checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from .checkpoint import checkpoint_direction, load_checkpoint
from .model import Seq2SeqTransformer
from .notation import parse_notation
from .rsk import parse_permutation
from .runtime import resolve_device
from .tokenizer import HandmadeTokenizer


@torch.no_grad()
def infer_rsk_tableau(
    model: Seq2SeqTransformer,
    tokenizer: HandmadeTokenizer,
    text: str,
    *,
    max_new_tokens: int | None = None,
) -> str:
    """Generate the RSK insertion tableau ``P`` for one permutation."""

    parse_permutation(text)
    source_ids = tokenizer.encode(text)
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
            "model emitted invalid YT notation; "
            f"generated tokens: {' '.join(symbolic)}"
        ) from exc
    if prediction_kind != "raw":
        raise ValueError(
            f"model emitted {prediction_kind!r} notation, expected 'raw'"
        )
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an RSK insertion tableau P from a permutation."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--text",
        required=True,
        help="canonical [perm start] ... [perm end] input",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device=device)
    if checkpoint_direction(metadata) != "perm_to_yt":
        raise SystemExit("checkpoint direction must be perm_to_yt")
    print(
        infer_rsk_tableau(
            model,
            tokenizer,
            args.text,
            max_new_tokens=args.max_new_tokens,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["infer_rsk_tableau", "main"]
