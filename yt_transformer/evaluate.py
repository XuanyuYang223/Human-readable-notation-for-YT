"""Held-out and round-trip evaluation for trained notation models."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .checkpoint import Direction, checkpoint_direction, load_checkpoint
from .data import (
    HumanKind,
    TranslationDataset,
    TranslationExample,
    generate_translation_splits,
    make_collate_fn,
)
from .infer import translate
from .model import Seq2SeqTransformer
from .notation import format_notation, parse_notation
from .runtime import resolve_device
from .tokenizer import HandmadeTokenizer


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    examples: int
    loss: float
    token_accuracy: float
    exact_match: float
    semantic_accuracy: float
    invalid_output_rate: float


def _config_value(config: Mapping[str, Any], key: str, expected: type) -> Any:
    value = config.get(key)
    if not isinstance(value, expected) or isinstance(value, bool):
        raise ValueError(f"checkpoint training_config has invalid {key!r}")
    return value


def test_examples_from_metadata(
    metadata: Mapping[str, Any],
    direction: Direction,
) -> tuple[TranslationExample, ...]:
    """Recreate the exact held-out synthetic split described by a checkpoint."""

    config = metadata.get("training_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint does not contain a training_config mapping")
    human_values = config.get("human_kinds")
    ratio_values = config.get("split_ratios")
    if not isinstance(human_values, (list, tuple)) or not human_values:
        raise ValueError("checkpoint has invalid human_kinds")
    if not isinstance(ratio_values, (list, tuple)) or len(ratio_values) != 3:
        raise ValueError("checkpoint has invalid split_ratios")
    if any(value not in ("row", "col") for value in human_values):
        raise ValueError("checkpoint has an unknown human notation kind")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ratio_values):
        raise ValueError("checkpoint has invalid split ratios")

    seed = _config_value(config, "seed", int)
    splits = generate_translation_splits(
        _config_value(config, "num_tableaux", int),
        seed=seed,
        split_seed=seed + 1,
        split_ratios=tuple(float(value) for value in ratio_values),
        directions=(direction,),
        human_kinds=cast(tuple[HumanKind, ...], tuple(human_values)),
        max_rows=_config_value(config, "max_rows", int),
        max_columns=_config_value(config, "max_columns", int),
        max_cells=_config_value(config, "max_cells", int),
    )
    if not splits["test"]:
        raise ValueError("checkpoint configuration produces an empty test split")
    return splits["test"]


def _batch_tensors(batch: Mapping[str, object], device: torch.device) -> tuple[Tensor, Tensor]:
    source = batch.get("source_ids")
    target = batch.get("target_ids")
    if not isinstance(source, Tensor) or not isinstance(target, Tensor):
        raise TypeError("batch is missing token tensors")
    return source.to(device), target.to(device)


@torch.no_grad()
def evaluate_examples(
    model: Seq2SeqTransformer,
    tokenizer: HandmadeTokenizer,
    examples: Sequence[TranslationExample],
    *,
    batch_size: int = 64,
) -> EvaluationMetrics:
    """Evaluate text, token, parse validity, and tableau semantics."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = TranslationDataset(examples, tokenizer)
    loader: DataLoader[Mapping[str, object]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )
    maximum_target = max(
        int(cast(Tensor, dataset[index]["target_ids"]).numel())
        for index in range(len(dataset))
    )
    max_new_tokens = maximum_target - 1
    if max_new_tokens > model.config.max_seq_len - 1:
        raise ValueError("evaluation target exceeds the model sequence limit")

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")
    loss_sum = 0.0
    correct_tokens = 0
    token_count = 0
    exact = 0
    semantic = 0
    invalid = 0
    seen = 0
    model.eval()

    for batch in loader:
        source, target = _batch_tensors(batch, device)
        expected_next = target[:, 1:]
        logits = model(source, target[:, :-1])
        loss_sum += float(
            criterion(logits.reshape(-1, logits.size(-1)), expected_next.reshape(-1)).item()
        )
        token_mask = expected_next.ne(tokenizer.pad_id)
        correct_tokens += int((logits.argmax(-1).eq(expected_next) & token_mask).sum().item())
        token_count += int(token_mask.sum().item())

        generated = model.greedy_decode(
            source,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.pad_id,
            max_new_tokens,
        )
        batch_examples = batch.get("examples")
        if not isinstance(batch_examples, tuple):
            raise TypeError("batch is missing TranslationExample metadata")
        for token_row, example in zip(generated, batch_examples, strict=True):
            if not isinstance(example, TranslationExample):
                raise TypeError("invalid example metadata")
            ids = [int(value) for value in token_row.tolist()]
            try:
                if tokenizer.eos_id not in ids[1:]:
                    raise ValueError("missing EOS")
                prediction = tokenizer.decode(ids)
                predicted_tableau, predicted_kind = parse_notation(prediction)
            except ValueError:
                invalid += 1
            else:
                exact += prediction == example.target
                _, expected_kind = parse_notation(example.target)
                semantic += (
                    predicted_tableau == example.tableau
                    and predicted_kind == expected_kind
                )
            seen += 1

    if seen == 0 or token_count == 0:  # pragma: no cover - guarded above
        raise RuntimeError("evaluation produced no observations")
    return EvaluationMetrics(
        examples=seen,
        loss=loss_sum / token_count,
        token_accuracy=correct_tokens / token_count,
        exact_match=exact / seen,
        semantic_accuracy=semantic / seen,
        invalid_output_rate=invalid / seen,
    )


@torch.no_grad()
def evaluate_round_trip(
    forward_model: Seq2SeqTransformer,
    reverse_model: Seq2SeqTransformer,
    tokenizer: HandmadeTokenizer,
    examples: Sequence[TranslationExample],
    *,
    limit_tableaux: int = 100,
) -> dict[str, float | int]:
    """Measure raw → human → raw exact accuracy for both human styles."""

    if limit_tableaux <= 0:
        raise ValueError("limit_tableaux must be positive")
    unique = []
    seen_keys = set()
    for example in examples:
        if example.tableau_key not in seen_keys:
            seen_keys.add(example.tableau_key)
            unique.append(example.tableau)
        if len(unique) == limit_tableaux:
            break

    attempts = 0
    correct = 0
    failures = 0
    for tableau in unique:
        raw = format_notation(tableau, "raw")
        for style in ("row", "col"):
            attempts += 1
            try:
                human = translate(
                    forward_model,
                    tokenizer,
                    "yt_to_human",
                    raw,
                    style=cast(HumanKind, style),
                )
                reconstructed = translate(
                    reverse_model,
                    tokenizer,
                    "human_to_yt",
                    human,
                )
            except ValueError:
                failures += 1
                continue
            correct += reconstructed == raw
    return {
        "attempts": attempts,
        "exact_match": correct / attempts if attempts else 0.0,
        "invalid_pipeline_rate": failures / attempts if attempts else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or both YT notation checkpoints on held-out data."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="repeat for the second direction to also measure round-trip accuracy",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="limit held-out examples per direction")
    parser.add_argument("--round-trip-limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(args.checkpoint) not in (1, 2):
        raise SystemExit("--checkpoint must be supplied once or twice")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    device = resolve_device(args.device)
    loaded: dict[Direction, tuple[Seq2SeqTransformer, HandmadeTokenizer, dict[str, Any]]] = {}
    report: dict[str, object] = {"device": str(device), "directions": {}}

    for path in args.checkpoint:
        model, tokenizer, metadata = load_checkpoint(path, device=device)
        direction = checkpoint_direction(metadata)
        if direction in loaded:
            raise SystemExit(f"duplicate {direction} checkpoint")
        loaded[direction] = (model, tokenizer, metadata)
        examples = test_examples_from_metadata(metadata, direction)
        if args.limit is not None:
            examples = examples[: args.limit]
        metrics = evaluate_examples(
            model, tokenizer, examples, batch_size=args.batch_size
        )
        cast(dict[str, object], report["directions"])[direction] = asdict(metrics)

    if set(loaded) == {"yt_to_human", "human_to_yt"}:
        forward, tokenizer, forward_metadata = loaded["yt_to_human"]
        reverse, reverse_tokenizer, _ = loaded["human_to_yt"]
        if tokenizer.vocab != reverse_tokenizer.vocab:  # defensive; loaders validate current vocab
            raise ValueError("forward and reverse tokenizer vocabularies differ")
        forward_examples = test_examples_from_metadata(
            forward_metadata, "yt_to_human"
        )
        report["round_trip"] = evaluate_round_trip(
            forward,
            reverse,
            tokenizer,
            forward_examples,
            limit_tableaux=args.round_trip_limit,
        )

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EvaluationMetrics",
    "evaluate_examples",
    "evaluate_round_trip",
    "main",
    "test_examples_from_metadata",
]
