"""Held-out evaluation for permutation-to-RSK-tableau checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .checkpoint import checkpoint_direction, load_checkpoint
from .model import Seq2SeqTransformer
from .notation import parse_notation
from .rsk_data import (
    RSKDataset,
    RSKExample,
    generate_rsk_splits,
    make_rsk_collate_fn,
)
from .runtime import resolve_device
from .tokenizer import HandmadeTokenizer


@dataclass(frozen=True, slots=True)
class RSKEvaluationMetrics:
    """Token, surface, and tableau-level metrics for one example group."""

    examples: int
    loss: float
    token_accuracy: float
    exact_match: float
    semantic_accuracy: float
    invalid_output_rate: float
    shape_exact_match: float
    content_preservation: float


@dataclass(frozen=True, slots=True)
class RSKEvaluationReport:
    """Overall metrics plus the same metrics grouped by permutation length."""

    overall: RSKEvaluationMetrics
    by_length: dict[int, RSKEvaluationMetrics]


@dataclass(slots=True)
class _MetricCounts:
    examples: int = 0
    loss_sum: float = 0.0
    correct_tokens: int = 0
    token_count: int = 0
    exact: int = 0
    semantic: int = 0
    invalid: int = 0
    shape_exact: int = 0
    content_preserved: int = 0

    def metrics(self) -> RSKEvaluationMetrics:
        if self.examples <= 0 or self.token_count <= 0:
            raise RuntimeError("RSK evaluation group contains no observations")
        return RSKEvaluationMetrics(
            examples=self.examples,
            loss=self.loss_sum / self.token_count,
            token_accuracy=self.correct_tokens / self.token_count,
            exact_match=self.exact / self.examples,
            semantic_accuracy=self.semantic / self.examples,
            invalid_output_rate=self.invalid / self.examples,
            shape_exact_match=self.shape_exact / self.examples,
            content_preservation=self.content_preserved / self.examples,
        )


def _required_int(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"checkpoint has invalid {key!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"checkpoint has invalid {key!r}")
    return value


def rsk_test_examples_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[RSKExample, ...]:
    """Strictly reconstruct the P-group-disjoint held-out RSK split."""

    config = metadata.get("training_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint does not contain a training_config mapping")

    count = _required_int(config, "num_permutations", minimum=3)
    min_length = _required_int(config, "min_length", minimum=1, maximum=50)
    max_length = _required_int(config, "max_length", minimum=1, maximum=50)
    if min_length > max_length:
        raise ValueError("checkpoint has min_length greater than max_length")
    seed = _required_int(config, "seed", minimum=0, maximum=2**63 - 1)
    split_seed = _required_int(
        config, "split_seed", minimum=0, maximum=2**63 - 1
    )

    ratio_values = config.get("split_ratios")
    if not isinstance(ratio_values, (list, tuple)) or len(ratio_values) != 3:
        raise ValueError("checkpoint has invalid split_ratios")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        for value in ratio_values
    ):
        raise ValueError("checkpoint has invalid split_ratios")
    split_ratios = cast(
        tuple[float, float, float], tuple(float(value) for value in ratio_values)
    )

    splits = generate_rsk_splits(
        count,
        seed=seed,
        split_seed=split_seed,
        min_length=min_length,
        max_length=max_length,
        split_ratios=split_ratios,
    )
    if not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}:
        raise RuntimeError("RSK split generator returned invalid split names")
    for name in ("train", "val", "test"):
        if not isinstance(splits[name], tuple) or any(
            not isinstance(example, RSKExample) for example in splits[name]
        ):
            raise RuntimeError(f"RSK split {name!r} contains invalid examples")

    all_examples = tuple(
        example
        for name in ("train", "val", "test")
        for example in splits[name]
    )
    if len(all_examples) != count or len(
        {example.permutation_key for example in all_examples}
    ) != count:
        raise RuntimeError("RSK split generator did not preserve every unique source")
    if any(not min_length <= example.length <= max_length for example in all_examples):
        raise RuntimeError("RSK split generator returned a length outside metadata bounds")

    group_keys = {
        name: {example.tableau_key for example in splits[name]}
        for name in ("train", "val", "test")
    }
    if (
        group_keys["train"] & group_keys["val"]
        or group_keys["train"] & group_keys["test"]
        or group_keys["val"] & group_keys["test"]
    ):
        raise RuntimeError("RSK split generator leaked a P-tableau group across splits")

    test_examples = cast(tuple[RSKExample, ...], splits["test"])
    if not test_examples:
        raise ValueError("reconstructed RSK test split is empty")
    return test_examples


def _batch_tensors(
    batch: Mapping[str, object], device: torch.device
) -> tuple[Tensor, Tensor, tuple[RSKExample, ...]]:
    source = batch.get("source_ids")
    target = batch.get("target_ids")
    examples = batch.get("examples")
    if not isinstance(source, Tensor) or not isinstance(target, Tensor):
        raise TypeError("collated RSK batch is missing token tensors")
    if not isinstance(examples, tuple) or any(
        not isinstance(example, RSKExample) for example in examples
    ):
        raise TypeError("collated RSK batch is missing example metadata")
    return source.to(device), target.to(device), cast(tuple[RSKExample, ...], examples)


@torch.no_grad()
def evaluate_rsk_examples(
    model: Seq2SeqTransformer,
    tokenizer: HandmadeTokenizer,
    examples: Sequence[RSKExample],
    *,
    batch_size: int = 64,
) -> RSKEvaluationReport:
    """Evaluate RSK generation overall and separately for every input length."""

    if not examples:
        raise ValueError("evaluation requires at least one RSK example")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    dataset = RSKDataset(examples, tokenizer)
    maximum_source = max(
        int(cast(Tensor, dataset[index]["source_ids"]).numel())
        for index in range(len(dataset))
    )
    maximum_target = max(
        int(cast(Tensor, dataset[index]["target_ids"]).numel())
        for index in range(len(dataset))
    )
    if max(maximum_source, maximum_target) > model.config.max_seq_len:
        raise ValueError("RSK evaluation example exceeds the model sequence limit")
    max_new_tokens = maximum_target - 1

    loader: DataLoader[Mapping[str, object]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_rsk_collate_fn(tokenizer.pad_id),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="none")
    device = next(model.parameters()).device
    overall = _MetricCounts()
    by_length: dict[int, _MetricCounts] = {}
    model.eval()

    for batch in loader:
        source, target, batch_examples = _batch_tensors(batch, device)
        expected_next = target[:, 1:]
        logits = model(source, target[:, :-1])
        token_losses = criterion(
            logits.reshape(-1, logits.size(-1)), expected_next.reshape(-1)
        ).reshape_as(expected_next)
        token_mask = expected_next.ne(tokenizer.pad_id)
        predictions = logits.argmax(dim=-1)
        generated = model.greedy_decode(
            source,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.pad_id,
            max_new_tokens,
        )

        for index, (token_row, example) in enumerate(
            zip(generated, batch_examples, strict=True)
        ):
            length_counts = by_length.setdefault(example.length, _MetricCounts())
            row_mask = token_mask[index]
            row_token_count = int(row_mask.sum().item())
            row_loss = float(token_losses[index][row_mask].sum().item())
            row_correct = int(
                (predictions[index].eq(expected_next[index]) & row_mask).sum().item()
            )
            for counts in (overall, length_counts):
                counts.examples += 1
                counts.loss_sum += row_loss
                counts.correct_tokens += row_correct
                counts.token_count += row_token_count

            generated_ids = [int(value) for value in token_row.tolist()]
            exact = semantic = invalid = shape_exact = content_preserved = 0
            try:
                if tokenizer.eos_id not in generated_ids[1:]:
                    raise ValueError("missing EOS")
                prediction = tokenizer.decode(generated_ids)
                predicted_tableau, _ = parse_notation(prediction)
            except ValueError:
                invalid = 1
            else:
                exact = int(prediction == example.target)
                semantic = int(predicted_tableau == example.tableau)
                predicted_shape = tuple(len(row) for row in predicted_tableau.rows)
                expected_shape = tuple(len(row) for row in example.tableau.rows)
                shape_exact = int(predicted_shape == expected_shape)
                predicted_content = sorted(
                    value for row in predicted_tableau.rows for value in row
                )
                content_preserved = int(
                    predicted_content == sorted(example.permutation)
                )
            for counts in (overall, length_counts):
                counts.exact += exact
                counts.semantic += semantic
                counts.invalid += invalid
                counts.shape_exact += shape_exact
                counts.content_preserved += content_preserved

    return RSKEvaluationReport(
        overall=overall.metrics(),
        by_length={
            length: counts.metrics() for length, counts in sorted(by_length.items())
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one permutation-to-RSK-tableau checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="limit reconstructed test examples")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    device = resolve_device(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device=device)
    direction = checkpoint_direction(metadata)
    if direction != "perm_to_yt":
        raise SystemExit("checkpoint direction must be perm_to_yt")
    examples = rsk_test_examples_from_metadata(metadata)
    if args.limit is not None:
        examples = examples[: args.limit]
    report = evaluate_rsk_examples(
        model, tokenizer, examples, batch_size=args.batch_size
    )
    output = {
        "device": str(device),
        "direction": direction,
        "overall": asdict(report.overall),
        "by_length": {
            str(length): asdict(metrics)
            for length, metrics in report.by_length.items()
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "RSKEvaluationMetrics",
    "RSKEvaluationReport",
    "evaluate_rsk_examples",
    "main",
    "rsk_test_examples_from_metadata",
]
