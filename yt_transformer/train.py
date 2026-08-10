"""Train the two small, independent notation translation Transformers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .checkpoint import Direction, save_checkpoint
from .data import (
    DEFAULT_HUMAN_KINDS,
    HumanKind,
    TranslationDataset,
    generate_translation_splits,
    make_collate_fn,
)
from .model import ModelConfig, Seq2SeqTransformer
from .runtime import resolve_device, seed_everything
from .tokenizer import HandmadeTokenizer


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """All reproducibility-relevant training and synthetic-data settings."""

    num_tableaux: int = 4_000
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    human_kinds: tuple[HumanKind, ...] = DEFAULT_HUMAN_KINDS
    max_rows: int = 5
    max_columns: int = 8
    max_cells: int = 20
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 5
    val_exact_limit: int = 256
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_seq_len: int = 256
    tie_embeddings: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        def is_int(value: object) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        if not is_int(self.num_tableaux) or self.num_tableaux < 3:
            raise ValueError("num_tableaux must be at least 3")
        if len(self.split_ratios) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in self.split_ratios
        ):
            raise ValueError("split_ratios must contain three positive finite values")
        if (
            not self.human_kinds
            or any(kind not in DEFAULT_HUMAN_KINDS for kind in self.human_kinds)
            or len(set(self.human_kinds)) != len(self.human_kinds)
        ):
            raise ValueError("human_kinds must contain row, col, and/or coord")
        if any(
            not is_int(value) or value <= 0
            for value in (self.max_rows, self.max_columns, self.max_cells)
        ):
            raise ValueError("shape limits must be positive")
        if self.max_cells > self.max_rows * self.max_columns:
            raise ValueError("max_cells cannot exceed max_rows * max_columns")
        if "coord" in self.human_kinds and (
            min(self.max_rows, self.max_cells) > 50
            or min(self.max_columns, self.max_cells) > 50
        ):
            raise ValueError("coordinate row and column indices must fit in 1..50")
        if any(
            not is_int(value) or value <= 0
            for value in (self.epochs, self.batch_size)
        ):
            raise ValueError("epochs and batch_size must be positive")
        numeric_optimizer_values = (
            self.learning_rate,
            self.weight_decay,
            self.grad_clip,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_optimizer_values
        ) or self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("optimizer settings are invalid")
        if not is_int(self.patience) or self.patience < 0:
            raise ValueError("patience must be non-negative")
        if not is_int(self.val_exact_limit) or self.val_exact_limit <= 0:
            raise ValueError("val_exact_limit must be positive")
        if any(
            not is_int(value) or value <= 0
            for value in (
                self.d_model,
                self.nhead,
                self.num_layers,
                self.dim_feedforward,
            )
        ):
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("nhead must divide d_model")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not is_int(self.max_seq_len) or self.max_seq_len < 4:
            raise ValueError("max_seq_len must be at least 4")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")
        if not is_int(self.seed) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in 0..2^63-1")

    def checkpoint_dict(self) -> dict[str, Any]:
        """Return a weights-only-loader-safe representation."""

        values = asdict(self)
        values["split_ratios"] = list(self.split_ratios)
        values["human_kinds"] = list(self.human_kinds)
        return values


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    loss: float
    token_accuracy: float
    exact_match: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    direction: Direction
    checkpoint_path: Path
    best_epoch: int
    best_metrics: EpochMetrics


def _batch_tensors(batch: Mapping[str, object], device: torch.device) -> tuple[Tensor, Tensor]:
    source = batch.get("source_ids")
    target = batch.get("target_ids")
    if not isinstance(source, Tensor) or not isinstance(target, Tensor):
        raise TypeError("collated batch is missing source_ids or target_ids tensors")
    return source.to(device), target.to(device)


def _teacher_forced_counts(logits: Tensor, targets: Tensor, pad_id: int) -> tuple[int, int]:
    mask = targets.ne(pad_id)
    correct = logits.argmax(dim=-1).eq(targets) & mask
    return int(correct.sum().item()), int(mask.sum().item())


def _normalized_ids(values: Tensor, *, pad_id: int, eos_id: int) -> tuple[int, ...]:
    result: list[int] = []
    for value in values.tolist():
        token_id = int(value)
        if token_id == pad_id:
            continue
        result.append(token_id)
        if token_id == eos_id:
            break
    return tuple(result)


def run_training_epoch(
    model: Seq2SeqTransformer,
    loader: DataLoader[Mapping[str, object]],
    optimizer: AdamW,
    *,
    device: torch.device,
    pad_id: int,
    grad_clip: float,
) -> EpochMetrics:
    """Run one teacher-forced optimization epoch."""

    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    correct_tokens = 0
    target_tokens = 0

    for batch in loader:
        source, target = _batch_tensors(batch, device)
        if target.size(1) < 2:
            raise ValueError("target sequence must contain BOS and at least one next token")
        target_input = target[:, :-1]
        target_output = target[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(source, target_input)
        loss_sum = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
        non_padding = int(target_output.ne(pad_id).sum().item())
        if non_padding == 0:
            raise ValueError("batch has no non-padding target tokens")
        (loss_sum / non_padding).backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        correct, count = _teacher_forced_counts(logits.detach(), target_output, pad_id)
        total_loss += float(loss_sum.detach().item())
        correct_tokens += correct
        target_tokens += count

    if target_tokens == 0:
        raise ValueError("training loader produced no target tokens")
    return EpochMetrics(
        loss=total_loss / target_tokens,
        token_accuracy=correct_tokens / target_tokens,
    )


@torch.no_grad()
def evaluate_model(
    model: Seq2SeqTransformer,
    loader: DataLoader[Mapping[str, object]],
    *,
    device: torch.device,
    tokenizer: HandmadeTokenizer,
    max_new_tokens: int,
    exact_limit: int | None = None,
) -> EpochMetrics:
    """Measure teacher-forced loss/token accuracy and true greedy exact match."""

    if exact_limit is not None and exact_limit <= 0:
        raise ValueError("exact_limit must be positive when supplied")
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")
    total_loss = 0.0
    correct_tokens = 0
    target_tokens = 0
    exact = 0
    examples_seen = 0

    for batch in loader:
        source, target = _batch_tensors(batch, device)
        target_output = target[:, 1:]
        logits = model(source, target[:, :-1])
        loss_sum = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
        correct, count = _teacher_forced_counts(logits, target_output, tokenizer.pad_id)
        total_loss += float(loss_sum.item())
        correct_tokens += correct
        target_tokens += count

        remaining = source.size(0) if exact_limit is None else exact_limit - examples_seen
        if remaining > 0:
            decode_count = min(source.size(0), remaining)
            predictions = model.greedy_decode(
                source[:decode_count],
                tokenizer.bos_id,
                tokenizer.eos_id,
                tokenizer.pad_id,
                max_new_tokens,
            )
            for predicted, expected in zip(
                predictions, target[:decode_count], strict=True
            ):
                exact += _normalized_ids(
                    predicted, pad_id=tokenizer.pad_id, eos_id=tokenizer.eos_id
                ) == _normalized_ids(
                    expected, pad_id=tokenizer.pad_id, eos_id=tokenizer.eos_id
                )
                examples_seen += 1

    if target_tokens == 0 or examples_seen == 0:
        raise ValueError("validation loader must contain at least one example")
    return EpochMetrics(
        loss=total_loss / target_tokens,
        token_accuracy=correct_tokens / target_tokens,
        exact_match=exact / examples_seen,
    )


def _maximum_encoded_length(dataset: TranslationDataset) -> int:
    maximum = 0
    for index in range(len(dataset)):
        item = dataset[index]
        for key in ("source_ids", "target_ids"):
            values = item[key]
            if not isinstance(values, Tensor):  # pragma: no cover - dataset invariant
                raise TypeError(f"{key} is not a Tensor")
            maximum = max(maximum, values.numel())
    return maximum


def _build_loaders(
    direction: Direction,
    tokenizer: HandmadeTokenizer,
    config: TrainingConfig,
) -> tuple[
    DataLoader[Mapping[str, object]],
    DataLoader[Mapping[str, object]],
    int,
    dict[str, int],
]:
    splits = generate_translation_splits(
        config.num_tableaux,
        seed=config.seed,
        split_seed=config.seed + 1,
        split_ratios=config.split_ratios,
        directions=(direction,),
        human_kinds=config.human_kinds,
        max_rows=config.max_rows,
        max_columns=config.max_columns,
        max_cells=config.max_cells,
    )
    train_dataset = TranslationDataset(splits["train"], tokenizer)
    val_dataset = TranslationDataset(splits["val"], tokenizer)
    test_dataset = TranslationDataset(splits["test"], tokenizer)
    if any(len(dataset) == 0 for dataset in (train_dataset, val_dataset, test_dataset)):
        raise ValueError("train, validation, and test splits must all be non-empty")

    largest = max(
        _maximum_encoded_length(train_dataset),
        _maximum_encoded_length(val_dataset),
        _maximum_encoded_length(test_dataset),
    )
    if largest > config.max_seq_len:
        raise ValueError(
            f"encoded sequence length {largest} exceeds --max-seq-len "
            f"{config.max_seq_len}; increase the limit or reduce shape sizes"
        )
    max_target_length = max(
        int(cast(Tensor, val_dataset[index]["target_ids"]).numel())
        for index in range(len(val_dataset))
    )
    max_new_tokens = max_target_length - 1  # greedy output already begins with BOS

    collate = make_collate_fn(tokenizer.pad_id)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader: DataLoader[Mapping[str, object]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader: DataLoader[Mapping[str, object]] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    counts = {name: len(examples) for name, examples in splits.items()}
    return train_loader, val_loader, max_new_tokens, counts


def train_direction(
    direction: Direction,
    *,
    output_dir: Path,
    device: torch.device,
    config: TrainingConfig,
) -> TrainingResult:
    """Train one translation direction and retain its best validation checkpoint."""

    if direction == "perm_to_yt":
        raise ValueError("perm_to_yt must be trained with train_rsk or yt-rsk-train")
    seed_everything(config.seed + (0 if direction == "yt_to_human" else 10_000))
    tokenizer = HandmadeTokenizer()
    train_loader, val_loader, max_new_tokens, split_counts = _build_loaders(
        direction, tokenizer, config
    )
    model_config = ModelConfig(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
        tie_embeddings=config.tie_embeddings,
    )
    model = Seq2SeqTransformer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_path = output_dir / f"{direction}.pt"
    print(
        f"\n[{direction}] device={device} parameters={parameter_count:,} "
        f"examples(train/val/test)={split_counts['train']}/"
        f"{split_counts['val']}/{split_counts['test']}"
    )

    best_metrics: EpochMetrics | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        train_metrics = run_training_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            pad_id=tokenizer.pad_id,
            grad_clip=config.grad_clip,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            exact_limit=config.val_exact_limit,
        )
        elapsed = time.monotonic() - started
        print(
            f"epoch {epoch:02d} | train loss {train_metrics.loss:.4f} "
            f"tok {train_metrics.token_accuracy:.3f} | val loss {val_metrics.loss:.4f} "
            f"tok {val_metrics.token_accuracy:.3f} exact {val_metrics.exact_match:.3f} "
            f"| {elapsed:.1f}s"
        )

        assert val_metrics.exact_match is not None
        improved = (
            best_metrics is None
            or val_metrics.exact_match > cast(float, best_metrics.exact_match)
            or (
                val_metrics.exact_match == best_metrics.exact_match
                and val_metrics.loss < best_metrics.loss
            )
        )
        if improved:
            best_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                tokenizer=tokenizer,
                direction=direction,
                epoch=epoch,
                metrics={
                    "val_loss": val_metrics.loss,
                    "val_token_accuracy": val_metrics.token_accuracy,
                    "val_exact_match": val_metrics.exact_match,
                },
                training_config=config.checkpoint_dict(),
            )
        else:
            epochs_without_improvement += 1
            if config.patience and epochs_without_improvement >= config.patience:
                print(f"early stopping after {config.patience} epochs without improvement")
                break

    if best_metrics is None:  # pragma: no cover - epochs is validated positive
        raise RuntimeError("training completed without validation metrics")
    print(
        f"best [{direction}] epoch={best_epoch} exact={best_metrics.exact_match:.3f} "
        f"checkpoint={checkpoint_path}"
    )
    return TrainingResult(direction, checkpoint_path, best_epoch, best_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train two small Transformers for YT ↔ human-readable notation."
    )
    parser.add_argument(
        "--direction",
        choices=("both", "yt_to_human", "human_to_yt"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--num-tableaux", type=int, default=4_000)
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--human-kinds",
        nargs="+",
        choices=DEFAULT_HUMAN_KINDS,
        default=DEFAULT_HUMAN_KINDS,
    )
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--max-columns", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5, help="0 disables early stopping")
    parser.add_argument(
        "--val-exact-limit",
        type=int,
        default=256,
        help="fixed validation subset used for autoregressive exact match",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share source, target, and output token weights (default: enabled)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        num_tableaux=args.num_tableaux,
        split_ratios=tuple(args.split_ratios),
        human_kinds=tuple(args.human_kinds),
        max_rows=args.max_rows,
        max_columns=args.max_columns,
        max_cells=args.max_cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        val_exact_limit=args.val_exact_limit,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        tie_embeddings=args.tie_embeddings,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    directions: tuple[Direction, ...]
    if args.direction == "both":
        directions = ("yt_to_human", "human_to_yt")
    else:
        directions = (cast(Direction, args.direction),)
    for direction in directions:
        train_direction(
            direction,
            output_dir=args.output_dir,
            device=device,
            config=config,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "EpochMetrics",
    "TrainingConfig",
    "TrainingResult",
    "evaluate_model",
    "main",
    "run_training_epoch",
    "train_direction",
]
