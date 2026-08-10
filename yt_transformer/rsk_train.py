"""Train an independent permutation-to-YT Transformer for RSK insertion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .model import ModelConfig, Seq2SeqTransformer
from .rsk_data import (
    RSKDataset,
    RSKExample,
    generate_rsk_splits,
    make_rsk_collate_fn,
)
from .runtime import resolve_device, seed_everything
from .tokenizer import HandmadeTokenizer, RSK_VOCAB
from .train import EpochMetrics, TrainingResult, evaluate_model, run_training_epoch


@dataclass(frozen=True, slots=True)
class RSKTrainingConfig:
    """Reproducibility-relevant RSK data, model, and optimizer settings."""

    num_permutations: int = 40_000
    min_length: int = 1
    max_length: int = 20
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int | None = None
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 5
    val_exact_limit: int = 256
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1_024
    dropout: float = 0.1
    max_seq_len: int = 128
    tie_embeddings: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        def is_int(value: object) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        if not is_int(self.num_permutations) or self.num_permutations < 3:
            raise ValueError("num_permutations must be at least 3")
        if (
            not is_int(self.min_length)
            or not is_int(self.max_length)
            or not 1 <= self.min_length <= self.max_length <= 50
        ):
            raise ValueError(
                "permutation lengths must satisfy "
                "1 <= min_length <= max_length <= 50"
            )

        try:
            ratios = tuple(self.split_ratios)
        except TypeError as exc:
            raise ValueError("split_ratios must contain three positive finite values") from exc
        if len(ratios) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in ratios
        ):
            raise ValueError("split_ratios must contain three positive finite values")
        object.__setattr__(self, "split_ratios", cast(tuple[float, float, float], ratios))

        if not is_int(self.seed) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in 0..2^63-1")
        resolved_split_seed = self.seed + 1 if self.split_seed is None else self.split_seed
        if not is_int(resolved_split_seed) or not 0 <= resolved_split_seed < 2**63:
            raise ValueError("split_seed must be an integer in 0..2^63-1")
        object.__setattr__(self, "split_seed", resolved_split_seed)

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
        # A descending length-n permutation yields n one-cell rows.  Its raw YT
        # target is the longest possible encoding: 4*n+3 tokens with BOS/EOS.
        required_sequence_length = 4 * self.max_length + 3
        if self.max_seq_len < required_sequence_length:
            raise ValueError(
                f"max_seq_len must be at least {required_sequence_length} for "
                f"max_length={self.max_length}"
            )
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")

    def checkpoint_dict(self) -> dict[str, Any]:
        """Return metadata sufficient to reproduce the exact P-group split."""

        values = asdict(self)
        values["split_ratios"] = list(self.split_ratios)
        return values


def _maximum_encoded_length(dataset: RSKDataset) -> int:
    maximum = 0
    for index in range(len(dataset)):
        item = dataset[index]
        for key in ("source_ids", "target_ids"):
            values = item[key]
            if not isinstance(values, Tensor):  # pragma: no cover - dataset invariant
                raise TypeError(f"{key} is not a Tensor")
            maximum = max(maximum, values.numel())
    return maximum


def _interleave_lengths(examples: Sequence[RSKExample]) -> tuple[RSKExample, ...]:
    """Order examples round-robin by length for a balanced exact-match prefix."""

    buckets: dict[int, list[RSKExample]] = {}
    for example in examples:
        buckets.setdefault(example.length, []).append(example)
    interleaved: list[RSKExample] = []
    offset = 0
    while len(interleaved) < len(examples):
        for length in sorted(buckets):
            bucket = buckets[length]
            if offset < len(bucket):
                interleaved.append(bucket[offset])
        offset += 1
    return tuple(interleaved)


def build_rsk_loaders(
    config: RSKTrainingConfig,
    *,
    tokenizer: HandmadeTokenizer | None = None,
) -> tuple[
    DataLoader[Mapping[str, object]],
    DataLoader[Mapping[str, object]],
    int,
    dict[str, int],
]:
    """Build deterministic P-group train/validation splits and data loaders."""

    selected_tokenizer = (
        HandmadeTokenizer(vocab=RSK_VOCAB) if tokenizer is None else tokenizer
    )
    if selected_tokenizer.vocab != RSK_VOCAB:
        raise ValueError("RSK training requires the 72-token RSK vocabulary")
    assert config.split_seed is not None  # normalized by RSKTrainingConfig
    splits = generate_rsk_splits(
        config.num_permutations,
        seed=config.seed,
        split_seed=config.split_seed,
        min_length=config.min_length,
        max_length=config.max_length,
        split_ratios=config.split_ratios,
    )
    train_dataset = RSKDataset(splits["train"], selected_tokenizer)
    # ``evaluate_model`` autoregressively decodes a fixed prefix.  Interleaving
    # lengths keeps that prefix representative instead of making model
    # selection depend on the incidental tuple sort order of the split.
    val_dataset = RSKDataset(
        _interleave_lengths(splits["val"]), selected_tokenizer
    )
    test_dataset = RSKDataset(splits["test"], selected_tokenizer)
    if any(len(dataset) == 0 for dataset in (train_dataset, val_dataset, test_dataset)):
        raise ValueError("P-group train, validation, and test splits must all be non-empty")

    largest = max(
        _maximum_encoded_length(train_dataset),
        _maximum_encoded_length(val_dataset),
        _maximum_encoded_length(test_dataset),
    )
    if largest > config.max_seq_len:
        raise ValueError(
            f"encoded sequence length {largest} exceeds --max-seq-len "
            f"{config.max_seq_len}; increase the limit or reduce permutation lengths"
        )
    max_target_length = max(
        int(cast(Tensor, val_dataset[index]["target_ids"]).numel())
        for index in range(len(val_dataset))
    )
    max_new_tokens = max_target_length - 1

    collate = make_rsk_collate_fn(selected_tokenizer.pad_id)
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


def train_rsk(
    *,
    output_dir: Path,
    device: torch.device,
    config: RSKTrainingConfig,
) -> TrainingResult:
    """Train permutation-to-YT and retain the best validation checkpoint."""

    seed_everything(config.seed)
    tokenizer = HandmadeTokenizer(vocab=RSK_VOCAB)
    train_loader, val_loader, max_new_tokens, split_counts = build_rsk_loaders(
        config, tokenizer=tokenizer
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
    checkpoint_path = output_dir / "perm_to_yt.pt"
    print(
        f"\n[perm_to_yt] device={device} parameters={parameter_count:,} "
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
                direction="perm_to_yt",
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
        f"best [perm_to_yt] epoch={best_epoch} exact={best_metrics.exact_match:.3f} "
        f"checkpoint={checkpoint_path}"
    )
    return TrainingResult("perm_to_yt", checkpoint_path, best_epoch, best_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Transformer to compute the RSK insertion tableau P."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/rsk"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--num-permutations", type=int, default=40_000)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1_024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128)
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
    config = RSKTrainingConfig(
        num_permutations=args.num_permutations,
        min_length=args.min_length,
        max_length=args.max_length,
        split_ratios=tuple(args.split_ratios),
        split_seed=args.split_seed,
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
    train_rsk(
        output_dir=args.output_dir,
        device=resolve_device(args.device),
        config=config,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RSKTrainingConfig",
    "build_parser",
    "build_rsk_loaders",
    "main",
    "train_rsk",
]
