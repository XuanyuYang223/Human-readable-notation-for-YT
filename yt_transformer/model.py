"""Small encoder-decoder Transformer used by both translation directions.

The module deliberately depends only on PyTorch.  Inputs are batches of token
ids with shape ``[batch, sequence_length]`` and the Transformer itself uses
``batch_first=True`` throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """JSON-serializable configuration for :class:`Seq2SeqTransformer`."""

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_seq_len: int = 256
    pad_id: int = 0
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.src_vocab_size <= 0 or self.tgt_vocab_size <= 0:
            raise ValueError("vocabulary sizes must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must be positive and divide d_model")
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("encoder and decoder layer counts must be positive")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0 <= self.pad_id < min(self.src_vocab_size, self.tgt_vocab_size):
            raise ValueError("pad_id must be present in both vocabularies")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")
        if self.tie_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError("tied embeddings require equal source and target vocabularies")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for JSON or a checkpoint."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Reconstruct a configuration saved with :meth:`to_dict`."""

        return cls(**dict(values))


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position vectors to batch-first embeddings."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()

        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        # For odd d_model, the cosine side has one fewer channel.
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("positional encoding expects [batch, sequence, features]")
        sequence_length = embeddings.size(1)
        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.encoding.size(1)}"
            )
        return embeddings + self.encoding[:, :sequence_length].to(
            dtype=embeddings.dtype
        )


class Seq2SeqTransformer(nn.Module):
    """A compact, batch-first encoder-decoder Transformer.

    ``forward`` consumes the source tokens and the right-shifted target tokens.
    It returns unnormalized target-vocabulary logits.  Target causality and all
    source/target padding masks are constructed internally.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(
            config.src_vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_seq_len
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        # PyTorch currently labels the encoder's automatic NestedTensor path as
        # prototype and emits a warning on every padded eval/inference call.
        # Dense tensors are fast enough for these short sequences and give a
        # quieter, more stable small-project interface across torch versions.
        if hasattr(self.transformer.encoder, "enable_nested_tensor"):
            self.transformer.encoder.enable_nested_tensor = False
        if hasattr(self.transformer.encoder, "use_nested_tensor"):
            self.transformer.encoder.use_nested_tensor = False
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)
        embedding_std = config.d_model ** -0.5
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=embedding_std)
        nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=embedding_std)
        with torch.no_grad():
            self.src_embedding.weight[config.pad_id].zero_()
            self.tgt_embedding.weight[config.pad_id].zero_()
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        if config.tie_embeddings:
            self.tgt_embedding.weight = self.src_embedding.weight
            self.output_projection.weight = self.src_embedding.weight
        self._embedding_scale = math.sqrt(config.d_model)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        """Return a boolean mask whose true entries cannot be attended to."""

        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _validate_token_batch(
        self, tokens: Tensor, name: str, vocab_size: int
    ) -> None:
        if tokens.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, sequence]")
        if tokens.size(0) == 0 or tokens.size(1) == 0:
            raise ValueError(f"{name} must contain a non-empty batch and sequence")
        if tokens.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"{name} length {tokens.size(1)} exceeds configured maximum "
                f"{self.config.max_seq_len}"
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must contain integer token ids")
        # A clear error here is preferable to an implementation-specific Embedding
        # exception.  The reductions are intentionally skipped for empty tensors,
        # which were rejected above.
        if torch.any(tokens < 0).item() or torch.any(tokens >= vocab_size).item():
            raise ValueError(f"{name} contains a token id outside its vocabulary")

    def _embed(self, tokens: Tensor, embedding: nn.Embedding) -> Tensor:
        values = embedding(tokens) * self._embedding_scale
        return self.embedding_dropout(self.position_encoding(values))

    def forward(self, src: Tensor, tgt_input: Tensor) -> Tensor:
        """Return logits with shape ``[batch, target_length, tgt_vocab_size]``."""

        self._validate_token_batch(src, "src", self.config.src_vocab_size)
        self._validate_token_batch(
            tgt_input, "tgt_input", self.config.tgt_vocab_size
        )
        if src.size(0) != tgt_input.size(0):
            raise ValueError("src and tgt_input batch sizes must match")
        if src.device != tgt_input.device:
            raise ValueError("src and tgt_input must be on the same device")

        src_padding_mask = src.eq(self.config.pad_id)
        tgt_padding_mask = tgt_input.eq(self.config.pad_id)
        tgt_mask = self._causal_mask(tgt_input.size(1), tgt_input.device)

        hidden = self.transformer(
            src=self._embed(src, self.src_embedding),
            tgt=self._embed(tgt_input, self.tgt_embedding),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output_projection(hidden)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int,
    ) -> Tensor:
        """Greedily decode a batch and return sequences including the BOS token.

        Decoding stops as soon as every batch item emits EOS, or after
        ``max_new_tokens`` generated tokens.  Once an item is finished, PAD is
        appended while other items continue.  Consequently the returned shape is
        ``[batch, 1 + generated_steps]`` and is never longer than
        ``1 + max_new_tokens``.
        """

        self._validate_token_batch(src, "src", self.config.src_vocab_size)
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if max_new_tokens > self.config.max_seq_len - 1:
            raise ValueError(
                "max_new_tokens plus BOS exceeds the configured maximum sequence "
                "length"
            )
        for name, token_id in (
            ("bos_id", bos_id),
            ("eos_id", eos_id),
            ("pad_id", pad_id),
        ):
            if (
                not isinstance(token_id, int)
                or not 0 <= token_id < self.config.tgt_vocab_size
            ):
                raise ValueError(f"{name} is outside the target vocabulary")
        if pad_id != self.config.pad_id:
            raise ValueError("pad_id must match ModelConfig.pad_id")
        if len({bos_id, eos_id, pad_id}) != 3:
            raise ValueError("bos_id, eos_id, and pad_id must be distinct")

        was_training = self.training
        self.eval()
        try:
            src_padding_mask = src.eq(self.config.pad_id)
            memory = self.transformer.encoder(
                self._embed(src, self.src_embedding),
                src_key_padding_mask=src_padding_mask,
            )
            decoded = torch.full(
                (src.size(0), 1),
                bos_id,
                dtype=torch.long,
                device=src.device,
            )
            finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

            for _ in range(max_new_tokens):
                tgt_padding_mask = decoded.eq(self.config.pad_id)
                hidden = self.transformer.decoder(
                    tgt=self._embed(decoded, self.tgt_embedding),
                    memory=memory,
                    tgt_mask=self._causal_mask(decoded.size(1), decoded.device),
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask,
                )
                next_token = self.output_projection(hidden[:, -1]).argmax(dim=-1)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, pad_id),
                    next_token,
                )
                decoded = torch.cat((decoded, next_token.unsqueeze(1)), dim=1)
                finished |= next_token.eq(eos_id)
                if torch.all(finished).item():
                    break

            return decoded
        finally:
            self.train(was_training)


__all__ = [
    "ModelConfig",
    "Seq2SeqTransformer",
    "SinusoidalPositionalEncoding",
]
