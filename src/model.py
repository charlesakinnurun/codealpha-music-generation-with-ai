"""Configurable LSTM-based next-step music model (PyTorch).

The model consumes a sequence of token ids (one per musical time-step) and
predicts a distribution over the next token id.

Architecture (all sizes configurable):

    Embedding(vocab -> embedding_dim)
        -> Dropout
        -> LSTM(embedding_dim -> hidden_dim, num_layers, dropout)
        -> Dropout
        -> Linear(hidden_dim -> vocab)           (next-step logits)
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class MusicLSTM(nn.Module):
    """LSTM next-note predictor.

    Parameters mirror :class:`ModelConfig` so checkpoints are self-describing.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Dropout between LSTM layers is only meaningful with num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Returns ``(logits, hidden_state)``.

        ``x``: ``(batch, seq_len)`` tensor of token ids.
        ``logits``: ``(batch, seq_len, vocab_size)``.
        """
        embedded = self.embedding(x)              # (B, S, E)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)  # output (B, S, H)
        output = self.dropout(output)
        logits = self.fc(output)                  # (B, S, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a zero-initialised hidden state for the LSTM."""
        shape = (self.num_layers, batch_size, self.hidden_dim)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "MusicLSTM":
        return cls(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

    def config(self) -> ModelConfig:
        """Export the architecture settings used to build this model."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
        )