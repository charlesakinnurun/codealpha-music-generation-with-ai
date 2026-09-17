"""Project paths and hyper-parameters.

All paths are derived from the repository root so the project can be
executed from anywhere and moved between machines without hard-coded
absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repository root: <repo>/src/config.py -> <repo>
ROOT = Path(__file__).resolve().parents[1]

# Standard project directories
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
NOTEBOOKS_DIR = ROOT / "notebooks"

PROCESSED_DATASET_FILE = "dataset.npz"
PROCESSED_VOCAB_FILE = "vocab.json"
PROCESSED_META_FILE = "meta.json"


def ensure_dirs() -> None:
    """Create every standard directory if it does not exist yet."""
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessConfig:
    """Options that control MIDI -> dataset conversion."""

    raw_dir: Path = DATA_RAW_DIR
    output_dir: Path = DATA_PROCESSED_DIR
    sequence_length: int = 64       # number of time-steps per training window
    stride: int = 16                # step between the start of consecutive windows
    step_size: float = 0.25         # grid resolution in quarter-note units (1/16th)
    max_tokens_per_file: int = 4000 # cap so unusually long pieces cannot dominate
    min_tokens_per_file: int = 128  # drop files with fewer steps than this
    max_files: int = 100000         # safety limit on number of files processed
    seed: int = 42


@dataclass
class ModelConfig:
    """Architecture of the LSTM music model."""

    vocab_size: int = 0              # filled in from the dataset
    embedding_dim: int = 128         # size of the token embedding layer
    hidden_dim: int = 256            # number of LSTM hidden units
    num_layers: int = 2              # number of stacked LSTM layers
    dropout: float = 0.3             # dropout applied between LSTM layers / at output

    def to_dict(self) -> dict:
        d = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        return d

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_split: float = 0.1           # fraction of windows reserved for validation
    log_interval: int = 20           # print loss every N batches
    seed: int = 42
    device: str = "auto"             # "auto" | "cuda" | "cpu"
    models_dir: Path = MODELS_DIR
    checkpoint_name: str = "latest.pt"
    resume: Path | None = None       # if set, continue training from checkpoint

    def to_dict(self) -> dict:
        d = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "val_split": self.val_split,
            "log_interval": self.log_interval,
            "seed": self.seed,
            "device": self.device,
        }
        return d

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        return TrainConfig(**{k: v for k, v in d.items() if k in TrainConfig.__dataclass_fields__})


@dataclass
class GenerateConfig:
    """Options for generating a new composition from a trained model."""

    checkpoint: Path = MODELS_DIR / "latest.pt"
    length: int = 200                # number of time-steps to generate
    temperature: float = 1.0         # higher => more random, lower => more conservative
    top_k: int | None = None         # restrict sampling to the k most likely tokens
    seed_token: str | None = None    # optional explicit token to start from, e.g. "P:C4"
    output_dir: Path = OUTPUTS_DIR
    output_name: str | None = None   # None => auto timestamped name
    tempo_bpm: int = 100

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "seed_token": self.seed_token,
            "tempo_bpm": self.tempo_bpm,
        }