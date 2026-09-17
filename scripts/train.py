"""Train the LSTM music model.

Usage:
    python scripts/train.py [--epochs 30] [--batch-size 64] [--learning-rate 1e-3]
                            [--sequence-length 64] [--embedding-dim 128]
                            [--hidden-dim 256] [--num-layers 2] [--dropout 0.3]
                            [--val-split 0.1] [--device auto] [--seed 42]
                            [--resume models/latest.pt]

The model is checkpointed to ``models/latest.pt`` after every epoch and the
best-scoring epoch (by validation loss) is kept at ``models/best.pt``.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import time

from src import train
from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    ModelConfig,
    PreprocessConfig,
    TrainConfig,
)
from src.preprocess import ensure_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the LSTM music model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--processed-dir", default=str(DATA_PROCESSED_DIR))
    parser.add_argument(
        "--force-preprocess", action="store_true",
        help="rebuild the processed dataset even if it already exists",
    )
    parser.add_argument(
        "--resume", default=None,
        help="path to a checkpoint (.pt) to continue training from",
    )
    args = parser.parse_args(argv)

    # Load (or build) the dataset.
    pre_cfg = PreprocessConfig(
        output_dir=args.processed_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    sequences, targets, vocab, idx_to_token, meta = ensure_dataset(
        pre_cfg, force_rebuild=args.force_preprocess
    )
    meta_seq_len = int(meta["sequence_length"])

    # The training window length must match the processed sequence length.
    if meta_seq_len != args.sequence_length:
        print(
            f"Dataset was built with sequence_length={meta_seq_len}, but you "
            f"requested --sequence-length {args.sequence_length}."
        )
        print("Re-run `python scripts/preprocess.py --sequence-length "
              f"{args.sequence_length}` (or use --force-preprocess).")
        return 1

    model_cfg = ModelConfig(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
        models_dir=args.models_dir,
        resume=args.resume,
    )

    print(f"Vocabulary: {len(vocab)} tokens | Sequences: {len(sequences)} "
          f"| Window length: {args.sequence_length}")

    start = time.time()
    history = train.train(
        train_cfg,
        model_cfg,
        sequences,
        targets,
        vocab,
        idx_to_token,
        step_size=float(meta["step_size"]),
    )
    print(f"Total training time: {time.time() - start:.1f}s")

    losses = history["val_loss"]
    if losses:
        best_epoch = history["epoch"][int(min(range(len(losses)), key=losses.__getitem__))]
        print(f"Best validation loss: {min(losses):.4f} at epoch {best_epoch}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
