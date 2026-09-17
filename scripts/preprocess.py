"""Preprocess MIDI files into a training-ready dataset.

Usage:
    python scripts/preprocess.py [--raw-dir data/raw] [--out-dir data/processed]
                                 [--sequence-length 64] [--stride 16]
                                 [--step-size 0.25] [--max-tokens-per-file 4000]
                                 [--min-tokens-per-file 128]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import time

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, PreprocessConfig
from src.preprocess import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess MIDI into windows.")
    parser.add_argument("--raw-dir", type=str, default=str(DATA_RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DATA_PROCESSED_DIR))
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--max-tokens-per-file", type=int, default=4000)
    parser.add_argument("--min-tokens-per-file", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = PreprocessConfig(
        raw_dir=args.raw_dir,
        output_dir=args.out_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
        step_size=args.step_size,
        max_tokens_per_file=args.max_tokens_per_file,
        min_tokens_per_file=args.min_tokens_per_file,
        max_files=args.max_files,
        seed=args.seed,
    )

    start = time.time()
    build_dataset(cfg)
    print(f"Preprocessing finished in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
