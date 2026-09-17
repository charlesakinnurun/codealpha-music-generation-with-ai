"""Generate a new piece of music from a trained checkpoint.

Usage:
    python scripts/generate.py [--checkpoint models/best.pt] [--length 200]
                               [--temperature 1.0] [--top-k 10]
                               [--seed-token P:C4] [--out-dir outputs]
                               [--output-name my_melody.mid] [--to-wav]

The output is written to ``outputs/generated_<timestamp>.mid`` by default.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

from src import audio, generate as gen
from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    GenerateConfig,
)
from src.preprocess import load_processed_dataset
from src.train import resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate music with a trained LSTM.")
    parser.add_argument(
        "--checkpoint", default=str(MODELS_DIR / "latest.pt"),
        help="path to the trained model checkpoint (default: models/latest.pt)",
    )
    parser.add_argument(
        "--length", type=int, default=200,
        help="number of time-steps to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sampling temperature; higher = more varied, lower = more repetitive (default: 1.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="restrict sampling to the top-k most likely tokens (optional)",
    )
    parser.add_argument(
        "--seed-token", default=None,
        help="explicit token to start from, e.g. 'P:C4'. Omitting it uses a "
             "random window from the training data.",
    )
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR))
    parser.add_argument("--output-name", default=None, help="output .mid filename")
    parser.add_argument("--tempo", type=int, default=100, help="tempo in BPM")
    parser.add_argument("--processed-dir", default=str(DATA_PROCESSED_DIR))
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--to-wav", action="store_true",
        help="also convert the generated MIDI to WAV (needs a synthesizer)",
    )
    parser.add_argument(
        "--soundfont", default=None,
        help="path to a .sf2/.sf3 SoundFont for fluidsynth",
    )
    args = parser.parse_args(argv)

    device = resolve_device(args.device)

    # Try to load the processed dataset purely to seed generation with a real
    # musical window. It is optional - if unavailable we rely on --seed-token.
    sequences = None
    try:
        seq, _t, _v, _i, _m = load_processed_dataset(args.processed_dir)
        sequences = seq
    except FileNotFoundError:
        if args.seed_token is None:
            print(
                "No processed dataset found and no --seed-token given; "
                "generation cannot build a seed context."
            )
            return 1

    cfg = GenerateConfig(
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        seed_token=args.seed_token,
        output_dir=args.out_dir,
        output_name=args.output_name,
        tempo_bpm=args.tempo,
    )

    try:
        midi_path = gen.generate(
            cfg,
            checkpoint_path=args.checkpoint,
            sequences=sequences,
            device=device,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    if args.to_wav:
        wav_path = midi_path.with_suffix(".wav")
        try:
            audio.midi_to_wav(midi_path, wav_path, soundfont=args.soundfont)
        except RuntimeError as exc:
            print(f"[audio] {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
