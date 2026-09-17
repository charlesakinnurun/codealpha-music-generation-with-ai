"""MIDI preprocessing with music21.

Pipeline overview
-----------------
1. Locate every ``*.mid`` (or ``*.midi``) file under ``data/raw``.
2. Parse each file with music21 into a timed grid of "steps" (default 1/16th of a
   quarter note). At every step we record the set of pitches that are sounding
   (notes that start that step or sustain into it).
3. Encode every step as a single token:
   ``R``                       -> silence (nothing sounding)
   ``P:C4,E4,G4``              -> chord sounding at that step
4. Turn the token stream into fixed-length (X, y) windows where ``y`` is the
   token that immediately follows window ``X`` (next-step prediction).
5. Store the integer arrays plus a token -> id vocabulary so the expensive MIDI
   parsing only has to run once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import music21
import numpy as np

from .config import (
    DATA_PROCESSED_DIR,
    PROCESSED_DATASET_FILE,
    PROCESSED_META_FILE,
    PROCESSED_VOCAB_FILE,
    PreprocessConfig,
)

REST_TOKEN = "R"
PITCH_PREFIX = "P:"

MIDI_EXTENSIONS = {".mid", ".midi"}


def find_midi_files(raw_dir: Path) -> list[Path]:
    """Recursively collect all MIDI files under ``raw_dir``."""
    if not raw_dir.exists():
        return []
    files = [
        p
        for p in raw_dir.rglob("*")
        if p.suffix.lower() in MIDI_EXTENSIONS and p.is_file()
    ]
    return sorted(files)


def load_score(path: Path) -> Optional[music21.stream.Score]:
    """Parse a MIDI file into a music21 score, returning None on failure.

    Malformed, corrupt or unsupported files are skipped silently so one bad
    file cannot abort the whole preprocessing run.
    """
    try:
        score = music21.converter.parse(str(path))
        return score
    except Exception as e:  # music21 raises many exception types on bad files
        print(f"  [skip] could not parse {path}: {e}")
        return None


def _note_pitches(note_or_chord) -> set[int]:
    """Return the MIDI pitch numbers for a Note or Chord."""
    if note_or_chord.isNote:
        return {note_or_chord.pitch.midi}
    if note_or_chord.isChord:
        return {p.midi for p in note_or_chord.pitches}
    return set()


def score_to_step_tokens(
    score: music21.stream.Score,
    step_size: float = 0.25,
    max_tokens: int = 4000,
) -> Optional[list[str]]:
    """Convert a score to a list of step tokens.

    Every step token describes everything that is audible on a grid of
    ``step_size`` quarter-note units, which preserves both rhythm and harmony.
    Returns None if the score contains no notes at all.
    """
    parts = list(score.parts) if len(score.parts) > 0 else [score]

    events: list[tuple[float, float, set[int]]] = []
    for part in parts:
        for element in part.flatten().notes:
            duration = getattr(element, "duration", None)
            if duration is None or duration.quarterLength <= 0:
                continue
            pitches = _note_pitches(element)
            if not pitches:
                continue
            events.append(
                (float(element.offset), float(element.offset + duration.quarterLength), pitches)
            )

    if not events:
        return None

    end_step = int(max(end for _, end, _ in events) / step_size) + 1
    end_step = min(end_step, max_tokens + 1)

    grid: list[set[int]] = [set() for _ in range(end_step + 1)]
    for onset, end, pitches in events:
        start = max(0, int(round(onset / step_size)))
        stop = max(start + 1, int(round(end / step_size)))
        for step in range(start, min(stop, len(grid))):
            grid[step] |= pitches

    tokens = []
    for step_pitches in grid:
        if not step_pitches:
            tokens.append(REST_TOKEN)
            continue
        names = [music21.pitch.Pitch(m).nameWithOctave for m in sorted(step_pitches)]
        tokens.append(PITCH_PREFIX + ",".join(names))

    return tokens


def build_vocab(token_streams: list[list[str]], start_index: int = 0) -> dict[str, int]:
    """Create a deterministic token -> integer vocabulary from a list of streams."""
    vocab: dict[str, int] = {}
    for stream in token_streams:
        for token in stream:
            if token not in vocab:
                vocab[token] = start_index + len(vocab)
    return vocab


def make_windows(
    tokens: list[int], sequence_length: int, stride: int
) -> Iterator[tuple[list[int], int]]:
    """Yield (input_window, target) pairs over one token stream.

    The target is the token that immediately follows the window, i.e. the next
    musical step the model is asked to predict.
    """
    if len(tokens) <= sequence_length:
        return
    for start in range(0, len(tokens) - sequence_length, stride):
        window = tokens[start : start + sequence_length]
        target = tokens[start + sequence_length]
        yield window, target


def build_dataset(cfg: PreprocessConfig) -> dict:
    """Run the full preprocessing pipeline and save the result to disk.

    Returns a dict with the token streams, vocabulary and the number of
    training windows created.
    """
    raw_dir = Path(cfg.raw_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = find_midi_files(raw_dir)
    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found under {raw_dir}. "
            "Run `python scripts/download_dataset.py` or add your own .mid files."
        )

    print(f"Found {len(midi_files)} MIDI file(s) under {raw_dir}")

    token_streams: list[list[str]] = []
    used_files: list[str] = []
    skipped = 0

    for i, path in enumerate(midi_files[: cfg.max_files], start=1):
        print(f"[{i}/{len(midi_files)}] parsing {path.name}")
        score = load_score(path)
        if score is None:
            skipped += 1
            continue
        tokens = score_to_step_tokens(
            score, step_size=cfg.step_size, max_tokens=cfg.max_tokens_per_file
        )
        if tokens is None or len(tokens) < cfg.min_tokens_per_file:
            skipped += 1
            continue
        tokens = tokens[: cfg.max_tokens_per_file]
        token_streams.append(tokens)
        used_files.append(str(path.resolve()))

    if not token_streams:
        raise RuntimeError(
            "No usable tokens were extracted (all files too short or unparseable)."
        )

    vocab = build_vocab(token_streams)
    idx_to_token: list[str] = [""] * len(vocab)
    for token, idx in vocab.items():
        idx_to_token[idx] = token

    encoded_streams = [[vocab[t] for t in stream] for stream in token_streams]

    windows: list[list[int]] = []
    targets: list[int] = []
    for stream in encoded_streams:
        for window, target in make_windows(stream, cfg.sequence_length, cfg.stride):
            windows.append(window)
            targets.append(target)

    if not windows:
        raise RuntimeError(
            "No training windows could be created. Lower --min-tokens-per-file "
            "or reduce --sequence-length."
        )

    meta = {
        "sequence_length": cfg.sequence_length,
        "stride": cfg.stride,
        "step_size": cfg.step_size,
        "num_tokens_vocab": len(vocab),
        "num_windows": len(windows),
        "num_files_used": len(used_files),
        "num_files_skipped": skipped,
        "files_used": used_files,
    }

    dataset_path = output_dir / PROCESSED_DATASET_FILE
    vocab_path = output_dir / PROCESSED_VOCAB_FILE
    meta_path = output_dir / PROCESSED_META_FILE

    np.savez_compressed(str(dataset_path), sequences=windows, targets=targets)
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2, sort_keys=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    exemplar = ", ".join(idx_to_token[:8]) + ", ..." if len(idx_to_token) > 8 else ", ".join(idx_to_token)
    print(f"Vocabulary size: {len(vocab)} tokens (e.g. {exemplar})")
    print(f"Training windows: {len(windows)}")
    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved vocabulary to: {vocab_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Files skipped: {skipped}")

    return {
        "token_streams": token_streams,
        "vocab": vocab,
        "idx_to_token": idx_to_token,
        "num_windows": len(windows),
        "num_files_used": len(used_files),
    }


def load_processed_dataset(
    processed_dir: Path = DATA_PROCESSED_DIR,
) -> tuple:
    """Load the previously saved dataset.

    Returns ``(sequences, targets, vocab, idx_to_token, meta)``.
    """
    processed_dir = Path(processed_dir)
    dataset_path = processed_dir / PROCESSED_DATASET_FILE
    vocab_path = processed_dir / PROCESSED_VOCAB_FILE
    meta_path = processed_dir / PROCESSED_META_FILE

    for p in (dataset_path.with_suffix(".npz"), vocab_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Processed dataset missing ({p}). Run `python scripts/preprocess.py` first."
            )

    data = np.load(dataset_path.with_suffix(".npz"))
    sequences = data["sequences"]
    targets = data["targets"]

    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    idx_to_token = [""] * len(vocab)
    for token, idx in vocab.items():
        idx_to_token[int(idx)] = token

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    return sequences, targets, vocab, idx_to_token, meta


def ensure_dataset(
    cfg: PreprocessConfig, force_rebuild: bool = False
) -> tuple:
    """Return the processed dataset, rebuilding it if necessary."""
    dataset_path = Path(cfg.output_dir) / PROCESSED_DATASET_FILE
    if force_rebuild or not dataset_path.exists():
        build_dataset(cfg)
    return load_processed_dataset(Path(cfg.output_dir))