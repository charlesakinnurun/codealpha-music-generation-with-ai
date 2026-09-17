"""Generate new music from a trained checkpoint.

The model is queried one step at a time (autoregressive). At every step the
next token is drawn from the model's probability distribution scaled by a
*temperature*, so generations are musical rather than fully deterministic.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import GenerateConfig
from .model import MusicLSTM
from .preprocess import REST_TOKEN
from .train import set_seed


# ---------------------------------------------------------------------------
# Model / checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint_for_generation(
    checkpoint_path: Path, device: str
) -> dict:
    """Load a checkpoint and return model + vocabulary + settings."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    model = MusicLSTM(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx_to_token = ckpt.get("idx_to_token")
    vocab = ckpt.get("vocab")
    if idx_to_token is None or vocab is None:
        raise ValueError(
            "Checkpoint does not carry vocabulary info. "
            "Please retrain with this version of the code."
        )
    step_size = ckpt.get("step_size", 0.25)
    sequence_length = ckpt.get("sequence_length")

    return {
        "model": model,
        "vocab": vocab,
        "idx_to_token": idx_to_token,
        "step_size": step_size,
        "sequence_length": sequence_length,
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_next_token(
    model: MusicLSTM,
    context: torch.Tensor,
    temperature: float,
    top_k: int | None,
    device: str,
) -> int:
    """Predict and sample one token given the current context tensor (1, S)."""
    logits, _ = model(context)             # (1, S, V)
    logits = logits[:, -1, :]              # only the last position
    scaled = logits / max(temperature, 1e-5)

    probs = torch.softmax(scaled, dim=-1)
    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.shape[-1])
        top_probs, top_idx = probs.topk(top_k)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, top_idx, top_probs)
        probs = filtered / filtered.sum(dim=-1, keepdim=True)

    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_sequence(
    model: MusicLSTM,
    seed_ids: list[int],
    length: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> list[int]:
    """Autoregressively generate ``length`` tokens continuing from ``seed_ids``."""
    model.eval()
    context = torch.as_tensor([seed_ids], dtype=torch.long, device=device)
    generated: list[int] = []

    for _ in range(length):
        next_id = sample_next_token(model, context, temperature, top_k, device)
        generated.append(next_id)
        # Slide the context one step (drop the oldest token).
        context = torch.cat(
            [context, torch.as_tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )[..., 1:]

    return generated


def choose_seed(
    vocab: dict,
    idx_to_token: list[str],
    seed_token: str | None,
    sequences: np.ndarray | None,
    sequence_length: int | None,
    rng: np.random.Generator,
) -> list[int]:
    """Build a starting context window of length ``sequence_length``.

    The seed token (if given) is placed at the *end* of the window so the first
    generated note follows it.
    """
    seq_len = sequences.shape[1] if sequences is not None else sequence_length
    if seed_token is not None:
        seed_token = seed_token.strip()
        if seed_token not in vocab:
            raise ValueError(
                f"Unknown seed token {seed_token!r}. Choose one of: "
                + ", ".join(list(vocab)[:12])
                + ("..." if len(vocab) > 12 else "")
            )
        seed_id = int(vocab[seed_token])

    if sequences is not None and len(sequences) > 0:
        window = [int(i) for i in sequences[int(rng.integers(len(sequences)))]]
        if seed_token is not None:
            window[-1] = seed_id
        return window

    if seed_token is not None:
        if seq_len is None:
            raise ValueError("Could not determine sequence length from the checkpoint.")
        return [seed_id] * seq_len

    raise ValueError(
        "No seed available. Either provide --seed-token or make sure the "
        "preprocessed dataset is present so a random training window can be used."
    )


# ---------------------------------------------------------------------------
# MIDI writing
# ---------------------------------------------------------------------------

def tokens_to_midi(
    generated_ids: list[int],
    idx_to_token: list[str],
    step_size: float,
    output_path: Path,
    tempo_bpm: int = 100,
) -> Path:
    """Turn generated token ids back into a MIDI file using music21.

    MIDI is a symbolic format: it stores *instructions* about which notes to
    play, with what timing and instrument - not recorded audio. Play the file
    with any MIDI player or synth (see the README for audio conversion).
    """
    import music21

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score = music21.stream.Score()
    part = music21.stream.Part()
    instrument = music21.instrument.Piano()
    part.append(instrument)

    tempo = music21.tempo.MetronomeMark(number=tempo_bpm)
    score.append(music21.meter.TimeSignature("4/4"))
    part.append(tempo)

    current_offset = 0.0
    for token_id in generated_ids:
        token = idx_to_token[token_id]
        if token == REST_TOKEN:
            current_offset += step_size
            continue
        if not token.startswith("P:"):
            # Defensive: unknown token simply consumes a step.
            current_offset += step_size
            continue

        names = [n for n in token[2:].split(",") if n]
        if len(names) == 1:
            element = music21.note.Note(names[0])
        elif len(names) > 1:
            element = music21.chord.Chord(names)
        else:
            current_offset += step_size
            continue

        element.offset = current_offset
        element.duration.quarterLength = step_size
        part.append(element)
        current_offset += step_size

    # Merge the notes back into a score (music21 orders by offset on write).
    part.makeMeasures(inPlace=True)
    score.append(part)
    score.write("midi", fp=str(output_path))
    return output_path.resolve()


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------

def generate(
    cfg: GenerateConfig,
    checkpoint_path: Path,
    sequences: np.ndarray | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """Run the whole generation pipeline, returning the path of the .mid file."""
    set_seed(seed)
    store = load_checkpoint_for_generation(checkpoint_path, device)
    model = store["model"]
    vocab = store["vocab"]
    idx_to_token = store["idx_to_token"]
    step_size = store["step_size"]

    rng = np.random.default_rng(seed)
    seed_ids = choose_seed(
        vocab, idx_to_token, cfg.seed_token, sequences, store["sequence_length"], rng
    )

    print(f"Model      : {checkpoint_path}")
    print(f"Vocabulary : {len(vocab)} tokens")
    print(f"Seed token : {cfg.seed_token or '(random training window)'}")
    print(f"Generating {cfg.length} steps at temperature {cfg.temperature}...")

    generated = generate_sequence(
        model, seed_ids, cfg.length, cfg.temperature, cfg.top_k, device
    )

    if cfg.output_name is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg.output_name = f"generated_{stamp}.mid"
    output_path = Path(cfg.output_dir) / cfg.output_name

    tokens_to_midi(
        generated, idx_to_token, step_size, output_path, tempo_bpm=cfg.tempo_bpm
    )
    print(f"Generated MIDI saved to: {output_path}")

    preview = " ".join(idx_to_token[i] for i in generated[:12])
    print(f"First generated tokens : {preview} ...")
    print("Note: this is a MIDI file (musical instructions), not audio.")
    return output_path