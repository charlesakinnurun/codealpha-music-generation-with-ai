"""Optional MIDI -> WAV conversion.

MIDI files contain note *instructions*, not recorded sound. To hear them you
need a synthesizer. This module converts a generated .mid into a .wav using a
command-line synth if one is installed:

* ``timidity``   (often ships with a built-in instrument set, no soundfont needed)
* ``fluidsynth`` (requires a SoundFont, path may be passed in or set via the
                 ``SOUNDFONT`` environment variable)
* ``mscore`` / ``musescore3`` / ``musescore4`` (MuseScore headless rendering)

If none of these are available the script prints clear instructions instead of
failing silently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SYNTH_ORDER = (
    "timidity",
    "fluidsynth",
    "mscore",
    "musescore3",
    "musescore4",
)


def _which(name: str) -> Path | None:
    return Path(shutil.which(name)) if shutil.which(name) else None


def find_soundfont(explicit: str | None = None) -> Path | None:
    """Locate a SoundFont (.sf2/.sf3) using an explicit path or ``SOUNDFONT``."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("SOUNDFONT")
    if env:
        candidates.append(Path(env))
    candidates.extend(sorted(Path(".").glob("*.[sS][fF][23]")))
    candidates.extend(sorted(Path("data").glob("*.[sS][fF][23]")))
    for c in candidates:
        if c.is_file():
            return c
    return None


def midi_to_wav(
    midi_path: Path, output_path: Path, soundfont: str | None = None
) -> Path:
    """Convert a MIDI file to WAV with the first available synthesizer."""
    midi_path = Path(midi_path)
    output_path = Path(output_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for name in SYNTH_ORDER:
        exe = _which(name)
        if exe is None:
            continue

        if name == "timidity":
            cmd = [str(exe), "-Ow", "-o", str(output_path), str(midi_path)]
        elif name == "fluidsynth":
            sf = find_soundfont(soundfont)
            if sf is None:
                print(
                    "[audio] fluidsynth found but no SoundFont. "
                    "Set SOUNDFONT=/path/to/your.sf2 (or pass --soundfont)."
                )
                continue
            cmd = [str(exe), str(sf), str(midi_path), "-F", str(output_path)]
        else:  # MuseScore family: headless MIDI -> audio
            cmd = [
                str(exe),
                "-o",
                str(output_path),
                str(midi_path),
            ]

        print(f"[audio] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[audio] WAV saved to {output_path.resolve()}")
            return output_path.resolve()
        print(
            "[audio] synthesizer exited with an error: "
            + (result.stderr or result.stdout or "no output")
        )

    raise RuntimeError(
        "No MIDI synthesizer is installed. Install one of: "
        "timidity, fluidsynth (with a SoundFont), or MuseScore, "
        "then re-run with --to-wav."
    )