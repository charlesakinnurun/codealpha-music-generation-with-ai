# Music Generation with AI (LSTM / PyTorch)

A complete, runnable deep-learning project that learns patterns from
public-domain classical MIDI files and **generates new musical sequences**.
Notes and chords are extracted with `music21`, encoded into a compact
step-based token vocabulary, learnt by a configurable PyTorch LSTM, and finally
decoded back into playable `.mid` files.

> **Important**: MIDI files store musical *instructions* (which notes, when,
> how loud, with what instrument) - **not recorded audio**. The output of this
> project is a `.mid` file that must be played through a synthesizer to be
> heard (see [Rendering to audio](#optional-midi--audio-rendering)).

---

## Table of contents

1. [Project overview](#project-overview)
2. [How the system works](#how-the-system-works)
3. [Dataset setup](#dataset-setup)
4. [Installation](#installation)
5. [Preprocessing](#preprocessing)
6. [Training](#training)
7. [Music generation](#music-generation)
8. [Example commands](#example-commands)
9. [Project structure](#project-structure)
10. [Technologies used](#technologies-used)
11. [Limitations and possible improvements](#limitations-and-possible-improvements)
12. [License and data rights](#license-and-data-rights)

---

## Project overview

The system has five stages:

| Stage | What happens | Entry point |
|-------|--------------|-------------|
| 1. Dataset | Download public-domain classical MIDI files | `scripts/download_dataset.py` |
| 2. Preprocessing | Convert MIDI -> token sequences -> tensors | `scripts/preprocess.py` |
| 3. Training | Train an LSTM to predict the next musical step | `scripts/train.py` |
| 4. Generation | Sample new sequences from the trained model | `scripts/generate.py` |
| 5. Playback | (Optional) render the `.mid` to `.wav` | `scripts/generate.py --to-wav` |

All stages run from a clean environment with just `pip install -r requirements.txt`.

---

## How the system works

### Step tokenisation

Each MIDI file is parsed with `music21`. Every piece is then projected onto a
fixed time grid (1/16th of a quarter note by default). At every time-step we
record *everything that is sounding* (notes that begin on that step or sustain
into it) and encode it as a single token:

```
R               -> silence at this step
P:C4            -> a single note sounding
P:C4,E4,G4      -> a chord sounding
```

This single-token-per-step scheme keeps rhythm *and* harmony in one sequence,
which is exactly what the LSTM is asked to model.

### Sequence encoding

The token streams are converted to integers (one id per unique token), then
split into fixed-length windows of `sequence_length` steps with one *target*:
the token that comes right after the window (next-step prediction).

The result is cached in `data/processed/` (`.npz` tensors + `vocab.json` +
`meta.json`) so the slow `music21` parsing only happens once.

### Model (`src/model.py`)

A configurable PyTorch network:

```
Embedding(vocab -> embedding_dim)
  -> Dropout
  -> LSTM(embedding_dim -> hidden_dim, num_layers, dropout)
  -> Dropout
  -> Linear(hidden_dim -> vocab)      # next-step logits
```

Everything (embedding size, hidden size, layer count, dropout) is configurable
via the CLI. The training loss is cross-entropy on the next-step prediction;
training is teacher-forced, gradient-clipped, and checkpoints (`latest.pt` +
best-validation `best.pt`) are saved after every epoch.

### Generation (`src/generate.py`)

Generation is **autoregressive**: starting from a seed window, the model
predicts a probability distribution over the next token and we *sample* from it
rather than always taking the argmax. A **temperature** parameter controls
creativity (`< 1` -> safer/repetitive, `> 1` -> more varied), and an optional
`--top-k` keeps sampling inside the k most likely tokens.

The generated token stream is converted back into `music21` notes/chords and
written as a `.mid` file into `outputs/`.

---

## Dataset setup

By default the project uses **[The Mutopia Project](https://www.mutopiaproject.org)**,
an archive of classical and traditional music that is explicitly **public domain
or Creative Commons** ("free cultural works" licenses). No copyrighted
recordings are used.

Run the download script from the project root:

```bash
python scripts/download_dataset.py
```

This downloads a curated set of 16 well-known pieces (Bach, Beethoven, Chopin,
Mozart, Joplin, Pachelbel) into `data/raw/` - a small corpus that already
trains in a few minutes on CPU.

**Want more data?** Two easy options:

```bash
# 1. Crawl an entire composer directory on Mutopia (first N MIDIs found):
python scripts/download_dataset.py --composer BachJS --limit 50

# 2. Provide your own list of direct .mid URLs:
python scripts/download_dataset.py --urls my_urls.txt

# 3. Or simply drop any .mid / .midi files you already have into data/raw/
#    (the preprocessor picks up everything recursively)
```

`--composer` accepts any folder name from <https://www.mutopiaproject.org/ftp>,
e.g. `ChopinFF`, `MozartWA`, `JoplinS`, `PachelbelJ`, `GershwinG`, `Homage` ...

> Mutopia files are small (a few KB each), so the default dataset is included
> directly by the downloader rather than committed to this repository. The
> `.gitignore` excludes downloaded files; if you fork this project you can
> reproduce them with one command.

---

## Installation

Requires **Python 3.9+** (tested on Python 3.14 with PyTorch 2.13).

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs `torch`, `music21` and `numpy` (plus `tqdm` for
progress bars). Everything else uses the standard library.

---

## Preprocessing

```bash
python scripts/preprocess.py
```

Useful flags (defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--raw-dir` | `data/raw` | where MIDI files are read from |
| `--out-dir` | `data/processed` | where tensors/vocab are written |
| `--sequence-length` | `64` | steps per training window |
| `--stride` | `16` | window step between consecutive windows |
| `--step-size` | `0.25` | grid resolution in quarter-note units |
| `--max-tokens-per-file` | `4000` | cap per file (keep the corpus balanced) |
| `--min-tokens-per-file` | `128` | skip files shorter than this |

Malformed, corrupt or unreadable MIDI files are **skipped with a warning**
rather than aborting the run. The dataset is saved once and reused on
subsequent `train.py` runs - pass `--force-preprocess` to `train.py` if you
change preprocessing options.

Output files in `data/processed/`:

- `dataset.npz` - the `(windows, sequence_length)` inputs + targets
- `vocab.json` - token -> integer mapping
- `meta.json` - preprocessing settings + the list of source files used

---

## Training

```bash
python scripts/train.py
```

This loads the cached dataset (preprocessing automatically if needed), builds
the model, and trains. By default it runs **30 epochs** (adjust to taste).

Useful flags (defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs` | `30` | number of training epochs |
| `--batch-size` | `64` | training batch size |
| `--learning-rate` | `1e-3` | Adam learning rate |
| `--sequence-length` | `64` | must match the preprocessed data |
| `--embedding-dim` | `128` | token embedding size |
| `--hidden-dim` | `256` | LSTM hidden size |
| `--num-layers` | `2` | number of stacked LSTM layers |
| `--dropout` | `0.3` | dropout rate |
| `--val-split` | `0.1` | fraction of windows held out for validation |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--seed` | `42` | reproducibility seed |
| `--resume` | - | continue from a checkpoint, e.g. `models/latest.pt` |
| `--force-preprocess` | - | rebuild the cached dataset first |

Training prints a running loss during each epoch, plus a per-epoch summary:

```
epoch   1/30  batch  250/1204  loss 6.5412
>>> epoch 1/30  train_loss 6.8821  val_loss 7.0113
```

Checkpoints are written to `models/`:

- `models/latest.pt` - updated after **every** epoch (safe to interrupt)
- `models/best.pt` - the epoch with the lowest validation loss

Each checkpoint is self-contained: it stores the model weights, the
optimizer state, the vocabulary, the step size and all hyper-parameters, so
generation (or resuming) never needs to guess anything.

A short smoke test (loss drops quickly even on the small default corpus):

```bash
python scripts/train.py --epochs 5 --batch-size 32 --hidden-dim 96 \
    --embedding-dim 64 --num-layers 2 --sequence-length 64
```

---

## Music generation

```bash
python scripts/generate.py
```

Flags (defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoint` | `models/latest.pt` | model to generate with |
| `--length` | `200` | number of time-steps to generate |
| `--temperature` | `1.0` | sampling temperature (see below) |
| `--top-k` | - | sample only among the top-k tokens |
| `--seed-token` | - | explicit starting token, e.g. `P:C4` |
| `--out-dir` | `outputs` | where the `.mid` is written |
| `--output-name` | timestamped | filename of the generated `.mid` |
| `--tempo` | `100` | tempo (BPM) baked into the MIDI |
| `--seed` | `42` | random seed for reproducibility |
| `--to-wav` | - | also render to WAV (needs a synthesizer) |

Without `--seed-token` the model starts from a random window of the training
corpus. With a `--seed-token` the seed note (or chord) is placed at the end of
the context window so the piece starts on that exact token.

**Temperature**:

- `--temperature 0.5` - a faithful, low-risk continuation (near-argmax)
- `--temperature 1.0` - balanced musical sampling
- `--temperature 1.5` - more adventurous / occasionally wild

**Top-k** restricts how many candidate tokens can be sampled, removing the very
long tail of improbable chords; `--temperature 1.0 --top-k 20` is a good,
musical default.

Example:

```bash
python scripts/generate.py --checkpoint models/best.pt --length 300 \
    --temperature 1.1 --top-k 20 --seed-token "P:D4" --output-name etude.mid
```

Output is written to `outputs/etude.mid`, ready to play in any MIDI player,
DAW or notation editor.

---

### Optional: MIDI -> audio rendering

Since MIDI is *instructions* and not sound, hearing the result requires a
synthesizer. `--to-wav` will render the generated piece to a `.wav` using the
first synthesized available on your system:

- **timidity** (built-in instruments, no SoundFont needed)
- **fluidsynth** (needs a SoundFont - set `SOUNDFONT=/path/to/gm.sf2` or pass `--soundfont`)
- **MuseScore** (`mscore`) in "headless" mode

```bash
# Ubuntu / macOS examples
sudo apt install timidity          # -> python scripts/generate.py --to-wav
# or
sudo apt install fluidsynth
fluidsynth /usr/share/sounds/sf2/FluidR3_GM.sf2 outputs/etude.mid -F outputs/etude.wav
# or simply open the .mid in MuseScore / GarageBand / any DAW instead
```

If none of these programs is present the script explains what to install and
does not fail silently. A free, ready-to-go option on Windows is MuseScore 4
(which bundles its own synthesizer).

---

## Example commands

Full pipeline from scratch:

```bash
# 1. get the data
python scripts/download_dataset.py

# 2. preprocess (cached afterwards)
python scripts/preprocess.py --sequence-length 64 --stride 16

# 3. train
python scripts/train.py --epochs 30 --hidden-dim 256 --num-layers 2

# 4. generate (random seed window)
python scripts/generate.py --checkpoint models/best.pt --length 200 --temperature 1.1

# 5. generate from a chosen starting note, render to audio too
python scripts/generate.py --checkpoint models/best.pt --length 200 \
    --seed-token "P:G4" --top-k 20 --to-wav

# 6. continue training from where you left off
python scripts/train.py --epochs 60 --resume models/latest.pt
```

Expected runtime on the default 16-piece corpus (CPU): preprocessing ~30 s,
30 epochs of `--hidden-dim 256` a few minutes. Add files to `data/raw/` (or use
`--composer`) for a richer, more musical model.

---

## Project structure

```text
music-generation-with-ai/
├── data/
│   ├── raw/                     # downloaded .mid files (download_dataset.py)
│   └── processed/               # cached tensors, vocab.json, meta.json
├── models/                      # trained checkpoints (latest.pt, best.pt)
├── outputs/                     # generated .mid (and .wav) files
├── notebooks/
│   └── Music_Generation_Demo.ipynb   # end-to-end demo notebook
├── scripts/
│   ├── download_dataset.py      # Mutopia downloader (+ crawler mode)
│   ├── preprocess.py            # CLI: MIDI -> token windows
│   ├── train.py                 # CLI: training loop
│   └── generate.py              # CLI: generation + optional audio
├── src/
│   ├── __init__.py              # package marker
│   ├── config.py                # paths + Preprocess/Model/Train/Generate configs
│   ├── preprocess.py            # music21 parsing, tokenisation, windowing
│   ├── model.py                 # configurable LSTM (PyTorch)
│   ├── train.py                 # training / validation / checkpointing
│   ├── generate.py              # temperature/top-k sampling, MIDI writing
│   └── audio.py                 # optional MIDI -> WAV conversion
├── .gitignore                   # large reproducible files are not committed
├── CODE_OF_CONDUCT.md           # community conduct guidelines
├── CONTRIBUTING.md              # contribution guidelines
├── LICENSE                      # MIT licence
├── README.md                    # this file
├── requirements.txt             # Python dependencies
└── SECURITY.md                  # security reporting policy
```

---

## Technologies used

- **PyTorch 2.x** - model definition, training loop, sampling
- **music21 9+/10+** - MIDI parsing and MIDI serialisation
- **NumPy** - dataset tensors and windowing
- **Mutopia Project** - copyright-safe source data (public domain / CC)
- Python standard library throughout the CLI scripts

---

## Limitations and possible improvements

**Known limitations**

- The default corpus is small (16 pieces), so the model quickly memorises the
  training pieces and validation loss rises after a few epochs. More data fixes
  this: `--composer` crawl or add files to `data/raw/`.
- Pitches are not transposed to a common key, so the vocabulary (and the
  model's sample space) is large relative to the amount of data.
- Step-based encoding keeps rhythm simple; there is no explicit
  beat/metre-aware embedding.
- Note velocity/tempo changes inside the source files are ignored.
- Generation is an n-gram-like *next-step* model: it produces musical-sounding
  material but has no notion of long-range form or repetition structures.
- Multi-instrument MIDI is flattened into one note grid
  (understandable on a small corpus, but it loses the instrument dimension).
- Rendering to audio requires an external synthesizer (timidity/fluidsynth).

**Possible improvements**

1. Transpose all pieces to a common tonic to shrink the vocabulary.
2. Add duration/velocity tokens, or a [REMI/REMI+](https://arxiv.org/abs/2002.00212)-style
   event encoding with explicit note-onset, duration and tempo events.
3. Use the whole-window teacher-forcing loss (predict every position, not just
   the last) for faster convergence.
4. Try Transformer/GPT-style architectures (e.g. a small decoder-only
   transformer) - LSTMs are a great baseline, transformers scale better.
5. Add a `--key`/mode transposition augraph and dataset augmentation.
6. Implement "priming" with an actual melody/rhythm fragment as seed context.
7. Use a larger permitted corpus (MAESTRO, GiantMIDI-Piano, etc.) - see the
   licensing section before switching data sources.
8. Evaluate generations with objective metrics (pitch-class histograms,
   note-transition likelihood, "human-likeness") and export a loss plot.

---

## License and data rights

- **Code** in this repository is MIT licensed - see `LICENSE`.
- **Data**: The bundled downloader only fetches files from the Mutopia Project
  (public domain or CC-licensed "free cultural works"). The readme of each
  piece states its licence; all are fine to train on, share, and remix.
  If you switch to other datasets (MAESTRO, IMSLP crawls, Kaggle scrapes, ...)
  check *their* licences first - many are research-only or non-commercial.
- No generated MIDI in this project is claimed to be a trained result unless it
  was actually produced by running the training step.

For any questions or issues, open a GitHub issue.