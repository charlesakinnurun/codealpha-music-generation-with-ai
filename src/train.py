"""Training and validation loop for the LSTM music model."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfig, TrainConfig
from .model import MusicLSTM

def set_seed(seed: int) -> None:
    """Pin down all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    """Translate a user requested device into a concrete device string."""
    if requested in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def make_loaders(
    sequences: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Split windows into train/validation tensors and build DataLoaders."""
    n = len(sequences)
    if val_split > 0:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        train_idx = np.arange(n)
        val_idx = np.array([], dtype=int)

    train_x = torch.as_tensor(sequences[train_idx], dtype=torch.long)
    train_y = torch.as_tensor(targets[train_idx], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader: DataLoader | None = None
    if len(val_idx) > 0:
        val_x = torch.as_tensor(sequences[val_idx], dtype=torch.long)
        val_y = torch.as_tensor(targets[val_idx], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)

    if val_loader is None:
        val_loader = train_loader

    return train_loader, val_loader


def train_epoch(
    model: MusicLSTM,
    loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    log_interval: int,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one training epoch with teacher forcing; returns mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = max(1, len(loader))

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)                    # (B, S, V)
        # The dataset stores one target (the next token) per window, so we
        # evaluate the prediction made at the last time-step of the window.
        loss = criterion(logits[:, -1, :], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        if step == 1 or step % log_interval == 0 or step == num_batches:
            print(
                f"epoch {epoch:>3d}/{total_epochs}  "
                f"batch {step:>5d}/{num_batches}  "
                f"loss {loss.item():.4f}"
            )

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: MusicLSTM, loader: DataLoader, criterion, device: str) -> float:
    """Compute the average loss over a (validation) loader."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits[:, -1, :], y)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(1, num_batches)


def save_checkpoint(path: Path, **payload) -> None:
    """Persist a model checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path), _use_new_zipfile_serialization=True)
    print(f"[checkpoint] saved {path}")


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint, detaching it from the device it was trained on."""
    return torch.load(str(path), map_location=map_location, weights_only=False)


def train(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    sequences: np.ndarray,
    targets: np.ndarray,
    vocab: dict[str, int],
    idx_to_token: list[str],
    step_size: float,
) -> dict:
    """Run the full training run.

    Returns a history dict with per-epoch ``train_loss`` / ``val_loss`` and
    the path of the best checkpoint. Visible loss is printed to stdout.
    """
    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    print(f"Device: {device}")

    create_new = train_cfg.resume is None
    if create_new:
        model = MusicLSTM.from_config(model_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
        start_epoch = 0
        val_loss = float("inf")
    else:
        ckpt = load_checkpoint(Path(train_cfg.resume), map_location=device)
        vocab_size = ckpt["model_cfg"]["vocab_size"]
        model = MusicLSTM.from_config(ModelConfig.from_dict(ckpt["model_cfg"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from {train_cfg.resume} (epoch {start_epoch})")

    criterion = torch.nn.CrossEntropyLoss()
    models_dir = Path(train_cfg.models_dir)

    if train_cfg.epochs <= start_epoch:
        print(
            f"Checkpoint is already at epoch {start_epoch}, but you requested "
            f"--epochs {train_cfg.epochs}. Use --epochs N with N > {start_epoch} "
            "to continue training."
        )
        return {"train_loss": [], "val_loss": [], "epoch": [],
                "best_checkpoint": str(models_dir / "best.pt"),
                "latest_checkpoint": str(models_dir / train_cfg.checkpoint_name)}

    train_loader, val_loader = make_loaders(
        sequences,
        targets,
        batch_size=train_cfg.batch_size,
        val_split=train_cfg.val_split,
        seed=train_cfg.seed,
    )

    latest_path = models_dir / train_cfg.checkpoint_name
    best_path = models_dir / "best.pt"

    print(f"Batch size : {train_cfg.batch_size}")
    print(f"Sequences  : {len(sequences)}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")

    history = {"train_loss": [], "val_loss": [], "epoch": []}

    for epoch in range(start_epoch + 1, train_cfg.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_cfg.log_interval,
            epoch,
            train_cfg.epochs,
        )
        epoch_val_loss = (
            evaluate(model, val_loader, criterion, device)
            if train_cfg.val_split > 0
            else train_loss
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["epoch"].append(epoch)

        print(
            f">>> epoch {epoch}/{train_cfg.epochs}  "
            f"train_loss {train_loss:.4f}  val_loss {epoch_val_loss:.4f}"
        )

        save_checkpoint(
            latest_path,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            train_loss=train_loss,
            val_loss=epoch_val_loss,
            model_cfg=model_cfg.to_dict(),
            train_cfg=train_cfg.to_dict(),
            vocab=vocab,
            idx_to_token=idx_to_token,
            step_size=step_size,
            sequence_length=int(sequences.shape[1]),
        )

        if epoch_val_loss < val_loss:
            val_loss = epoch_val_loss
            save_checkpoint(
                best_path,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                train_loss=train_loss,
                val_loss=epoch_val_loss,
                model_cfg=model_cfg.to_dict(),
                train_cfg=train_cfg.to_dict(),
                vocab=vocab,
                idx_to_token=idx_to_token,
                step_size=step_size,
                sequence_length=int(sequences.shape[1]),
            )
            print(f"*** new best validation loss {epoch_val_loss:.4f} (epoch {epoch})")

    print(f"Training finished. Best checkpoint: {best_path}")
    history["best_checkpoint"] = str(best_path)
    history["latest_checkpoint"] = str(latest_path)
    return history