"""Train one softmax MAP or discriminative VBLL classifier."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_utils import augment_batch, normalize, train_val_split
from metrics import summarize
from models.classifier import ImageClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(images, labels, batch_size, shuffle):
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_split(model, images, labels, batch_size, n_samples, device):
    model.eval()
    probs, ys = [], []
    for x, y in _loader(images, labels, batch_size, shuffle=False):
        x = normalize(x.to(device))
        probs.append(model.probabilities(x, n_samples=n_samples).cpu())
        ys.append(y)
    return summarize(torch.cat(probs), torch.cat(ys))


def train_one(
    images,
    labels,
    kind: str,
    seed: int,
    out_path: Path,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    feat_dim: int = 128,
    n_samples: int = 16,
    reg_scale: float = 1.0,
):
    set_seed(seed)
    device = get_device()
    x_train, y_train, x_val, y_val = train_val_split(images, labels)
    model = ImageClassifier(
        kind, n_train=x_train.shape[0], feat_dim=feat_dim, reg_scale=reg_scale
    ).to(device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(weight_decay), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for x, y in _loader(x_train, y_train, batch_size, shuffle=True):
            x = augment_batch(x.to(device))
            x = normalize(x)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, pred = model.training_step(x, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * y.shape[0]
            total_correct += int(pred.eq(y).sum())
            total_n += y.shape[0]
        scheduler.step()
        val = evaluate_split(model, x_val, y_val, batch_size, n_samples, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_n,
            "train_acc": total_correct / total_n,
            "val_acc": val["accuracy"],
            "val_nll": val["nll"],
            "val_ece": val["ece"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"{kind} seed={seed} epoch={epoch:02d} "
            f"loss={row['train_loss']:.4f} train_acc={row['train_acc']:.3f} "
            f"val_acc={row['val_acc']:.3f} val_nll={row['val_nll']:.3f} val_ece={row['val_ece']:.3f}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "seed": seed,
        "feat_dim": feat_dim,
        "n_train": int(x_train.shape[0]),
        "epochs": epochs,
        "reg_scale": reg_scale,
        "state_dict": model.state_dict(),
        "history": history,
        "head_stats": model.head.stats() if kind == "vbll" else {},
    }
    torch.save(payload, out_path)
    history_path = out_path.with_suffix(".history.json")
    history_path.write_text(json.dumps({"history": history, "head_stats": payload["head_stats"]}, indent=2))
    return out_path
