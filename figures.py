"""Figures for the report, drawn from saved metrics and probability caches."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import CLASS_NAMES, CORRUPTIONS

COLORS = {"map": "#4C78A8", "vbll": "#F58518", "gvbll": "#54A24B"}
LABELS = {"map": "MAP", "vbll": "D-VBLL", "gvbll": "G-VBLL"}
MAIN = {
    "map_seed0_eval.json",
    "map_seed1_eval.json",
    "map_seed2_eval.json",
    "vbll_seed0_eval.json",
    "vbll_seed1_eval.json",
    "vbll_seed2_eval.json",
    "gvbll_seed0_eval.json",
    "gvbll_seed1_eval.json",
    "gvbll_seed2_eval.json",
}


def _load_results(output_dir: Path):
    rows = []
    for path in sorted(output_dir.glob("*_eval.json")):
        if path.name not in MAIN:
            continue
        rows.append(json.loads(path.read_text()))
    return rows


def _mean_std(rows, kind, getter):
    vals = [getter(r) for r in rows if r["kind"] == kind]
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def plot_curves(rows, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for kind in ("map", "vbll"):
        histories = [r["history"] for r in rows if r["kind"] == kind]
        epochs = np.array([h["epoch"] for h in histories[0]])
        acc = np.array([[h["val_acc"] for h in hist] for hist in histories])
        nll = np.array([[h["val_nll"] for h in hist] for hist in histories])
        for ax, series, title in (
            (axes[0], acc, "Validation accuracy"),
            (axes[1], nll, "Validation NLL"),
        ):
            mean, std = series.mean(0), series.std(0)
            ax.plot(epochs, mean, color=COLORS[kind], label=LABELS[kind], linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, color=COLORS[kind], alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "curves.png", dpi=160)
    plt.close(fig)


def plot_reliability(rows, output_dir: Path):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1, label="Perfect")
    for kind in ("map", "vbll", "gvbll"):
        chosen = next(r for r in rows if r["kind"] == kind and r["seed"] == 0)
        bins = [b for b in chosen["conditions"]["clean"]["reliability"] if b["count"]]
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            color=COLORS[kind],
            label=LABELS[kind],
        )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability (seed 0)")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "reliability.png", dpi=160)
    plt.close(fig)


def plot_corruptions(rows, output_dir: Path):
    names = list(CORRUPTIONS)
    x = np.arange(len(names))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, key, title, ylim in (
        (axes[0], "accuracy", "Accuracy under corruption", (0, 1)),
        (axes[1], "entropy", "Mean predictive entropy", None),
    ):
        for i, kind in enumerate(("map", "vbll", "gvbll")):
            means, stds = [], []
            for name in names:
                mean, std = _mean_std(
                    rows, kind, lambda r, name=name, key=key: r["conditions"][name][key]
                )
                means.append(mean)
                stds.append(std)
            ax.bar(
                x + (i - 1) * width,
                means,
                width=width,
                yerr=stds,
                color=COLORS[kind],
                label=LABELS[kind],
                capsize=2,
            )
        ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "corruption_bars.png", dpi=160)
    plt.close(fig)


def plot_ood(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, kind in zip(axes, ("map", "vbll")):
        cache = torch.load(output_dir / f"{kind}_seed0_eval.tensors.pt", weights_only=False)
        ax.hist(cache["id_maxprob"].numpy(), bins=30, density=True, alpha=0.75, label="Fashion-MNIST", color="#54A24B")
        ax.hist(cache["ood_maxprob"].numpy(), bins=30, density=True, alpha=0.65, label="MNIST (OOD)", color="#E45756")
        ax.set_title(LABELS[kind])
        ax.set_xlabel("Max predictive probability")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ood_hist.png", dpi=160)
    plt.close(fig)


def plot_confusion(output_dir: Path):
    cache = torch.load(output_dir / "vbll_seed0_eval.tensors.pt", weights_only=False)
    pred = cache["clean_probs"].argmax(dim=-1)
    labels = cache["clean_labels"]
    cm = torch.zeros(10, 10)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1
    cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(cm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("D-VBLL confusion (seed 0, row-normalized)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion.png", dpi=160)
    plt.close(fig)


def plot_gallery(output_dir: Path):
    cache = torch.load(output_dir / "vbll_seed0_eval.tensors.pt", weights_only=False)
    images = cache["gallery_images"]
    labels = cache["gallery_labels"]
    probs = cache["gallery_probs"]
    n = images.shape[0]
    fig, axes = plt.subplots(2, 6, figsize=(9.2, 5.4))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        pred = int(probs[i].argmax())
        conf = float(probs[i, pred])
        ok = pred == int(labels[i])
        color = "#1B7F3A" if ok else "#B00020"
        ax.set_title(
            f"{CLASS_NAMES[pred]} {conf:.2f}\ntrue {CLASS_NAMES[int(labels[i])]}",
            fontsize=8,
            color=color,
            pad=4,
        )
    fig.suptitle("D-VBLL confident mistakes and high-entropy cases", fontsize=11)
    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.88, bottom=0.04)
    fig.savefig(output_dir / "gallery.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_norm_entropy(rows, output_dir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for kind in ("map", "vbll"):
        chosen = [r for r in rows if r["kind"] == kind]
        mids, ents = [], []
        for i in range(5):
            lo = np.mean([r["feature_norm_entropy"][i]["norm_lo"] for r in chosen])
            hi = np.mean([r["feature_norm_entropy"][i]["norm_hi"] for r in chosen])
            ent = np.mean([r["feature_norm_entropy"][i]["entropy"] for r in chosen])
            mids.append(0.5 * (lo + hi))
            ents.append(ent)
        ax.plot(mids, ents, marker="o", color=COLORS[kind], label=LABELS[kind], linewidth=2)
    ax.set_xlabel("Feature L2 norm (bin centre)")
    ax.set_ylabel("Mean predictive entropy")
    ax.set_title("Uncertainty versus feature norm")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "norm_entropy.png", dpi=160)
    plt.close(fig)


def write_summary(rows, output_dir: Path):
    summary = {"models": {}}
    for kind in ("map", "vbll"):
        block = {}
        for name in CORRUPTIONS:
            for key in ("accuracy", "nll", "ece", "entropy"):
                mean, std = _mean_std(rows, kind, lambda r, name=name, key=key: r["conditions"][name][key])
                block[f"{name}_{key}"] = {"mean": mean, "std": std}
        for key in ("auroc_maxprob", "fashion_maxprob", "mnist_maxprob", "mnist_entropy"):
            mean, std = _mean_std(rows, kind, lambda r, key=key: r["ood"][key])
            block[key] = {"mean": mean, "std": std}
        stats = [r["head_stats"] for r in rows if r["kind"] == kind and r["head_stats"]]
        if stats:
            block["head_stats"] = stats
        summary["models"][kind] = block
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_all(output_dir: Path):
    rows = _load_results(output_dir)
    plot_curves(rows, output_dir)
    plot_reliability(rows, output_dir)
    plot_corruptions(rows, output_dir)
    plot_ood(output_dir)
    plot_confusion(output_dir)
    plot_gallery(output_dir)
    plot_norm_entropy(rows, output_dir)
    summary = write_summary(rows, output_dir)
    print(json.dumps(summary, indent=2))
