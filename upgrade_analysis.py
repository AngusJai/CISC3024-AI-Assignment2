"""Figures and tables for the follow-up experiments. Run after run_upgrades.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from data_utils import CLASS_NAMES, load_split, normalize, train_val_split
from evaluate import collect, load_model
from extra_analysis import brier_score, detection_auroc, map_features
from metrics import summarize
from train import get_device

OUT = Path("outputs")
SHIRT = CLASS_NAMES.index("Shirt")
TSHIRT = CLASS_NAMES.index("T-shirt")


def _mean(vals):
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": [float(v) for v in arr]}


def _load_eval(path: Path):
    row = json.loads(path.read_text())
    clean = row["conditions"]["clean"]
    return {
        "accuracy": clean["accuracy"],
        "nll": clean["nll"],
        "ece": clean["ece"],
        "entropy": clean["entropy"],
        "auroc": row["ood"]["auroc_maxprob"],
        "head": row.get("head_stats") or {},
        "per_class": clean.get("per_class_accuracy"),
    }


def _group(paths):
    rows = [_load_eval(p) for p in paths]
    keys = ("accuracy", "nll", "ece", "entropy", "auroc")
    return {key: _mean([row[key] for row in rows]) for key in keys}


def selective_at(probs, labels, frac=0.9):
    confidence = probs.max(dim=-1).values
    correct = probs.argmax(dim=-1).eq(labels)
    keep = max(1, int(round(frac * labels.shape[0])))
    chosen = confidence.argsort(descending=True)[:keep]
    return {
        "accuracy": float(correct.float().mean()),
        "acc_at_90": float(correct[chosen].float().mean()),
    }


def shirt_block(probs, labels):
    subset = labels.eq(SHIRT)
    summary = summarize(probs[subset], labels[subset])
    pred = probs.argmax(dim=-1)
    shirt = labels.eq(SHIRT)
    return {
        "accuracy": summary["accuracy"],
        "ece": summary["ece"],
        "mean_confidence": float(probs[shirt].max(dim=-1).values.mean()),
        "to_tshirt": float(pred[shirt].eq(TSHIRT).float().mean()),
    }


def classical_with_ranking(device):
    images, labels = load_split("data", "fashion", train=True)
    x_test, y_test = load_split("data", "fashion", train=False)
    x_train, y_train, _, _ = train_val_split(images, labels)
    gnb = GaussianNB()
    gnb.fit(x_train.view(len(x_train), -1).numpy(), y_train.numpy())
    gnb_probs = torch.tensor(
        gnb.predict_proba(x_test.view(len(x_test), -1).numpy()), dtype=torch.float32
    )
    model, _ = load_model(OUT / "map_seed0.pt", device)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(map_features(model, x_train, device).numpy(), y_train.numpy())
    lda_probs = torch.tensor(
        lda.predict_proba(map_features(model, x_test, device).numpy()), dtype=torch.float32
    )
    out = {}
    for name, probs in (("gaussian_nb_pixels", gnb_probs), ("lda_on_map_features", lda_probs)):
        summary = summarize(probs, y_test)
        out[name] = {
            "accuracy": summary["accuracy"],
            "nll": summary["nll"],
            "ece": summary["ece"],
            "brier": brier_score(probs, y_test),
            "error_auroc": detection_auroc(probs, y_test),
        }
    return out


def plot_cdf():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15), sharey=True)
    specs = (
        ("map_seed0_eval.tensors.pt", "Softmax MAP"),
        ("vbll_seed0_eval.tensors.pt", "D-VBLL"),
        ("gvbll_seed0_eval.tensors.pt", "G-VBLL"),
    )
    for ax, (name, title) in zip(axes, specs):
        cache = torch.load(OUT / name, weights_only=False)
        for scores, label, color in (
            (cache["id_maxprob"], "Fashion-MNIST", "#54A24B"),
            (cache["ood_maxprob"], "MNIST", "#E45756"),
        ):
            ordered = np.sort(scores.numpy())
            cdf = np.linspace(0, 1, ordered.shape[0], endpoint=False)
            ax.plot(ordered, cdf, color=color, label=label, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Max predictive probability")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ood_cdf.png", dpi=160)
    plt.close(fig)


def plot_selective(rows):
    conditions = ["clean", "rotate_60", "low_contrast"]
    kinds = ["map", "vbll", "gvbll"]
    labels = {"map": "MAP", "vbll": "D-VBLL", "gvbll": "G-VBLL"}
    colors = {"map": "#4C78A8", "vbll": "#F58518", "gvbll": "#54A24B"}
    fig, ax = plt.subplots(figsize=(6.4, 2.85))
    x = np.arange(len(conditions))
    width = 0.24
    for i, kind in enumerate(kinds):
        means, stds = [], []
        for condition in conditions:
            vals = [r["acc_at_90"] for r in rows if r["kind"] == kind and r["condition"] == condition]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=stds,
            color=colors[kind],
            label=labels[kind],
            capsize=2,
        )
    ax.set_xticks(x, ["Clean", "Rotate 60°", "Low contrast"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy among top 90% confidence")
    ax.set_title("Rejecting the least confident 10%")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "selective_corruption.png", dpi=160)
    plt.close(fig)


def main():
    device = get_device()
    x_test, y_test = load_split("data", "fashion", train=False)
    x_ood, y_ood = load_split("data", "mnist", train=False)
    del y_ood

    summary = {
        "map": _group([OUT / f"map_seed{s}_eval.json" for s in (0, 1, 2)]),
        "vbll": _group([OUT / f"vbll_seed{s}_eval.json" for s in (0, 1, 2)]),
        "gvbll": _group([OUT / f"gvbll_seed{s}_eval.json" for s in (0, 1, 2)]),
    }

    kl_rows = []
    for scale, seeds in ((1, (0, 1, 2)), (10, (0, 1)), (100, (0, 1))):
        paths = []
        for seed in seeds:
            path = OUT / (f"vbll_seed{seed}_eval.json" if scale == 1 else f"vbll_seed{seed}_reg{scale}_eval.json")
            paths.append(path)
        block = _group(paths)
        block["scale"] = scale
        kl_rows.append(block)

    prior_rows = []
    for scale, name in ((0.1, "vbll_seed0_prior0p1_eval.json"), (1.0, "vbll_seed0_eval.json"), (10.0, "vbll_seed0_prior10_eval.json")):
        row = _load_eval(OUT / name)
        row["prior_scale"] = scale
        prior_rows.append(row)

    mc_rows = []
    shirt_rows = []
    brier_rows = []
    for kind, seeds in (("map", (0, 1, 2)), ("vbll", (0, 1, 2)), ("gvbll", (0, 1, 2))):
        for seed in seeds:
            cache = torch.load(OUT / f"{kind}_seed{seed}_eval.tensors.pt", weights_only=False)
            probs, labels = cache["clean_probs"], cache["clean_labels"]
            shirt_rows.append({"kind": kind, "seed": seed, **shirt_block(probs, labels)})
            brier_rows.append({"kind": kind, "seed": seed, "brier": brier_score(probs, labels), "error_auroc": detection_auroc(probs, labels)})

    for seed in (0, 1, 2):
        model, _ = load_model(OUT / f"vbll_seed{seed}.pt", device)
        for n_samples in (8, 32, 64):
            probs, labels, _ = collect(model, x_test, y_test, device, n_samples=n_samples)
            stats = summarize(probs, labels)
            mc_rows.append(
                {
                    "seed": seed,
                    "n_samples": n_samples,
                    "accuracy": stats["accuracy"],
                    "nll": stats["nll"],
                    "ece": stats["ece"],
                }
            )
            print("mc", mc_rows[-1], flush=True)

    selective_rows = []
    for kind, seeds in (("map", (0, 1, 2)), ("vbll", (0, 1, 2)), ("gvbll", (0, 1, 2))):
        for seed in seeds:
            model, _ = load_model(OUT / f"{kind}_seed{seed}.pt", device)
            for condition in ("clean", "rotate_60", "low_contrast"):
                probs, labels, _ = collect(
                    model, x_test, y_test, device, n_samples=32, corruption=condition, seed=1000
                )
                row = {"kind": kind, "seed": seed, "condition": condition, **selective_at(probs, labels)}
                selective_rows.append(row)
                print("selective", row, flush=True)

    plot_cdf()
    plot_selective(selective_rows)
    payload = {
        "three_seed": summary,
        "kl": kl_rows,
        "prior": prior_rows,
        "mc": mc_rows,
        "shirt": shirt_rows,
        "brier": brier_rows,
        "selective": selective_rows,
        "classical": classical_with_ranking(device),
    }
    (OUT / "upgrade_summary.json").write_text(json.dumps(payload, indent=2))
    print("wrote upgrade_summary.json", flush=True)


if __name__ == "__main__":
    main()
