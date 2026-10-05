"""Post-hoc checks that do not retrain the main models.

1. Brier score, misclassification-detection AUROC, and risk-coverage from saved probabilities.
2. Temperature scaling of the softmax MAP head, fit on the validation split.
3. Classical Bayes baselines: Gaussian naive Bayes on pixels, and shrinkage LDA on frozen MAP features.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

from data_utils import load_split, normalize, train_val_split
from evaluate import load_model
from metrics import summarize
from train import get_device


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    target = torch.zeros_like(probs)
    target[torch.arange(labels.shape[0]), labels] = 1.0
    return float((probs - target).square().sum(dim=-1).mean())


def detection_auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Does confidence rank correct predictions above mistakes?"""
    confidence = probs.max(dim=-1).values.numpy()
    correct = probs.argmax(dim=-1).eq(labels).numpy().astype(np.int32)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, confidence))


def risk_coverage(probs: torch.Tensor, labels: torch.Tensor):
    confidence = probs.max(dim=-1).values
    correct = probs.argmax(dim=-1).eq(labels)
    order = confidence.argsort(descending=True)
    correct = correct[order].float()
    count = torch.arange(1, correct.shape[0] + 1, dtype=torch.float32)
    selective_acc = correct.cumsum(0) / count
    coverage = count / count[-1]
    risk = 1.0 - selective_acc
    aurc = float(torch.trapz(risk, coverage))
    keep = int(0.9 * correct.shape[0])
    acc_at_90 = float(correct[:keep].mean())
    return coverage.numpy(), risk.numpy(), aurc, acc_at_90


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp_min(1e-4), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp())


@torch.no_grad()
def map_logits(model, images, device, batch_size=256):
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        batch = normalize(images[start : start + batch_size].to(device))
        chunks.append(model.head(model.features(batch)).cpu())
    return torch.cat(chunks)


@torch.no_grad()
def map_features(model, images, device, batch_size=256):
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        batch = normalize(images[start : start + batch_size].to(device))
        chunks.append(model.features(batch).cpu())
    return torch.cat(chunks)


def classical_baselines(device):
    images, labels = load_split("data", "fashion", train=True)
    x_test, y_test = load_split("data", "fashion", train=False)
    x_train, y_train, _, _ = train_val_split(images, labels)
    pixel_train = x_train.view(x_train.shape[0], -1).numpy()
    pixel_test = x_test.view(x_test.shape[0], -1).numpy()

    gnb = GaussianNB()
    gnb.fit(pixel_train, y_train.numpy())
    gnb_probs = torch.tensor(gnb.predict_proba(pixel_test), dtype=torch.float32)
    gnb_summary = summarize(gnb_probs, y_test)
    gnb_summary["brier"] = brier_score(gnb_probs, y_test)

    model, _ = load_model(Path("outputs/map_seed0.pt"), device)
    feat_train = map_features(model, x_train, device).numpy()
    feat_test = map_features(model, x_test, device).numpy()
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(feat_train, y_train.numpy())
    lda_probs = torch.tensor(lda.predict_proba(feat_test), dtype=torch.float32)
    lda_summary = summarize(lda_probs, y_test)
    lda_summary["brier"] = brier_score(lda_probs, y_test)
    return {
        "gaussian_nb_pixels": {k: gnb_summary[k] for k in ("accuracy", "nll", "ece", "brier")},
        "lda_on_map_features": {k: lda_summary[k] for k in ("accuracy", "nll", "ece", "brier")},
    }


def main():
    out = Path("outputs")
    device = get_device()
    images, labels = load_split("data", "fashion", train=True)
    x_test, y_test = load_split("data", "fashion", train=False)
    _, _, x_val, y_val = train_val_split(images, labels)

    rows = []
    curves = {}
    for kind in ("map", "vbll", "gvbll"):
        for seed in (0, 1, 2):
            cache = torch.load(out / f"{kind}_seed{seed}_eval.tensors.pt", weights_only=False)
            probs, y = cache["clean_probs"], cache["clean_labels"]
            coverage, risk, aurc, acc90 = risk_coverage(probs, y)
            row = {
                "kind": kind,
                "seed": seed,
                "brier": brier_score(probs, y),
                "error_auroc": detection_auroc(probs, y),
                "aurc": aurc,
                "acc_at_90_coverage": acc90,
            }
            rows.append(row)
            if seed == 0:
                curves[kind] = (coverage[::20], risk[::20])
            print(kind, seed, row, flush=True)

    temperatures = []
    for seed in (0, 1):
        model, _ = load_model(out / f"map_seed{seed}.pt", device)
        val_logits = map_logits(model, x_val, device)
        test_logits = map_logits(model, x_test, device)
        temperature = fit_temperature(val_logits, y_val)
        scaled = torch.softmax(test_logits / temperature, dim=-1)
        summary = summarize(scaled, y_test)
        temperatures.append(
            {
                "seed": seed,
                "temperature": temperature,
                "accuracy": summary["accuracy"],
                "nll": summary["nll"],
                "ece": summary["ece"],
                "brier": brier_score(scaled, y_test),
                "error_auroc": detection_auroc(scaled, y_test),
            }
        )
        print("temperature", temperatures[-1], flush=True)

    classical = classical_baselines(device)
    print("classical", classical, flush=True)

    payload = {"predictive": rows, "temperature_scaling": temperatures, "classical": classical}
    (out / "extra_analysis.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colors = {"map": "#4C78A8", "vbll": "#F58518", "gvbll": "#54A24B"}
    labels = {"map": "Softmax MAP", "vbll": "D-VBLL", "gvbll": "G-VBLL"}
    for kind, (coverage, risk) in curves.items():
        ax.plot(coverage, risk, color=colors[kind], label=labels[kind], linewidth=2)
    ax.set_xlabel("Coverage (fraction kept)")
    ax.set_ylabel("Selective risk")
    ax.set_title("Risk-coverage, seed 0")
    ax.set_xlim(0.2, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
