"""Accuracy, negative log-likelihood, calibration, and OOD scores."""

import torch


def summarize(probabilities: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> dict:
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels)
    picked = probabilities[torch.arange(labels.shape[0]), labels].clamp_min(1e-8)
    nll = -picked.log().mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    accuracy = correct.float().mean()

    edges = torch.linspace(0, 1, n_bins + 1)
    ece = probabilities.new_zeros(())
    reliability = []
    for i in range(n_bins):
        if i == 0:
            mask = confidence <= edges[i + 1]
        else:
            mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            reliability.append({"bin": i, "count": 0, "confidence": None, "accuracy": None})
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].float().mean()
        ece = ece + (count / labels.shape[0]) * (bin_acc - bin_conf).abs()
        reliability.append(
            {
                "bin": i,
                "count": count,
                "confidence": float(bin_conf),
                "accuracy": float(bin_acc),
            }
        )

    per_class = []
    for c in range(probabilities.shape[1]):
        mask = labels.eq(c)
        per_class.append(float(correct[mask].float().mean()) if int(mask.sum()) else None)

    return {
        "accuracy": float(accuracy),
        "nll": float(nll),
        "ece": float(ece),
        "entropy": float(entropy.mean()),
        "max_prob": float(confidence.mean()),
        "per_class_accuracy": per_class,
        "reliability": reliability,
    }


def ood_scores(probabilities: torch.Tensor) -> torch.Tensor:
    """Higher means more in-distribution. Maximum softmax / predictive probability."""
    return probabilities.max(dim=-1).values
