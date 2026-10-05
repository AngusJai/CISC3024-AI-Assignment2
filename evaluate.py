"""Test-set accuracy, calibration, corruptions, and MNIST out-of-distribution detection."""

import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from data_utils import CORRUPTIONS, corrupt, normalize
from metrics import ood_scores, summarize
from models.classifier import ImageClassifier
from train import get_device


def load_model(ckpt_path: Path, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ImageClassifier(
        payload["kind"],
        n_train=payload["n_train"],
        feat_dim=payload["feat_dim"],
        reg_scale=payload.get("reg_scale", 1.0),
        prior_scale=payload.get("prior_scale", 1.0),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


@torch.no_grad()
def collect(model, images, labels, device, n_samples, corruption="clean", seed=0, batch_size=256):
    probs, ys, feats = [], [], []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        y = labels[start : start + batch_size]
        viewed = corrupt(batch, corruption, seed=seed + start)
        x = normalize(viewed.to(device))
        features = model.features(x)
        if model.kind == "map":
            p = torch.softmax(model.head(features), dim=-1)
        else:
            p = model.head.predictive(features, n_samples=n_samples)
        probs.append(p.cpu())
        ys.append(y)
        feats.append(features.cpu())
    return torch.cat(probs), torch.cat(ys), torch.cat(feats)


def auroc_id_vs_ood(id_probs, ood_probs) -> float:
    scores = torch.cat([ood_scores(id_probs), ood_scores(ood_probs)]).numpy()
    labels = torch.cat(
        [
            torch.ones(id_probs.shape[0]),
            torch.zeros(ood_probs.shape[0]),
        ]
    ).numpy()
    return float(roc_auc_score(labels, scores))


def evaluate_checkpoint(
    ckpt_path: Path,
    fashion_test,
    mnist_test,
    out_json: Path,
    n_samples: int = 32,
):
    device = get_device()
    model, payload = load_model(ckpt_path, device)
    x_test, y_test = fashion_test
    x_ood, y_ood = mnist_test

    conditions = {}
    clean_probs = clean_labels = clean_feats = None
    for name in CORRUPTIONS:
        probs, labels, feats = collect(
            model, x_test, y_test, device, n_samples, corruption=name, seed=1000
        )
        summary = summarize(probs, labels)
        print(
            f"eval {payload['kind']} seed={payload['seed']} {name}: "
            f"acc={summary['accuracy']:.3f} nll={summary['nll']:.3f} "
            f"ece={summary['ece']:.3f} H={summary['entropy']:.3f}",
            flush=True,
        )
        if name != "clean":
            summary = {k: v for k, v in summary.items() if k != "reliability"}
        else:
            clean_probs, clean_labels, clean_feats = probs, labels, feats
        conditions[name] = summary

    ood_probs, _, ood_feats = collect(
        model, x_ood, y_ood, device, n_samples, corruption="clean", seed=1000
    )
    auroc = auroc_id_vs_ood(clean_probs, ood_probs)
    print(f"eval {payload['kind']} seed={payload['seed']} OOD AUROC={auroc:.3f}", flush=True)

    result = {
        "kind": payload["kind"],
        "seed": payload["seed"],
        "history": payload["history"],
        "head_stats": payload.get("head_stats", {}),
        "conditions": conditions,
        "ood": {
            "auroc_maxprob": auroc,
            "fashion_maxprob": float(ood_scores(clean_probs).mean()),
            "mnist_maxprob": float(ood_scores(ood_probs).mean()),
            "fashion_entropy": float(conditions["clean"]["entropy"]),
            "mnist_entropy": float(
                -(ood_probs * ood_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            ),
        },
    }
    # Feature-norm bins on the clean set, used to check distance-aware uncertainty.
    norms = clean_feats.norm(dim=-1)
    edges = torch.quantile(norms, torch.linspace(0, 1, 6))
    bins = []
    entropy = -(clean_probs * clean_probs.clamp_min(1e-8).log()).sum(dim=-1)
    for i in range(5):
        if i < 4:
            mask = (norms >= edges[i]) & (norms < edges[i + 1])
        else:
            mask = (norms >= edges[i]) & (norms <= edges[i + 1])
        bins.append(
            {
                "norm_lo": float(edges[i]),
                "norm_hi": float(edges[i + 1]),
                "entropy": float(entropy[mask].mean()) if int(mask.sum()) else None,
                "count": int(mask.sum()),
            }
        )
    result["feature_norm_entropy"] = bins
    result["feature_norm_mean"] = {
        "fashion": float(clean_feats.norm(dim=-1).mean()),
        "mnist": float(ood_feats.norm(dim=-1).mean()),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    prediction = clean_probs.argmax(dim=-1)
    entropy = -(clean_probs * clean_probs.clamp_min(1e-8).log()).sum(dim=-1)
    confidence = clean_probs.max(dim=-1).values
    wrong = prediction.ne(clean_labels)
    right = ~wrong
    hard_wrong = torch.where(wrong)[0][confidence[wrong].argsort(descending=True)[:8]]
    unsure = entropy.argsort(descending=True)[:8]
    gallery_idx = torch.unique(torch.cat([hard_wrong, unsure]), sorted=False)[:12]

    cache = out_json.with_suffix(".tensors.pt")
    torch.save(
        {
            "clean_probs": clean_probs,
            "clean_labels": clean_labels,
            "id_maxprob": ood_scores(clean_probs),
            "ood_maxprob": ood_scores(ood_probs),
            "gallery_images": x_test[gallery_idx],
            "gallery_labels": clean_labels[gallery_idx],
            "gallery_probs": clean_probs[gallery_idx],
        },
        cache,
    )
    return result
