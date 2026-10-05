"""KL-weight ablation for D-VBLL. Scale 1 is the paper's 1/T setting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_split
from evaluate import evaluate_checkpoint
from train import train_one


def main():
    out = Path("outputs")
    images, labels = load_split("data", "fashion", train=True)
    fashion_test = load_split("data", "fashion", train=False)
    mnist_test = load_split("data", "mnist", train=False)
    rows = []
    base = json.loads((out / "vbll_seed0_eval.json").read_text())
    rows.append(
        {
            "reg_scale": 1.0,
            "accuracy": base["conditions"]["clean"]["accuracy"],
            "nll": base["conditions"]["clean"]["nll"],
            "ece": base["conditions"]["clean"]["ece"],
            "entropy": base["conditions"]["clean"]["entropy"],
            "auroc": base["ood"]["auroc_maxprob"],
            "kl": base["head_stats"].get("kl"),
            "mean_weight_var": base["head_stats"].get("mean_weight_var"),
        }
    )
    for scale in (10.0, 100.0):
        ckpt = out / f"vbll_seed0_reg{int(scale)}.pt"
        train_one(
            images,
            labels,
            kind="vbll",
            seed=0,
            out_path=ckpt,
            epochs=12,
            reg_scale=scale,
        )
        result = evaluate_checkpoint(
            ckpt,
            fashion_test,
            mnist_test,
            out_json=out / f"vbll_seed0_reg{int(scale)}_eval.json",
        )
        rows.append(
            {
                "reg_scale": scale,
                "accuracy": result["conditions"]["clean"]["accuracy"],
                "nll": result["conditions"]["clean"]["nll"],
                "ece": result["conditions"]["clean"]["ece"],
                "entropy": result["conditions"]["clean"]["entropy"],
                "auroc": result["ood"]["auroc_maxprob"],
                "kl": result["head_stats"].get("kl"),
                "mean_weight_var": result["head_stats"].get("mean_weight_var"),
            }
        )
    (out / "kl_ablation.json").write_text(json.dumps(rows, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1))
    xs = [r["reg_scale"] for r in rows]
    labels = [f"{int(x)}/T" for x in xs]
    specs = (
        (axes[0], "accuracy", "Clean accuracy", (0.7, 1.0)),
        (axes[1], "ece", "Clean ECE", None),
        (axes[2], "entropy", "Clean entropy", None),
    )
    for ax, key, title, ylim in specs:
        ax.bar(labels, [r[key] for r in rows], color="#F58518")
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "kl_ablation.png", dpi=160)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
