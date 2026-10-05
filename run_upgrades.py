"""Extra runs: third seed, generative VBLL, second-seed KL ablation, prior scale."""

from pathlib import Path

from data_utils import load_split
from evaluate import evaluate_checkpoint
from models.vbll import self_check
from train import train_one


JOBS = (
    ("map", 2, 1.0, 1.0, "map_seed2.pt"),
    ("vbll", 2, 1.0, 1.0, "vbll_seed2.pt"),
    ("gvbll", 0, 1.0, 1.0, "gvbll_seed0.pt"),
    ("gvbll", 1, 1.0, 1.0, "gvbll_seed1.pt"),
    ("gvbll", 2, 1.0, 1.0, "gvbll_seed2.pt"),
    ("vbll", 1, 10.0, 1.0, "vbll_seed1_reg10.pt"),
    ("vbll", 1, 100.0, 1.0, "vbll_seed1_reg100.pt"),
    ("vbll", 0, 1.0, 0.1, "vbll_seed0_prior0p1.pt"),
    ("vbll", 0, 1.0, 10.0, "vbll_seed0_prior10.pt"),
)


def main():
    self_check()
    out = Path("outputs")
    images, labels = load_split("data", "fashion", train=True)
    fashion_test = load_split("data", "fashion", train=False)
    mnist_test = load_split("data", "mnist", train=False)
    for kind, seed, reg_scale, prior_scale, name in JOBS:
        ckpt = out / name
        done = out / f"{ckpt.stem}_eval.json"
        if done.exists():
            print(f"skip {name}", flush=True)
            continue
        if not ckpt.exists():
            print(f"training {name}", flush=True)
            train_one(
                images,
                labels,
                kind=kind,
                seed=seed,
                out_path=ckpt,
                epochs=12,
                reg_scale=reg_scale,
                prior_scale=prior_scale,
            )
        else:
            print(f"evaluating existing {name}", flush=True)
        evaluate_checkpoint(ckpt, fashion_test, mnist_test, out_json=done)


if __name__ == "__main__":
    main()
