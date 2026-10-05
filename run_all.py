"""Train softmax MAP and D-VBLL, then evaluate and draw figures.

Usage:
    python run_all.py
    python run_all.py --epochs 12 --seeds 0 1
"""

import argparse
from pathlib import Path

from data_utils import load_split
from evaluate import evaluate_checkpoint
from figures import make_all
from models.vbll import self_check
from train import train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    self_check()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading Fashion-MNIST and MNIST", flush=True)
    fashion_train_x, fashion_train_y = load_split(args.data, "fashion", train=True)
    fashion_test = load_split(args.data, "fashion", train=False)
    mnist_test = load_split(args.data, "mnist", train=False)

    for seed in args.seeds:
        for kind in ("map", "vbll"):
            ckpt = out / f"{kind}_seed{seed}.pt"
            print(f"training {kind} seed {seed}", flush=True)
            train_one(
                fashion_train_x,
                fashion_train_y,
                kind=kind,
                seed=seed,
                out_path=ckpt,
                epochs=args.epochs,
            )
            evaluate_checkpoint(
                ckpt,
                fashion_test,
                mnist_test,
                out_json=out / f"{kind}_seed{seed}_eval.json",
            )

    make_all(out)


if __name__ == "__main__":
    main()
