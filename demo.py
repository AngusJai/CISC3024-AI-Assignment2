"""Print predictions from a saved checkpoint on the first Fashion-MNIST test images."""

import argparse

import torch

from data_utils import CLASS_NAMES, load_split, normalize
from evaluate import load_model
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/vbll_seed0.pt")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    device = get_device()
    model, payload = load_model(args.ckpt, device)
    images, labels = load_split("data", "fashion", train=False)
    images, labels = images[: args.n], labels[: args.n]
    with torch.no_grad():
        probs = model.probabilities(normalize(images.to(device)), n_samples=32).cpu()
    print(f"checkpoint {args.ckpt}  kind={payload['kind']}  seed={payload['seed']}")
    for i in range(args.n):
        pred = int(probs[i].argmax())
        print(
            f"{i:02d}  true={CLASS_NAMES[int(labels[i])]:12s} "
            f"pred={CLASS_NAMES[pred]:12s}  p={float(probs[i, pred]):.3f}"
        )


if __name__ == "__main__":
    main()
