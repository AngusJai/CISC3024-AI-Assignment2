# CISC3024 AI Assignment #2

**Variational Bayesian Last Layers for clothing image classification**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment2_Report.pdf`](./AIAssignment2_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment2

Discriminative and generative VBLL (Harrison, Willes, and Snoek, ICLR 2024) on Fashion-MNIST, compared with a softmax MAP head on the same CNN. MNIST is the out-of-distribution set.

## Layout

```
├── AIAssignment2_Report.pdf
├── report.tex / process_log.md
├── models/          # backbone, D-VBLL, G-VBLL, classifier
├── train.py / evaluate.py / metrics.py / data_utils.py
├── run_all.py / run_ablation.py / run_upgrades.py / figures.py / demo.py
└── outputs/         # figures, metrics JSON, checkpoints
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --epochs 12 --seeds 0 1
python run_ablation.py
python demo.py --ckpt outputs/vbll_seed0.pt
```

`data/` is downloaded by torchvision and is gitignored.

## Main result

MAP, D-VBLL, and G-VBLL are means over seeds 0--2. Temperature is fit on the validation set of each MAP seed.

| Model | Clean acc. | NLL | ECE | MNIST AUROC |
|-------|----------:|----:|----:|------------:|
| Softmax MAP | 0.931 | 0.199 | 0.0104 | 0.851 |
| MAP + validation temperature | 0.931 | 0.197 | 0.0054 | — |
| D-VBLL, KL weight 1/T | 0.928 | 0.201 | 0.0074 | 0.856 |
| G-VBLL | 0.929 | 0.205 | 0.0139 | 0.910 |

Pixel Gaussian naive Bayes reaches accuracy 0.579. Shrinkage LDA on the MAP features reaches 0.917 accuracy but NLL 0.544. `extra_analysis.py` reproduces the controls.

Paper: Harrison, Willes, and Snoek, *Variational Bayesian Last Layers*, ICLR 2024.
