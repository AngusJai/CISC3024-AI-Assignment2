# CISC3024 AI Assignment #2

**Variational Bayesian Last Layers for clothing image classification**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment2_Report.pdf`](./AIAssignment2_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment2

Discriminative VBLL (Harrison, Willes, and Snoek, ICLR 2024) on Fashion-MNIST, compared with a softmax MAP head on the same CNN. MNIST is the out-of-distribution set.

## Layout

```
├── AIAssignment2_Report.pdf
├── report.tex / process_log.md
├── models/          # backbone, diagonal D-VBLL, classifier
├── train.py / evaluate.py / metrics.py / data_utils.py
├── run_all.py / run_ablation.py / figures.py / demo.py
└── outputs/         # figures, metrics JSON, main checkpoints
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

## Main result (two seeds)

| Model | Clean acc. | NLL | ECE |
|-------|----------:|----:|----:|
| Softmax MAP | 0.930 | 0.200 | 0.0109 |
| D-VBLL, KL weight 1/T | 0.928 | 0.200 | 0.0075 |

Paper: Harrison, Willes, and Snoek, *Variational Bayesian Last Layers*, ICLR 2024.
