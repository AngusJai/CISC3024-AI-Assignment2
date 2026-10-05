# AI Assignment #2 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code. A Cursor agent does the search, implementation, experiments, figures, and report draft.  
**Human role:** provide the Moodle brief, point at Assignment 1, and leave the algorithm choice and the runs to the agent.

---

## How I directed the agent

I told the agent to find a recent Bayes classifier for computer vision, not to reuse FasterNet, to check the loss on a toy batch, and to write the code, experiments, and report. I did not write Python.

## How the algorithm was chosen

The agent searched recent Bayesian classifiers and compared three:

- Variational Bayesian Last Layers, Harrison, Willes, and Snoek, ICLR 2024.
- FSP-Laplace, Cinquin, Pförtner, Fortuin, Hennig, and Bamler, NeurIPS 2024.
- Last Layer Empirical Bayes, Villecroze et al., 2025.

It kept discriminative VBLL. The training loss is a closed-form lower bound (Theorem 2), which can be checked on a random batch. The other two add a function-space Laplace approximation or a normalizing-flow prior. The task is Fashion-MNIST clothing classification, with MNIST digits as a far out-of-distribution set, so the write-up does not repeat the Assignment 1 digit-robustness story.

## What was implemented

- `models/vbll.py`: diagonal Gaussian posterior on each class row, Theorem 2 Jensen bound, KL against an isotropic prior with variance `2/D`, inverse-Wishart-style noise penalty, Monte Carlo softmax predictive.
- `models/backbone.py`: shared CNN, 128-d features, about 158k parameters.
- `models/classifier.py`: softmax MAP versus D-VBLL. Weight decay is on the backbone only for VBLL.
- `self_check()` before training: finite loss, bound matches a hand-expanded log-sum-exp, larger features increase logit variance.
- The public `vbll` package was not imported. The loss is written out so it can be read against the paper.

## Experiments

Command:

```bash
python run_all.py --epochs 12 --seeds 0 1
python run_ablation.py
```

Device: Apple MPS, PyTorch 2.14. Fashion-MNIST split 54k/6k/10k. AdamW, cosine schedule, 12 epochs, batch 128. Test predictive distribution uses 32 logit samples.

Clean test, mean of two seeds:

| Model | Acc | NLL | ECE |
|-------|----:|----:|----:|
| Softmax MAP | 0.930 | 0.200 | 0.0109 |
| D-VBLL (KL weight 1/T) | 0.928 | 0.200 | 0.0075 |

MNIST AUROC (max predictive probability) was 0.902 and 0.808 for MAP, and 0.825 and 0.855 for D-VBLL. The seed gap is larger than the model gap.

Learned weight variance at `1/T` is about `4.9e-4`, versus a prior variance of `0.0156`. Entropy falls as the feature norm grows. Low-contrast test images stay confidently wrong (ECE above 0.65).

KL ablation, seed 0:

| KL weight | Acc | NLL | ECE | Entropy | MNIST AUROC | Mean weight var |
|----------:|----:|----:|----:|--------:|-------------:|----------------:|
| 1/T | 0.929 | 0.201 | 0.0075 | 0.187 | 0.825 | 4.9e-4 |
| 10/T | 0.927 | 0.212 | 0.0154 | 0.280 | 0.869 | 4.7e-3 |
| 100/T | 0.927 | 0.289 | 0.0937 | 0.639 | 0.916 | 8.1e-3 |

Shirt is the weak class (about 0.77). The main confusions are shirt / T-shirt / coat.

## Controls added after the first full runs

`python extra_analysis.py` does not retrain the CNNs.

- Temperature scaling on the MAP validation logits: $T = 1.114$ and $1.088$. Test ECE $0.0050$ and $0.0067$, below D-VBLL. The ECE win over raw softmax is mostly a temperature.
- Gaussian naive Bayes on pixels: accuracy $0.579$, NLL $7.63$.
- Shrinkage LDA on seed-0 MAP features: accuracy $0.917$, NLL $0.544$, ECE $0.063$.
- Misclassification AUROC about $0.923$ for both heads. Accuracy at $90\%$ coverage about $0.968$ (MAP) and $0.967$ (D-VBLL). Risk-coverage curves overlap.

## Further runs

`python run_upgrades.py` then `python upgrade_analysis.py`.

- MAP and D-VBLL seed 2. Three-seed clean means: MAP accuracy 0.931, NLL 0.199, ECE 0.0104, MNIST AUROC 0.851. D-VBLL accuracy 0.928, NLL 0.201, ECE 0.0074, AUROC 0.856.
- G-VBLL seeds 0 and 1. Accuracy 0.928, NLL 0.206, ECE 0.0131, entropy 0.252, MNIST AUROC 0.910 (standard deviation 0.003).
- D-VBLL KL weights 10/T and 100/T, seed 1, combined with the earlier seed-0 ablation. Mean AUROC is 0.844 at 10/T and 0.913 at 100/T. The seed-0-only rise at 10/T did not repeat.
- Prior scales 0.1 and 10, D-VBLL seed 0. Posterior variance stays between 2.5e-4 and 5.4e-4. AUROC falls as the prior widens (0.883, 0.825, 0.791).
- Monte Carlo sizes 8, 32, and 64 agree to about 0.001 on accuracy and NLL.
- Shirt-only ECE is about 0.07 for MAP and D-VBLL and 0.049 for G-VBLL. About 10% of shirts are called T-shirts.
- Keeping the most confident 90% does not raise accuracy under 60 degree rotation or low contrast.
- Temperature on MAP seed 2 is T = 1.119, ECE 0.0046. Three-seed mean ECE is 0.0054.

## Report revision after feedback

The report states the task in English: a recent Bayes classifier, not FasterNet, a checkable loss, and a clothing task with a softmax baseline. A second instruction asks for the third seed, G-VBLL, the KL and prior sweeps, Monte Carlo size, shirt calibration, and rejection under corruption.

## Report

`report.tex` is compiled to `AIAssignment2_Report.pdf`. It uses the six sections required by the brief.

## Submission checklist

- [x] Recent probability / Bayes classifier (D-VBLL, ICLR 2024)
- [x] Computer-vision task (Fashion-MNIST, MNIST OOD, corruptions)
- [x] Search, code, experiments, and report draft done by the agent
- [x] No student-written code
- [x] English report with the six required sections
- [x] Name: CHE CHI HIN, Angus / ID: UC325182
- [x] Source: https://github.com/AngusJai/CISC3024-AI-Assignment2
- [ ] Upload the PDF to UMMoodle
