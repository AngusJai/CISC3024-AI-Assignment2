# AI Assignment #2 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code. A Cursor agent does the search, implementation, experiments, figures, and report draft.  
**Human role:** provide the Moodle brief, point at Assignment 1, and leave the algorithm choice and the runs to the agent.

---

## How I directed the agent

I pasted the Assignment 2 brief in one message and told the agent it was responsible for the whole submission, including questions if it needed them. I did not write Python. Assignment 1 is already a FasterNet digit project, so this assignment had to be a different algorithm and a probability model.

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
