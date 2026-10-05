"""Diagonal discriminative variational Bayesian last layer.

Harrison, Willes, and Snoek, Variational Bayesian Last Layers, ICLR 2024.

The last-layer weight row for class k has a Gaussian variational posterior.
Theorem 2 gives a sampling-free lower bound on the expected log-softmax.
Prediction averages softmax probabilities over logit samples (Equation 19).
Diagonal covariances keep the cost linear in the feature width.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def diagonal_gaussian_kl(mean: torch.Tensor, var: torch.Tensor, prior_var: float) -> torch.Tensor:
    """KL of independent N(mean, var) rows against N(0, prior_var I)."""
    width = mean.shape[-1]
    ratio = var / prior_var
    kl_rows = 0.5 * (
        ratio.sum(dim=-1)
        + (mean.square() / prior_var).sum(dim=-1)
        - width
        - ratio.log().sum(dim=-1)
    )
    return kl_rows.sum()


class DiscVBLL(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        regularization_weight: float,
        prior_scale: float = 1.0,
        wishart_scale: float = 1.0,
        dof: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regularization_weight = float(regularization_weight)
        # Authors scale the isotropic prior variance like a Kaiming fan-in term.
        self.prior_var = float(prior_scale) * (2.0 / in_features)
        self.wishart_scale = float(wishart_scale)
        self.wishart_dof = (float(dof) + out_features + 1.0) / 2.0

        bound = math.sqrt(2.0 / in_features)
        self.w_mean = nn.Parameter(torch.randn(out_features, in_features) * bound)
        # Stored value is log standard deviation. Init std ≈ 1 / in_features.
        self.w_logstd = nn.Parameter(
            1e-3 * torch.randn(out_features, in_features) - math.log(in_features)
        )
        self.noise_logstd = nn.Parameter(torch.randn(out_features) - 1.0)

    def weight_variance(self) -> torch.Tensor:
        return torch.exp(2.0 * self.w_logstd)

    def noise_variance(self) -> torch.Tensor:
        return torch.exp(2.0 * self.noise_logstd)

    def logit_moments(self, features: torch.Tensor):
        """Return Gaussian logit mean and variance, shape [B, C] each."""
        mean = features @ self.w_mean.t()
        quad = features.square() @ self.weight_variance().t()
        return mean, quad + self.noise_variance().unsqueeze(0)

    def expected_loglik(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-example Theorem 2 bound. Higher is better."""
        mean, var = self.logit_moments(features)
        chosen = mean[torch.arange(features.shape[0], device=features.device), y]
        lse = torch.logsumexp(mean + 0.5 * var, dim=-1)
        return chosen - lse

    def kl_weight(self) -> torch.Tensor:
        return diagonal_gaussian_kl(self.w_mean, self.weight_variance(), self.prior_var)

    def wishart_penalty(self) -> torch.Tensor:
        """Inverse-Wishart style log prior on diagonal logit noise (Equation 16)."""
        var = self.noise_variance()
        logdet_precision = -var.log().sum()
        trace_precision = (1.0 / var).sum()
        return self.wishart_dof * logdet_precision - 0.5 * self.wishart_scale * trace_precision

    def loss(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative ELBO estimate. Minimize this."""
        elbo = self.expected_loglik(features, y).mean()
        elbo = elbo + self.regularization_weight * (self.wishart_penalty() - self.kl_weight())
        return -elbo

    def predictive(self, features: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
        """Monte Carlo estimate of the posterior predictive class probabilities."""
        mean, var = self.logit_moments(features)
        std = var.clamp_min(1e-12).sqrt()
        eps = torch.randn(
            (n_samples,) + mean.shape, device=mean.device, dtype=mean.dtype
        )
        probs = torch.softmax(mean.unsqueeze(0) + eps * std.unsqueeze(0), dim=-1).mean(0)
        probs = probs.clamp_min(1e-8)
        return probs / probs.sum(dim=-1, keepdim=True)

    def stats(self) -> dict:
        with torch.no_grad():
            return {
                "mean_weight_var": float(self.weight_variance().mean()),
                "mean_noise_var": float(self.noise_variance().mean()),
                "kl": float(self.kl_weight()),
            }


def self_check() -> None:
    torch.manual_seed(0)
    layer = DiscVBLL(4, 3, regularization_weight=1.0 / 100.0)
    features = torch.randn(8, 4)
    labels = torch.randint(0, 3, (8,))
    loss = layer.loss(features, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert layer.w_mean.grad is not None and torch.isfinite(layer.w_mean.grad).all()

    with torch.no_grad():
        mean, var = layer.logit_moments(features)
        manual = mean[torch.arange(8), labels] - torch.logsumexp(mean + 0.5 * var, dim=-1)
        assert torch.allclose(manual, layer.expected_loglik(features, labels))
        _, var_far = layer.logit_moments(features * 4)
        assert float(var_far.mean()) > float(var.mean())
        probs = layer.predictive(features, n_samples=16)
        assert probs.shape == (8, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
    print(f"vbll self-check ok  loss={float(loss.detach()):.4f}")


if __name__ == "__main__":
    self_check()
