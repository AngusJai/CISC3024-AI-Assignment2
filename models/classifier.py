"""Image classifier with either a softmax head or a VBLL head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SmallCNN
from models.vbll import DiscVBLL


class ImageClassifier(nn.Module):
    def __init__(
        self,
        kind: str,
        n_train: int,
        feat_dim: int = 128,
        n_classes: int = 10,
        reg_scale: float = 1.0,
    ):
        super().__init__()
        if kind not in {"map", "vbll"}:
            raise ValueError(f"unknown kind {kind}")
        self.kind = kind
        self.n_classes = n_classes
        self.reg_scale = float(reg_scale)
        self.backbone = SmallCNN(in_channels=1, feat_dim=feat_dim)
        if kind == "vbll":
            self.head = DiscVBLL(
                feat_dim,
                n_classes,
                regularization_weight=self.reg_scale / float(n_train),
            )
        else:
            self.head = nn.Linear(feat_dim, n_classes)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def training_step(self, images: torch.Tensor, labels: torch.Tensor):
        """One forward pass. Prediction uses mean logits so batch-norm is not updated twice."""
        feats = self.features(images)
        if self.kind == "vbll":
            loss = self.head.loss(feats, labels)
            with torch.no_grad():
                pred = self.head.logit_moments(feats)[0].argmax(dim=-1)
            return loss, pred
        logits = self.head(feats)
        return F.cross_entropy(logits, labels), logits.argmax(dim=-1)

    def probabilities(self, images: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
        feats = self.features(images)
        if self.kind == "vbll":
            return self.head.predictive(feats, n_samples=n_samples)
        return torch.softmax(self.head(feats), dim=-1)

    def optimizer_groups(self, weight_decay: float):
        if self.kind == "map":
            return [{"params": self.parameters(), "weight_decay": weight_decay}]
        return [
            {"params": self.backbone.parameters(), "weight_decay": weight_decay},
            {"params": self.head.parameters(), "weight_decay": 0.0},
        ]
