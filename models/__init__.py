"""Fashion-MNIST classifiers: softmax MAP and discriminative VBLL."""

from models.backbone import SmallCNN
from models.classifier import ImageClassifier
from models.vbll import DiscVBLL

__all__ = ["SmallCNN", "ImageClassifier", "DiscVBLL"]
