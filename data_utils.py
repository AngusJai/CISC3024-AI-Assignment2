"""Fashion-MNIST loaders, normalization, and test corruptions."""

import torch
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST, MNIST

FASHION_MEAN = 0.2860
FASHION_STD = 0.3530

CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CORRUPTIONS = (
    "clean",
    "noise_0.15",
    "noise_0.30",
    "rotate_30",
    "rotate_60",
    "blur",
    "low_contrast",
)


def load_split(root: str, name: str, train: bool):
    """Return float images in [0, 1] with shape [N, 1, 28, 28] and long labels."""
    dataset_cls = FashionMNIST if name == "fashion" else MNIST
    dataset = dataset_cls(root, train=train, download=True)
    images = dataset.data.unsqueeze(1).float().div(255.0)
    labels = dataset.targets.long()
    return images, labels


def train_val_split(images, labels, val_size: int = 6000, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(images.shape[0], generator=generator)
    val_idx, train_idx = order[:val_size], order[val_size:]
    return (
        images[train_idx],
        labels[train_idx],
        images[val_idx],
        labels[val_idx],
    )


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - FASHION_MEAN) / FASHION_STD


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Light augmentation shared by both models. images are in [0, 1]."""
    batch = images.shape[0]
    choice = (torch.rand(batch, device=images.device) < 0.5).view(batch, 1, 1, 1)
    images = torch.where(choice, images.flip(-1), images)
    padded = F.pad(images, (2, 2, 2, 2))
    tops = torch.randint(0, 5, (batch,), device=images.device)
    lefts = torch.randint(0, 5, (batch,), device=images.device)
    rows = tops[:, None, None] + torch.arange(28, device=images.device)[None, :, None]
    cols = lefts[:, None, None] + torch.arange(28, device=images.device)[None, None, :]
    index = torch.arange(batch, device=images.device)[:, None, None]
    return padded[index, 0, rows, cols].unsqueeze(1)


def corrupt(images: torch.Tensor, name: str, seed: int = 0) -> torch.Tensor:
    """Apply a corruption in pixel space. images are [B, 1, 28, 28] in [0, 1]."""
    if name == "clean":
        return images
    if name.startswith("noise_"):
        sigma = float(name.split("_")[1])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(images.shape, generator=generator)
        return (images.cpu() + sigma * noise).clamp(0, 1).to(images.device)
    if name.startswith("rotate_"):
        angle = float(name.split("_")[1])
        return _rotate(images, angle)
    if name == "blur":
        return F.avg_pool2d(images, kernel_size=5, stride=1, padding=2).clamp(0, 1)
    if name == "low_contrast":
        return (0.5 + 0.35 * (images - 0.5)).clamp(0, 1)
    raise ValueError(name)


def _rotate(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    # Affine rotation around the centre. One shared angle for the whole batch.
    theta = torch.tensor(angle_deg * torch.pi / 180.0)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    affine = torch.tensor(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]],
        dtype=images.dtype,
        device=images.device,
    ).unsqueeze(0).repeat(images.shape[0], 1, 1)
    grid = F.affine_grid(affine, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")
