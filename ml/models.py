"""
models.py — Networks and losses for the two training tracks.

Track A — from scratch:
    A compact ResNet trained on face crops with an ArcFace margin head.
    Demonstrates the full metric-learning pipeline end to end.

Track B — head on frozen dlib embeddings:
    A small residual MLP that re-projects the existing 128-d dlib vectors.
    dlib's backbone already saw ~3M faces; LFW alone cannot compete with that,
    so the useful move is to keep its features and learn a better metric
    space on top of them.

Why ArcFace rather than a plain softmax classifier: a classifier only has to
separate the identities it was trained on, and nothing forces the embedding to
generalise to strangers. ArcFace enforces an angular margin between classes on
the unit hypersphere, which directly optimises the cosine/Euclidean geometry
that the matcher uses at query time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Losses / margin heads
# ----------------------------------------------------------------------

class ArcMarginProduct(nn.Module):
    """
    ArcFace: additive angular margin softmax (Deng et al., 2019).

    Produces logits of the form s·cos(θ + m) for the true class and s·cos(θ)
    for the rest, where θ is the angle between the L2-normalised embedding and
    the L2-normalised class centre.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 32.0, margin: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_normal_(self.weight)

        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        # Beyond this angle, cos(θ+m) stops decreasing; fall back to a linear
        # penalty so gradients stay well-behaved for very hard samples.
        self._threshold = math.cos(math.pi - margin)
        self._mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        phi = cosine * self._cos_m - sine * self._sin_m          # cos(θ + m)
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)


# ----------------------------------------------------------------------
# Track A — backbone trained from scratch
# ----------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class FaceNetSmall(nn.Module):
    """
    A compact ResNet for 112x112 face crops.

    Deliberately small (~1.5M parameters). With only a few thousand training
    images, a larger backbone memorises the training identities instead of
    learning a transferable metric.
    """

    def __init__(self, embedding_dim: int = 128, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.PReLU(width),
        )
        self.layer1 = self._stage(width, width, blocks=2, stride=2)          # 56
        self.layer2 = self._stage(width, width * 2, blocks=2, stride=2)      # 28
        self.layer3 = self._stage(width * 2, width * 4, blocks=2, stride=2)  # 14
        self.layer4 = self._stage(width * 4, width * 8, blocks=2, stride=2)  # 7

        self.head = nn.Sequential(
            nn.BatchNorm2d(width * 8),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(width * 8 * 7 * 7, embedding_dim),
            # BatchNorm on the output stabilises the embedding scale before
            # normalisation, as in the ArcFace reference implementation.
            nn.BatchNorm1d(embedding_dim),
        )

    @staticmethod
    def _stage(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        layers += [BasicBlock(out_ch, out_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ----------------------------------------------------------------------
# Track B — head over frozen dlib embeddings
# ----------------------------------------------------------------------

class EmbeddingHead(nn.Module):
    """
    Residual MLP that re-projects a frozen 128-d dlib embedding.

    The residual connection matters: it starts the model at (near) the
    identity function, so training can only improve on dlib's own geometry
    rather than having to rediscover it. With a small dataset that is the
    difference between a useful head and a destructive one.
    """

    def __init__(self, in_dim: int = 128, hidden: int = 512, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.out_norm = nn.BatchNorm1d(out_dim)
        # Start as a no-op so the head begins from dlib's own space.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.skip(x) + self.body(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
