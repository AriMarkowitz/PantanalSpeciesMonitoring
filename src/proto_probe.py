"""Prototypical probing head (Bird-MAE, arXiv:2504.12880).

Replaces the HDBSCAN + GMM + MLP stack with a single learned prototype
head trained directly on frozen Student spatial embeddings.

Architecture (per class c):
  - J learnable prototypes p_{c,j} ∈ R^D           (D = 1536, Student embedding dim)
  - For a segment with L spatial tokens z_1..z_L ∈ R^D:
      sim_{c,j} = max_l cos(z_l, p_{c,j})           (max-pool over tokens)
  - Logit: y_c = Σ_j w_{c,j} · sim_{c,j} + b_c       with w_{c,j} ≥ 0
  - BCE loss on multi-hot labels + orthogonality regulariser on prototypes
    within each class to prevent collapse.

Shapes:
  - spatial embedding batch:  (B, L, D)
  - prototypes:               (C, J, D)
  - similarity tensor:        (B, C, J)   after max-pool over L
  - logits:                   (B, C)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int,
                 prototypes_per_class: int = 10):
        super().__init__()
        self.D = embed_dim
        self.C = num_classes
        self.J = prototypes_per_class

        # Prototypes initialised with small Gaussian (will be L2-normalised at use)
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, prototypes_per_class, embed_dim) * 0.02
        )
        # Non-negative combination weights: parameterise via softplus to enforce w ≥ 0
        # without post-step clamping (keeps gradients smooth).
        self.raw_weights = nn.Parameter(
            torch.full((num_classes, prototypes_per_class), -2.0)
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def weights(self) -> torch.Tensor:
        """Positive per-class combination weights, shape (C, J)."""
        return F.softplus(self.raw_weights)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        """Compute per-class logits from spatial embeddings.

        Args:
            spatial: (B, L, D) float tensor of L spatial tokens per segment.

        Returns:
            logits: (B, C)
        """
        # Normalise tokens and prototypes for cosine similarity
        z = F.normalize(spatial, dim=-1)                    # (B, L, D)
        p = F.normalize(self.prototypes, dim=-1)            # (C, J, D)

        # Cosine sim: (B, L, D) × (C, J, D) → (B, L, C, J) via einsum
        sim = torch.einsum("bld,cjd->blcj", z, p)           # (B, L, C, J)

        # Max-pool over L spatial tokens → (B, C, J)
        pooled = sim.max(dim=1).values

        # Weighted sum over J prototypes → (B, C)
        logits = (pooled * self.weights.unsqueeze(0)).sum(dim=-1) + self.bias
        return logits

    def orthogonality_loss(self) -> torch.Tensor:
        """Off-diagonal cosine-similarity penalty within each class's prototypes.

        Encourages each class's J prototypes to spread out instead of collapsing
        onto a single direction. Computed as the squared off-diagonal entries
        of P P^T where P is the L2-normalised prototype matrix per class.
        """
        p = F.normalize(self.prototypes, dim=-1)            # (C, J, D)
        gram = torch.einsum("cjd,ckd->cjk", p, p)           # (C, J, J)
        eye = torch.eye(self.J, device=p.device).unsqueeze(0)  # (1, J, J)
        off = gram - eye
        return (off ** 2).mean()


class MaskedBCEWithOrthoLoss(nn.Module):
    """BCE on masked multi-hot targets + orthogonality regulariser.

    Masks follow the existing Pantanal convention:
      mask[i, c] = 1 means "class c is supervised for sample i"
      mask[i, c] = 0 means "unknown — exclude from loss"
    """

    def __init__(self, ortho_weight: float = 0.1):
        super().__init__()
        self.ortho_weight = ortho_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor, head: PrototypicalHead) -> torch.Tensor:
        per_el = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        denom = masks.sum().clamp_min(1.0)
        bce = (per_el * masks).sum() / denom
        ortho = head.orthogonality_loss()
        return bce + self.ortho_weight * ortho
