"""Model definitions: MLP classifier, masked focal loss, and ASL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalLoss(nn.Module):
    """Binary focal loss with per-sample, per-class masking.

    Masked positions (mask=0) contribute zero to the loss.
    This handles:
      - XC/iNat: only primary + listed secondary species are supervised
      - Soundscapes: all species are supervised (absence = true negative)
      - Unlabeled: no supervision (all zeros in mask)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits
            targets: (B, C) multi-hot targets
            masks: (B, C) loss masks — 1.0 where loss should be computed
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                  reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(targets >= 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma

        loss = focal_weight * bce * masks

        # Normalize by number of supervised entries (avoid div-by-zero)
        n_supervised = masks.sum().clamp(min=1.0)
        return loss.sum() / n_supervised


class MaskedAsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) with per-sample, per-class masking.

    From "Asymmetric Loss For Multi-Label Classification" (Ridnik et al. 2021).
    Key idea: use different focusing parameters for positives vs negatives,
    and apply hard thresholding on negative probabilities to suppress
    easy negatives entirely. This is critical for multi-label problems
    with heavy negative imbalance (234 species, most absent per segment).

    Args:
        gamma_pos: focusing parameter for positive samples (lower = less down-weighting)
        gamma_neg: focusing parameter for negative samples (higher = more suppression of easy negatives)
        clip: hard threshold for negative probability shifting. Probabilities below
              this value are clamped to zero before computing the loss, eliminating
              gradient signal from very easy negatives.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)

        # Negative probability shifting (hard thresholding)
        prob_neg = prob.clamp(min=self.clip)

        # Positive: -log(p) weighted by (1-p)^gamma_pos
        # Negative: -log(1-p_shifted) weighted by p_shifted^gamma_neg
        pos_loss = -targets * ((1.0 - prob) ** self.gamma_pos) * torch.log(prob.clamp(min=1e-8))
        neg_loss = -(1.0 - targets) * (prob_neg ** self.gamma_neg) * torch.log((1.0 - prob_neg).clamp(min=1e-8))

        loss = (pos_loss + neg_loss) * masks
        n_supervised = masks.sum().clamp(min=1.0)
        return loss.sum() / n_supervised


class MotifClassifier(nn.Module):
    """Multi-label MLP classifier on pre-computed feature vectors.

    Architecture:
        Input(D) → [Linear → BatchNorm → ReLU → Dropout] × L → Linear(num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: list[int] = (512, 256),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, num_classes)."""
        return self.head(self.backbone(x))
