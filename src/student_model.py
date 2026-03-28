"""Stage 1d: Student embedder model.

EfficientNet-B1-BirdSet-XCL backbone with two projection heads:
  - global_head:  global avg pool → Linear(1536)   matches Perch global embedding
  - spatial_head: adaptive pool (5,3) → Linear(1536) per cell  matches Perch spatial

Both heads produce L2-normalized embeddings so cosine similarity to frozen
Perch prototypes works identically at inference.

Usage (import):
    from student_model import StudentEmbedder
    model = StudentEmbedder.from_pretrained()
    global_emb, spatial_emb = model(mel_spec)  # (B,1536), (B,5,3,1536)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EfficientNetModel, EfficientNetConfig


BIRDSET_BACKBONE = "DBD-research-group/EfficientNet-B1-BirdSet-XCL"
PERCH_EMBED_DIM = 1536
SPATIAL_H = 5
SPATIAL_W = 3


class StudentEmbedder(nn.Module):
    """Lightweight student that mimics Perch 2.0's embedding outputs.

    At inference: audio → mel spectrogram → StudentEmbedder → (global, spatial)
    embeddings, then cosine similarity to frozen prototypes — same pipeline
    as the teacher, no UMAP/HDBSCAN needed.
    """

    def __init__(self, backbone: nn.Module, backbone_hidden: int = 1280,
                 embed_dim: int = PERCH_EMBED_DIM,
                 spatial_h: int = SPATIAL_H, spatial_w: int = SPATIAL_W):
        super().__init__()
        self.backbone = backbone
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Global head: pool feature map → 1536-D
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_hidden, embed_dim),
        )

        # Spatial head: pool to (5,3) grid → linear per cell
        # Implemented as a conv to share computation
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_h, spatial_w))
        self.spatial_proj = nn.Linear(backbone_hidden, embed_dim)

    @classmethod
    def from_pretrained(cls, backbone_name: str = BIRDSET_BACKBONE,
                        embed_dim: int = PERCH_EMBED_DIM) -> "StudentEmbedder":
        """Load EfficientNet-B1-BirdSet backbone and attach projection heads."""
        backbone = EfficientNetModel.from_pretrained(backbone_name)
        # EfficientNet-B1 last conv output: 1280 channels
        hidden = backbone.config.hidden_dim
        return cls(backbone, backbone_hidden=hidden, embed_dim=embed_dim)

    def encode(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning embeddings (not normalized).

        Args:
            mel: (B, 1, H, W) mel spectrogram, single-channel

        Returns:
            global_emb: (B, embed_dim)
            spatial_emb: (B, spatial_h, spatial_w, embed_dim)
        """
        # EfficientNet expects 3-channel input — tile single mel channel
        if mel.shape[1] == 1:
            mel = mel.expand(-1, 3, -1, -1)

        # Backbone: returns last_hidden_state (B, C, H', W')
        feat = self.backbone(mel).last_hidden_state  # (B, 1280, H', W')

        # Global embedding
        global_emb = self.global_head(feat)  # (B, 1536)

        # Spatial embedding: pool to (5,3), then project each cell
        spatial_pooled = self.spatial_pool(feat)  # (B, 1280, 5, 3)
        B, C, H, W = spatial_pooled.shape
        # Reshape to (B*H*W, C), project, reshape back
        cells = spatial_pooled.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        cells_proj = self.spatial_proj(cells)                        # (B*H*W, 1536)
        spatial_emb = cells_proj.reshape(B, H, W, -1)               # (B, 5, 3, 1536)

        return global_emb, spatial_emb

    def forward(self, mel: torch.Tensor,
                normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional L2 normalization.

        Returns:
            global_emb: (B, embed_dim)   — L2 normalized if normalize=True
            spatial_emb: (B, H, W, embed_dim) — L2 normalized per cell
        """
        global_emb, spatial_emb = self.encode(mel)

        if normalize:
            global_emb = F.normalize(global_emb, dim=-1)
            spatial_emb = F.normalize(spatial_emb, dim=-1)

        return global_emb, spatial_emb


class DistillationLoss(nn.Module):
    """Multi-target distillation loss for matching Perch embedding outputs.

    L = λ_global * cosine_loss(global_student, global_teacher)
      + λ_spatial * cosine_loss(spatial_student, spatial_teacher)  [mean over cells]
      + λ_logit  * mse_loss(logit_student, logit_teacher)          [optional]

    Cosine loss = mean(1 - cosine_similarity) — optimizes direction not magnitude,
    which is correct since prototype assignment uses cosine similarity.
    """

    def __init__(self, lambda_global: float = 1.0,
                 lambda_spatial: float = 1.0,
                 lambda_logit: float = 0.15,
                 use_logit_loss: bool = True):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_spatial = lambda_spatial
        self.lambda_logit = lambda_logit
        self.use_logit_loss = use_logit_loss

    def forward(self,
                student_global: torch.Tensor,   # (B, D)
                teacher_global: torch.Tensor,   # (B, D)
                student_spatial: torch.Tensor,  # (B, H, W, D)
                teacher_spatial: torch.Tensor,  # (B, H, W, D)
                student_logits: torch.Tensor | None = None,   # (B, top_k)
                teacher_logits: torch.Tensor | None = None,   # (B, top_k)
                ) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, breakdown_dict)."""
        # Global cosine loss
        loss_global = (1.0 - F.cosine_similarity(student_global,
                                                   teacher_global, dim=-1)).mean()

        # Spatial cosine loss — flatten H*W cells, mean over all
        B, H, W, D = student_spatial.shape
        s_flat = student_spatial.reshape(-1, D)
        t_flat = teacher_spatial.reshape(-1, D)
        loss_spatial = (1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()

        total = (self.lambda_global * loss_global
                 + self.lambda_spatial * loss_spatial)

        breakdown = {
            "loss_global": loss_global.item(),
            "loss_spatial": loss_spatial.item(),
        }

        # Optional logit MSE
        if self.use_logit_loss and student_logits is not None and teacher_logits is not None:
            loss_logit = F.mse_loss(student_logits, teacher_logits)
            total = total + self.lambda_logit * loss_logit
            breakdown["loss_logit"] = loss_logit.item()

        breakdown["loss_total"] = total.item()
        return total, breakdown
