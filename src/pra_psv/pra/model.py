"""PRA model and DH objective, extracted without scientific changes."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

class PrefixFeatureAdapter(nn.Module):
    """PRA head; ~1–4M trainable parameters depending on backbone width."""

    def __init__(self, input_dim: int, width: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_dim, self.width = input_dim, width
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim + 1),
            nn.Linear(input_dim + 1, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
        length = (prefix_lengths.float() / 16).unsqueeze(-1)
        return self.net(torch.cat((features.float(), length), dim=-1)).squeeze(-1)

@dataclass
class FeatureTask:
    features: torch.Tensor
    prefix_lengths: torch.Tensor
    harmful: bool
    teacher: torch.Tensor
    teacher_mask: torch.Tensor

def task_loss(model: PrefixFeatureAdapter, tasks: list[FeatureTask], *,
              kd_weight: float = 0.3, positive_floor: float = 0.7,
              benign_max_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Every prefix participates, with equal total weight per task.

    Positive teacher scores are blended with the hard target rather than discarded;
    the floor prevents an uncertain early teacher verdict erasing timely detection.
    """
    losses, kd_count = [], 0
    for task in tasks:
        logits = model(task.features, task.prefix_lengths)
        hard = torch.ones_like(logits) if task.harmful else torch.zeros_like(logits)
        loss = F.binary_cross_entropy_with_logits(logits, hard)
        mask = task.teacher_mask.bool()
        if kd_weight and mask.any():
            soft = task.teacher[mask].clamp(0, 1)
            if task.harmful:
                soft = positive_floor + (1 - positive_floor) * soft
            loss = loss + kd_weight * F.binary_cross_entropy_with_logits(logits[mask], soft)
            kd_count += int(mask.sum())
        # Both training and official evaluation use an any-prefix decision.
        task_logit = logits.max()
        loss = loss + F.binary_cross_entropy_with_logits(task_logit, hard[0])
        if not task.harmful:
            loss = loss + benign_max_weight * F.softplus(task_logit)
        losses.append(loss)
    if not losses:
        raise ValueError("empty task batch")
    return torch.stack(losses).mean(), {"teacher_prefixes": kd_count, "tasks": len(tasks)}
