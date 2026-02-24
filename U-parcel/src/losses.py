import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    加权 + Focal 的交叉熵：
        CE * (1-p_t)^γ ，支持 per-class α 权重
    """
    def __init__(
        self,
        alpha: torch.Tensor,     # shape = [C]，最好让均值≈1
        gamma: float = 1.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : [N,C]   parcel_logits
        target : [N]     parcel_label
        """
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.

        logits = logits[valid]
        target = target[valid]

        ce = F.cross_entropy(
            logits, target,
            weight=self.alpha,
            reduction="none"
        )                              # -log(p_t)
        pt = torch.exp(-ce)            # p_t
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()
