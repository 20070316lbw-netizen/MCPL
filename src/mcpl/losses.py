import torch
from torch import nn
from torch.nn import functional as F


class PairwiseRankLoss(nn.Module):
    """单日截面中, 对真实标签较高的股票施加更高分数的配对损失"""

    def forward(self, scores, targets):
        if scores.ndim != 1 or scores.shape != targets.shape:
            raise ValueError("scores and targets must be matching vectors")
        higher = targets[:, None] > targets[None, :]
        differences = (scores[:, None] - scores[None, :])[higher]
        if differences.numel() == 0:
            return scores.sum() * 0
        return F.softplus(-differences).mean()
