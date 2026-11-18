"""Joint objective in Section 3.5."""
import torch
from torch import nn
from torch.nn import functional as F


class PromptConsistencyLoss(nn.Module):
    def __init__(self, channels, embedding_dim=128):
        super().__init__()
        if len(channels) < 2:
            raise ValueError('PPC requires at least two prompt levels.')
        self.projections = nn.ModuleList(nn.Linear(c, embedding_dim) for c in channels)

    def forward(self, prompts):
        if len(prompts) != len(self.projections):
            raise ValueError('PPC prompt levels must match the projection layers.')
        z = [layer(prompt.mean(dim=(-2, -1))) for layer, prompt in zip(self.projections, prompts)]
        return torch.stack([1 - F.cosine_similarity(a, b, dim=-1) for a, b in zip(z[:-1], z[1:])]).mean()


def restoration_weight(epoch):
    """Epoch is zero-based: epochs 1--20 use weight 1.0."""
    return max(0., 1. - .1 * (epoch // 20))


def restoration_loss(restored, clean):
    # Undo the inherited input standardization; offsets cancel in the L1 difference.
    std = restored.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return F.l1_loss(restored * std, clean * std, reduction='mean')
