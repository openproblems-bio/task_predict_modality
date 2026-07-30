"""BABEL-style losses: negative binomial reconstruction for RNA, BCE for binarized ATAC,
combined via QuadLoss with a constant (non-warmup) cross-modality weight, matching the
weighting actually used in BABEL's shipped bin/train_model.py (cross_warmup_delay=0,
link_strength=0, i.e. no alignment term, no warmup schedule).
"""

import torch
from torch import nn

_EPS = 1e-8


def negative_binom_loss(y_true, y_pred, theta, mean=True):
    """DCA-style negative binomial negative log-likelihood."""
    theta = torch.clamp(theta, max=1e6)
    t1 = (
        torch.lgamma(theta + _EPS)
        + torch.lgamma(y_true + 1.0)
        - torch.lgamma(y_true + theta + _EPS)
    )
    t2 = (theta + y_true) * torch.log1p(y_pred / (theta + _EPS)) + y_true * (
        torch.log(theta + _EPS) - torch.log(y_pred + _EPS)
    )
    loss = t1 + t2
    return loss.mean() if mean else loss


class NegativeBinomialLoss(nn.Module):
    def forward(self, preds, target):
        mean, theta, _ = preds
        return negative_binom_loss(target, mean, theta)


class BCELoss(nn.Module):
    """Uses only the first decoder output head (the probability/accessibility head)."""

    def __init__(self):
        super().__init__()
        self._bce = nn.BCELoss()

    def forward(self, preds, target):
        prob = preds[0]
        return self._bce(torch.clamp(prob, min=_EPS, max=1.0 - _EPS), target)


class QuadLoss(nn.Module):
    """Combines the four AssymSplicedAutoEncoder paths:
    loss11 (RNA->RNA, NB), loss22 (ATAC->ATAC, BCE),
    loss21 (ATAC->RNA, NB), loss12 (RNA->ATAC, BCE).

    total = loss11 + loss2_weight*loss22 + cross_weight*(loss21 + loss2_weight*loss12)

    No warmup/link terms, matching BABEL's actual shipped training config.
    """

    def __init__(self, loss2_weight=3.0, cross_weight=1.0):
        super().__init__()
        self.loss1 = NegativeBinomialLoss()
        self.loss2 = BCELoss()
        self.loss2_weight = loss2_weight
        self.cross_weight = cross_weight

    def get_component_losses(self, preds11, preds12, preds21, preds22, target1, target2_bin):
        loss11 = self.loss1(preds11, target1)
        loss21 = self.loss1(preds21, target1)
        loss12 = self.loss2(preds12, target2_bin)
        loss22 = self.loss2(preds22, target2_bin)
        return loss11, loss12, loss21, loss22

    def forward(self, preds11, preds12, preds21, preds22, target1, target2_bin):
        loss11, loss12, loss21, loss22 = self.get_component_losses(
            preds11, preds12, preds21, preds22, target1, target2_bin
        )
        same_modality = loss11 + self.loss2_weight * loss22
        cross_modality = loss21 + self.loss2_weight * loss12
        return same_modality + self.cross_weight * cross_modality
