import torch
import torch.nn as nn


# Support: ['FocalLoss']


class FocalLoss(nn.Module):
    def __init__(self, elementwise = False, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.elementwise = elementwise
        if self.elementwise:
            self.ce = nn.CrossEntropyLoss(reduction = 'none')
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.elementwise:
            return loss
        else:
            return loss.mean()


