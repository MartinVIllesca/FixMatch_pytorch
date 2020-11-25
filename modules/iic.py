import torch
import torch.nn as nn
from numpy import log


class IIC(nn.Module):
    def __init__(self, c=10):
        super(IIC, self).__init__()

        self.C = c
        self.EPS = 1e-9

    def forward(self, z, zt):
        P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        P = ((P + P.t()) / 2) / P.sum()
        P[(P < self.EPS).data] = self.EPS
        Pi = P.sum(dim=1).view(self.C, 1).expand(self.C, self.C)
        Pj = P.sum(dim=0).view(1, self.C).expand(self.C, self.C)
        return (P * (log(Pi) + log(Pj) - log(P))).sum()
