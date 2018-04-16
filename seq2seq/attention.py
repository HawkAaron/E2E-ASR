import math
import torch
from torch import nn
import torch.nn.functional as F

class NNAttention(nn.Module):

    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super(NNAttention, self).__init__()
        assert kernel_size % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding)
        self.nn = nn.Sequential(
                     nn.ReLU(),
                     nn.Linear(n_channels, 1))
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None):
        """ `eh` (BTH), `dhx` (BH) """
        pax = eh + dhx.unsqueeze(dim=1) # BTH

        if ax is not None:
            ax = ax.unsqueeze(dim=1) # B1T
            ax = self.conv(ax).transpose(1, 2) # BTH
            pax = pax + ax

        pax = self.nn(pax) # BT1
        pax = pax.squeeze(dim=2)
        if self.log_t:
            log_t = math.log(pax.shape[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1) # BT

        sx = ax.unsqueeze(2) # BT1
        sx = torch.sum(eh * sx, dim=1) # BH
        return sx, ax

class Attention(nn.Module):
    def __init(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def forward(self, hidden, enc_out):
        # NOTE calculated seperately for each batch
        # TODO change bmm to point-wise product
        # BTH * BH
        output = enc_out * hidden[-1].unsqueeze(dim=1) # BTH
        output = output.sum(dim=2) # BT
        # output = torch.bmm(enc_out, hidden[0].unsqueeze(dim=2)).squeeze(dim=2) # NOTE which hidden ?
        output = F.softmax(output, dim=1) # attention alignment
        # BT * BTH
        output = output.unsqueeze(dim=2) * enc_out
        output = output.sum(dim=1) # BH
        return output
        # output = torch.bmm(output.unsqueeze(dim=1), enc_out)
        # return output.squeeze(dim=1)