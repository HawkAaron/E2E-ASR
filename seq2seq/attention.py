import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def forward(self, hidden, enc_out):
        # NOTE calculated seperately for each batch
        # TODO change bmm to point-wise product
        # BTH * BH
        output = enc_out * hidden[0] # BTH
        output = output.sum(dim=2) # BT
        # output = torch.bmm(enc_out, hidden[0].unsqueeze(dim=2)).squeeze(dim=2) # NOTE which hidden ?
        output = F.softmax(output, dim=1) # attention alignment
        # BT * BTH
        output = output.unsqueeze(dim=2) * enc_out
        output = output.sum(dim=1) # BH
        return output
        # output = torch.bmm(output.unsqueeze(dim=1), enc_out)
        # return output.squeeze(dim=1)