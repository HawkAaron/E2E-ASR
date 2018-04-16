import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, inputs, hidden=None):
        '''
        `inputs`: (batch, length, input_size)
        `hidden`: Initial hidden state (num_layer, batch_size, hidden_size)
        '''
        x, h = self.rnn(inputs, hidden)
        dim = h.shape[0]
        h = h.sum(dim=0) / dim
        if self.rnn.bidirectional:
            half = x.shape[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]
        return x, h