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
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden