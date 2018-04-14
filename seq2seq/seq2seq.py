import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class Seq2seq(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        if bidirectional: hidden_size *= 2
        self.decoder = Decoder(vocab_size, hidden_size, 1)

    def forward(self, inputs, targets):
        '''
        `inputs`: (batch, length, dim)
        `targets`: (batch, length)
        '''
        enc_out, enc_hid = self.encoder(inputs)
        output, loss = self.decoder(targets, enc_out, enc_hid, False)
        return output, loss