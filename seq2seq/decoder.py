import torch
from torch import nn
import torch.nn.functional as F
from .attention import Attention

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention()
        self.rnn = nn.GRU(hidden_size*2, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size - 1) # no 'sos'
        self.vocab_size = vocab_size
        self.loss = nn.CrossEntropyLoss()
        self.num_layers = num_layers

    def forward(self, target, enc_out, enc_hid, teaching=False):
        '''
        `target`: (batch, length)
        `enc_out`: Encoder output, (batch, length, dim)
        `enc_hid`: last hidden state of encoder
        '''
        target = target.transpose(0, 1)
        length, batch_size = target.shape
        result = torch.zeros((length - 1, batch_size, self.vocab_size - 1)) # NOTE liuqi error
        if target.is_cuda:
            result = result.cuda()
        inputs = target[0]
        # hidden = enc_hid[-1]
        hidden = torch.autograd.Variable(torch.zeros((self.num_layers, batch_size, enc_hid.shape[-1])))
        if target.is_cuda:
            hidden = hidden.cuda()
        hidden[0] = enc_hid[-1] # NOTE only use last layer hidden vector
        loss = 0
        # target remove beginning 'sos', in my config, 'sos' is the last label
        for i in range(1, length):
            output, hidden = self._step(inputs, hidden, enc_out)
            result[i - 1] = output.data # NOTE liuqi
            if teaching:
                inputs = target[i]
            else:
                inputs = output.max(dim=1)[1]
            loss += self.loss(output, target[i]) # NOTE Liuqi, doesn't matter, but for difference length, should mask

        return result.transpose(0, 1), loss
    
    def _step(self, inputs, hidden, enc_out):
        embeded = self.embedding(inputs)
        att = self.attention(hidden, enc_out)
        output, hidden = self.rnn(torch.cat((embeded, att), dim=1).unsqueeze(dim=1), hidden)
        output = self.output(output.squeeze(dim=1))
        return output, hidden