import random
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .attention import Attention, NNAttention

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, sample_rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = NNAttention(hidden_size, log_t=True)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size - 1) # no 'sos'
        self.vocab_size = vocab_size
        self.sample_rate = sample_rate

    def forward(self, target, enc_out, enc_hid):
        '''
        `target`: (batch, length)
        `enc_out`: Encoder output, (batch, length, dim)
        `enc_hid`: last hidden state of encoder
        '''
        target = target.transpose(0, 1)
        length = target.shape[0]
        inputs = target[0] # B
        hidden = enc_hid # BH
        ax = sx = None 
        out = []; align = []
        loss = 0
        # target remove beginning 'sos', in my config, 'sos' is the last label
        for i in range(1, length):
            output, hidden, ax, sx = self._step(inputs, hidden, enc_out, ax, sx)
            out.append(output); align.append(ax)
            if random.random() < self.sample_rate:
                inputs = output.max(dim=1)[1]
            else:
                inputs = target[i]
            # truncated BPTT
            if i % 40 == 0:
                out = torch.cat(out, dim=0)
                out = out.view(-1, out.shape[-1])
                t = target[i-39:i+1].contiguous().view(-1)
                loss = loss + F.cross_entropy(out, t, size_average=False)
                out = []
                loss.backward()
                loss = Variable(loss.data)
                hidden = Variable(hidden.data)
                ax = Variable(ax.data); sx = Variable(sx.data)
        # loss
        left = len(out)
        if left > 0:
            out = torch.cat(out, dim=0)
            out = out.view(-1, out.shape[-1])
            target = target[length-left:].contiguous().view(-1)
            loss = loss + F.cross_entropy(out, target, size_average=False)
            loss.backward()
        # out = torch.cat(out, dim=0)
        # out = out.view(-1, out.shape[-1])
        # target = target[1:].contiguous().view(-1)
        # return F.cross_entropy(out, target, size_average=False)
        return loss
    
    def _step(self, inputs, hidden, enc_out, ax, sx):
        embeded = self.embedding(inputs)
        if sx is not None:
            # last context vector
            embeded = embeded + sx
        out = self.rnn(embeded, hidden)
        sx, ax = self.attention(enc_out, out, ax)
        output = self.fc(out + sx)
        return output, out, ax, sx