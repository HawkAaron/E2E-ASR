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
        self.vocab_size = vocab_size

    def forward(self, inputs, targets):
        '''
        `inputs`: (batch, length, dim)
        `targets`: (batch, length)
        '''
        enc_out, enc_hid = self.encoder(inputs)
        output, loss = self.decoder(targets, enc_out, enc_hid, False)
        return output, loss
    
    def greedy_decode(self, inputs):
        """ only support one sequence """
        enc_out, enc_hid = self.encoder(inputs)

        inputs = torch.autograd.Variable(torch.LongTensor([self.vocab_size-1]), volatile=True)
        if enc_out.is_cuda: inputs = inputs.cuda()
        # hidden = enc_hid[-1]
        hidden = torch.autograd.Variable(torch.zeros((self.decoder.num_layers, 1, enc_hid.shape[-1])))
        if inputs.is_cuda:
            hidden = hidden.cuda()
        hidden[0] = enc_hid[-1] # NOTE only use last layer hidden vector
        y_seq = []
        label = 1; logp = 0
        # target remove beginning 'sos', in my config, 'sos' is the last label
        while label != 0:
            output, hidden = self.decoder._step(inputs, hidden, enc_out)
            output = torch.nn.functional.log_softmax(output, dim=1)
            pred, inputs = output.max(dim=1)
            label = int(inputs.data[0]); logp += float(pred)
            y_seq.append(label)

        print(y_seq, logp)
        return y_seq, -logp
    