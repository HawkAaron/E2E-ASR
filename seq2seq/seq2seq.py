import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class Seq2seq(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout, bidirectional, sample_rate=.4, **kwargs):
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(vocab_size, hidden_size, sample_rate)
        self.vocab_size = vocab_size

    def forward(self, inputs, targets):
        '''
        `inputs`: (batch, length, dim)
        `targets`: (batch, length)
        '''
        enc_out, enc_hid = self.encoder(inputs)
        loss = self.decoder(targets, enc_out, enc_hid)
        return loss
    
    def greedy_decode(self, inputs):
        """ only support one sequence """
        enc_out, enc_hid = self.encoder(inputs)
        # '<sos>'
        inputs = torch.autograd.Variable(torch.LongTensor([self.vocab_size-1]), volatile=True)
        if enc_out.is_cuda: inputs = inputs.cuda()
        hidden = enc_hid
        y_seq = []
        label = 1; logp = 0; ax = sx = None
        # target remove beginning 'sos', in my config, 'sos' is the last label
        while label != 0:
            output, hidden, ax, sx = self.decoder._step(inputs, hidden, enc_out, ax, sx)
            output = torch.nn.functional.log_softmax(output, dim=1)
            pred, inputs = output.max(dim=1)
            label = int(inputs.data[0]); logp += float(pred)
            y_seq.append(label)
        print(y_seq, logp)
        return y_seq, -logp

    def decode_step(self, x, y, state=None, softmax=False):
        """ `x` (TH), `y` (1) """
        if state is None:
            hx, ax, sx = None, None, None
        else:
            hx, ax, sx = state
        out, hx, ax, sx = self.decoder._step(y, hx, x, ax, sx)
        if softmax:
            out = nn.functional.log_softmax(out, dim=1)
        return out, (hx, ax, sx)

    def beam_search(self, xs, beam_size=10, max_len=200):
        start_tok = self.vocab_size - 1; end_tok = 0
        x, h = self.encode(xs)
        y = torch.autograd.Variable(torch.LongTensor([start_tok]), volatile=True)
        beam = [((start_tok,), 0, (h, None, None))];
        complete = []
        for _ in range(max_len):
            new_beam = []
            for hyp, score, state in beam:

                y[0] = hyp[-1]
                out, state = self.decode_step(x, y, state=state, softmax=True)
                out = out.cpu().data.numpy().squeeze(axis=0).tolist()
                for i, p in enumerate(out):
                    new_score = score + p
                    new_hyp = hyp + (i,)
                    new_beam.append((new_hyp, new_score, state))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

            # Remove complete hypotheses
            for cand in new_beam[:beam_size]:
                if cand[0][-1] == end_tok:
                    complete.append(cand)

            beam = filter(lambda x : x[0][-1] != end_tok, new_beam)
            beam = beam[:beam_size]

            if len(beam) == 0:
                break

            # Stopping criteria:
            # complete contains beam_size more probable
            # candidates than anything left in the beam
            if sum(c[1] > beam[0][1] for c in complete) >= beam_size:
                break

        complete = sorted(complete, key=lambda x: x[1], reverse=True)
        if len(complete) == 0:
            complete = beam
        hyp, score, _ = complete[0]
        return hyp, score
    