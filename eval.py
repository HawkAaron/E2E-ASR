import argparse
import logging
import math
import os
import time

import editdistance
import kaldi_io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from model import Transducer, RNNModel
from DataLoader import SequentialLoader, TokenAcc, rephone

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('model', help='trained model filename')
parser.add_argument('--beam', type=int, default=0, help='apply beam search, beam width')
parser.add_argument('--ctc', default=False, action='store_true', help='decode CTC acoustic model')
parser.add_argument('--bi', default=False, action='store_true', help='bidirectional LSTM')
parser.add_argument('--dataset', default='test', help='decoding data set')
parser.add_argument('--out', type=str, default='', help='decoded result output dir')
args = parser.parse_args()

logdir = args.out if args.out else os.path.dirname(args.model) + '/decode.log'
# if args.out: os.makedirs(args.out, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=logdir, level=logging.INFO)

# Load model
Model = RNNModel if args.ctc else Transducer
model = Model(123, 62, 250, 3, bidirectional=args.bi)
model.load_state_dict(torch.load(args.model, map_location='cpu'))

use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()

# data set
feat = 'ark:copy-feats scp:data/{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/{}/utt2spk scp:data/{}/cmvn.scp ark:- ark:- |\
 add-deltas --delta-order=2 ark:- ark:- | nnet-forward data/final.feature_transform ark:- ark:- |'.format(args.dataset, args.dataset, args.dataset)
with open('data/'+args.dataset+'/text', 'r') as f:
    label = {}
    for line in f:
        line = line.split()
        label[line[0]] = line[1:]

# Phone map
with open('conf/phones.60-48-39.map', 'r') as f:
    pmap = {rephone[0]: rephone[0]}
    for line in f:
        line = line.split()
        if len(line) < 3: pmap[line[0]] = rephone[0]
        else: pmap[line[0]] = line[2]
print(pmap)

def distance(y, t, blank=rephone[0]):
    def remap(y, blank):
        prev = blank
        seq = []
        for i in y:
            if i != blank and i != prev: seq.append(i)
            prev = i
        return seq
    y = remap(y, blank)
    t = remap(t, blank)
    return y, t, editdistance.eval(y, t)

model.eval()
def decode():
    logging.info('Decoding transduction model:')
    err = cnt = 0
    for k, v in kaldi_io.read_mat_ark(feat):
        xs = Variable(torch.FloatTensor(v[None, ...]), volatile=True)
        if use_gpu:
            xs = xs.cuda()
        if args.beam > 0:
            y, nll = model.beam_search(xs, args.beam)
        else:
            y, nll = model.greedy_decode(xs)
        y = [pmap[rephone[i]] for i in y]
        t = [pmap[i] for i in label[k]]
        y, t, e = distance(y, t)
        err += e; cnt += len(t)
        logging.info('[{}]: {}'.format(k, ' '.join(t)))
        logging.info('[{}]: {}\nlog-likelihood: {:.2f}\n'.format(k, ' '.join(y), nll))
    logging.info('{} set {} PER {:.2f}%\n'.format(
        args.dataset.capitalize(), 'CTC' if args.ctc else 'Transducer', 100*err/cnt))

decode()
