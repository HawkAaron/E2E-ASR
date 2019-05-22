import os
import time
import random
import argparse
import logging
import numpy as np
import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F
import kaldi_io
from model import Transducer
from DataLoader import SequentialLoader

parser = argparse.ArgumentParser(description='PyTorch LSTM CTC Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--stdout', default=False, action='store_true', help='log in terminal')
parser.add_argument('--out', type=str, default='exp/rnnt_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str, default='',
                    help='Initial am & pm parameters')
parser.add_argument('--initam', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--gradclip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'args'), 'w') as f:
    f.write(str(args))
if args.stdout: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
else: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

# TODO use config file
model = Transducer(123, 62, 250, 3, args.dropout, bidirectional=args.bi)
for param in model.parameters():
    torch.nn.init.uniform(param, -0.1, 0.1)
if args.init: model.load_state_dict(torch.load(args.init))
if args.initam: model.encoder.load_state_dict(torch.load(args.initam))
if args.cuda: model.cuda()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=.9)

# data set
trainset = SequentialLoader('train', args.batch_size)
devset = SequentialLoader('dev', args.batch_size)

def eval():
    losses = []
    for xs, ys, xlen, ylen in devset:
        xs = Variable(torch.FloatTensor(xs), volatile=True).cuda()
        ys = Variable(torch.LongTensor(ys), volatile=True).cuda()
        xlen = Variable(torch.IntTensor(xlen)); ylen = Variable(torch.IntTensor(ylen))
        model.eval()
        loss = model(xs, ys, xlen, ylen)
        loss = float(loss.data) * len(xlen)
        losses.append(loss)
    return sum(losses) / len(devset)

def train():
    def adjust_learning_rate(optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    def add_noise(x):
        dim = x.shape[-1]
        noise = torch.normal(torch.zeros(dim), 0.075)
        if x.is_cuda: noise = noise.cuda()
        x.data += noise

    prev_loss = 1000
    best_model = None
    lr = args.lr
    for epoch in range(1, args.epochs):
        totloss = 0; losses = []
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(trainset):
            xs = Variable(torch.FloatTensor(xs)).cuda()
            if args.noise: add_noise(xs)
            ys = Variable(torch.LongTensor(ys)).cuda()
            xlen = Variable(torch.IntTensor(xlen)); ylen = Variable(torch.IntTensor(ylen))
            model.train()
            optimizer.zero_grad()
            loss = model(xs, ys, xlen, ylen)
            loss.backward()
            loss = float(loss.data) * len(xlen)
            totloss += loss; losses.append(loss)
            if args.gradclip: grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)
            optimizer.step()


            if i % args.log_interval == 0 and i > 0:
                loss = totloss / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f'%(epoch, i, loss))
                totloss = 0

        losses = sum(losses) / len(trainset)
        val_l = eval()
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.3e'%(
            epoch, time.time()-start_time, losses, val_l, lr
        ))

        if val_l < prev_loss:
            prev_loss = val_l
            best_model = '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format(args.out, epoch, losses, val_l)
            torch.save(model.state_dict(), best_model)
        else:
            torch.save(model.state_dict(), '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.out, epoch, losses, val_l))
            model.load_state_dict(torch.load(best_model))
            if args.cuda: model.cuda()
            if args.schedule:
                lr /= 2
                adjust_learning_rate(optimizer, lr)

if __name__ == '__main__':
    train()
