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
from warpctc_pytorch import CTCLoss
import kaldi_io
from model import RNNModel
import tensorboard_logger as tb
from DataLoader import SequentialLoader, TokenAcc

parser = argparse.ArgumentParser(description='PyTorch LSTM CTC Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--out', type=str, default='exp/ctc_lr1e-3',
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
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
tb.configure(args.out)
random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

model = RNNModel(123, 49, 250, 3, args.dropout)
if args.init: model.load_state_dict(torch.load(args.init))
if args.initam: model.encoder.load_state_dict(torch.load(args.initam))
if args.cuda: model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
criterion = CTCLoss()

# data set
trainset = SequentialLoader('train', args.batch_size)
devset = SequentialLoader('dev', args.batch_size)

tri = cvi = 0
def eval():
    global cvi
    losses = []
    tacc = TokenAcc()
    for xs, _, ys, xlen, ylen in devset:
        x = Variable(torch.FloatTensor(xs), volatile=True).cuda()
        y = Variable(torch.IntTensor(ys)); xl = Variable(torch.IntTensor(xlen)); yl = Variable(torch.IntTensor(ylen))
        model.eval()
        out = model(x)[0]
        loss = criterion(out.transpose(0,1).contiguous(), y, xl, yl)
        loss = float(loss.data) * len(xlen) # batch size
        losses.append(loss)
        tacc.update(out.data.cpu().numpy(), xlen, ys)
        tb.log_value('cv_loss', loss/len(xlen), cvi)
        cvi += 1
    return sum(losses) / len(devset), tacc.getAll()

def train():
    def adjust_learning_rate(optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    global tri
    prev_loss = 1000
    best_model = None
    lr = args.lr
    for epoch in range(1, args.epochs):
        totloss = 0; losses = []
        start_time = time.time()
        tacc = TokenAcc()
        for i, (xs, _, ys, xlen, ylen) in enumerate(trainset):
            x = Variable(torch.FloatTensor(xs)).cuda()
            y = Variable(torch.IntTensor(ys)); xl = Variable(torch.IntTensor(xlen)); yl = Variable(torch.IntTensor(ylen))
            model.train()
            optimizer.zero_grad()
            out = model(x)[0]
            loss = criterion(out.transpose(0,1).contiguous(), y, xl, yl)
            loss.backward()
            loss = float(loss.data) * len(xlen) # batch size
            totloss += loss; losses.append(loss)
            tacc.update(out.data.cpu().numpy(), xlen, ys)
            if args.gradclip: grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)
            optimizer.step()

            tb.log_value('train_loss', loss/len(xlen), tri)
            if args.gradclip: tb.log_value('train_grad_norm', grad_norm, tri)
            tri += 1

            if i % args.log_interval == 0 and i > 0:
                loss = totloss / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f, PER %.2f'%(epoch, i, loss, tacc.get()))
                totloss = 0

        losses = sum(losses) / len(trainset)
        val_l, per = eval()
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f, PER %.2f; cv loss %.2f, PER %.2f; lr %.3e'%(
            epoch, time.time()-start_time, losses, tacc.getAll(), val_l, per, lr
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
