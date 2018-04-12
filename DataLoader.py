import sys
import numpy as np
import kaldi_io

with open('data/lang/phones.txt', 'r') as f:
    phone = {}; rephone = {}
    for line in f:
        line = line.split()
        phone[line[0]] = int(line[1])
        rephone[int(line[1])] = line[0]
print(phone)

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat

def convert(inputs, labels):
    xlen = [i.shape[0] for i in inputs]
    ylen = [i.shape[0] for i in labels]
    xs = zero_pad_concat(inputs)
    # TODO move to transducer model forward
    ymat = np.concatenate((np.zeros((len(labels), 1)), zero_pad_concat(labels)), axis=1).astype(np.int32)
    ys = np.hstack(labels)
    return xs, ymat, ys, xlen, ylen


class NpyLoader:
    def __init__(self, dtype, batch_size=1):
        self.label = []
        self.feat = []
        self.batch_size = batch_size
        with open('data/'+dtype+'/text', 'r') as f:
            ids = [line.split()[0] for line in f]
        for i in ids:
            with open('data-npy/'+i+'.x', 'rb') as f:
                self.feat.append(np.load(f))
            with open('data-npy/'+i+'.y', 'rb') as f:
                self.label.append(np.load(f))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        for i in range(0, len(self.label), self.batch_size):
            end = i + self.batch_size
            if end > len(self.label): end = len(self.label)
            yield convert(self.feat[i:end], self.label[i:end])

class SequentialLoader:
    def __init__(self, dtype, batch_size=1):
        self.labels = {}
        self.feats_rspecifier = 'ark:copy-feats scp:data/{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/{}/utt2spk scp:data/{}/cmvn.scp ark:- ark:- |\
 add-deltas --delta-order=2 ark:- ark:- | nnet-forward data/final.feature_transform ark:- ark:- |'.format(dtype, dtype, dtype)
        self.batch_size = batch_size
        # load label
        with open('data/'+dtype+'/text', 'r') as f:
            for line in f:
                line = line.split()
                self.labels[line[0]] = np.array([phone[i] for i in line[1:]])
        # load feature

    def __len__(self):
        return len(self.labels)

    def _dump(self):
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            label = self.labels[k]
            with open('data-npy/'+k+'.y', 'wb') as f:
                np.save(f, label)
            with open('data-npy/'+k+'.x', 'wb') as f:
                np.save(f, v)
            print(k)

    def __iter__(self):
        feats = []; label = []
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            if len(feats) >= self.batch_size:
                yield convert(feats, label)
                feats = []; label = []
            feats.append(v); label.append(self.labels[k])
        yield convert(feats, label)

import editdistance
class TokenAcc():
    def __init__(self, blank=0):
        self.err = 0
        self.cnt = 0
        self.tmp_err = 0
        self.tmp_cnt = 0
        self.blank = 0
    
    def update(self, pred, xlen, label):
        ''' label is one dimensinal '''
        pred = np.vstack([pred[i, :j] for i, j in enumerate(xlen)])
        e = self._distance(pred, label)
        c = label.shape[0]
        self.tmp_err += e; self.err += e
        self.tmp_cnt += c; self.cnt += c
        return 100 * e / c

    def get(self, err=True):
        # get interval
        if err: res = 100 * self.tmp_err / self.tmp_cnt
        else: res = 100 - 100 * self.tmp_err / self.tmp_cnt
        self.tmp_err = self.tmp_cnt = 0
        return res

    def getAll(self, err=True):
        if err: return 100 * self.err / self.cnt
        else: return 100 - 100 * self.err / self.cnt

    def _distance(self, y, t):
        if len(y.shape) > 1: 
            y = np.argmax(y, axis=1)
        prev = self.blank
        hyp = []
        for i in y:
            if i != self.blank and i != prev: hyp.append(i)
            prev = i
        return editdistance.eval(hyp, t)

class KaldiDataLoader:
    def __init__(self, labels_file, feats_rspecifier, batch_size, training=True):
        self.labels_file = labels_file
        self.feats_rspecifier = feats_rspecifier
        self.batch_size = batch_size
        self.training = training
        self.labels = {}
        self.randomizer_size = 1048576
        self.feats_dim = 123
        self.read_labels()
        
    def read_labels(self):
        with open(self.labels_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.labels[line[0]] = np.array([int(x) for x in line[1:]])

    def batch(self):
        self.trailing_labels = np.zeros((0), dtype=np.int64)
        self.trailing_feats = np.zeros((0, self.feats_dim), dtype=np.float32)
        self.randomizer_labels = np.zeros((self.randomizer_size), dtype=np.int64)
        self.randomizer_feats = np.zeros((self.randomizer_size, self.feats_dim), dtype=np.float32)

        if self.training:
            size = 0
            for key, feats in kaldi_io.read_mat_ark(self.feats_rspecifier):
                if key not in self.labels:
                    continue
                labels = self.labels[key]

                if self.trailing_feats.shape[0] > 0:
                    self.randomizer_feats[0:size] = self.trailing_feats
                    self.randomizer_labels[0:size] = self.trailing_labels
                    self.trailing_labels = np.zeros((0), dtype=np.int64)
                    self.trailing_feats = np.zeros((0, self.feats_dim), dtype=np.float32)
                    
                if size + feats.shape[0] < self.randomizer_size:
                    self.randomizer_feats[size:size + feats.shape[0]] = feats
                    self.randomizer_labels[size:size + feats.shape[0]] = labels
                    size = size + feats.shape[0]
                else:
                    self.randomizer_feats[size:self.randomizer_size] = feats[0:self.randomizer_size - size]
                    self.randomizer_labels[size:self.randomizer_size] = labels[0:self.randomizer_size - size]
                    self.trailing_feats = feats[self.randomizer_size - size:]
                    self.trailing_labels = labels[self.randomizer_size - size:]
                    size = size + feats.shape[0] - self.randomizer_size

                    permutation = np.random.permutation(self.randomizer_size)
                    for i in range(self.randomizer_size / self.batch_size):
                        yield self.randomizer_feats[permutation[self.batch_size * i:self.batch_size * (i + 1)]], self.randomizer_labels[permutation[self.batch_size * i:self.batch_size * (i + 1)]]
            permutation = np.random.permutation(size)
            for i in range(size / self.batch_size):
                yield self.randomizer_feats[permutation[self.batch_size * i:self.batch_size * (i + 1)]], self.randomizer_labels[permutation[self.batch_size * i:self.batch_size * (i + 1)]]
        else:
            for key, feats in kaldi_io.read_mat_ark(self.feats_rspecifier):
                if key not in self.labels:
                    continue
                labels = self.labels[key]
                yield feats, labels

if __name__ == '__main__':
    SequentialLoader('train')._dump()
    SequentialLoader('dev')._dump()