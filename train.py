import os
import time
import argparse

import numpy as np
import torch
import tqdm

import dataset
import decoder
import utils

# TODO:
# check training results in working recognizer
# WER evaluation
# beam search decoding
# integrate arpa language models

def main(args):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)

    datadir = '/share/data/speech/Data/dyunis/data/wsj_espnet'
    tmpdir = '/scratch/asr_tmp'
    jsons = {'train': 'dump/train_si284/deltafalse/data.json',
             'val': 'dump/test_dev93/deltafalse/data.json',
             'test': 'dump/test_eval92/deltafalse/data.json'}
    tok_file = 'lang_1char/train_si284_units.txt'
    utils.safe_copytree(datadir, tmpdir)
    train(tmpdir, jsons, tok_file)
    utils.safe_rmtree(tmpdir)

class LSTM(torch.nn.Module):
#     51 magic number comes from 50 characters plus blank
    def __init__(self, input_dim=40, hidden_dim=512, out_dim=51, num_layers=1, 
                 bias=False, bidirectional=True):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                                  num_layers=num_layers, bias=bias, 
                                  bidirectional=bidirectional)
        self.num_layers = num_layers
        self.num_dirs = int(bidirectional) + 1
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(self.num_dirs*self.hidden_dim, 
                                      self.out_dim)
        self.classifier = torch.nn.LogSoftmax(dim=-1)

    def forward(self, X):
        '''
        X: (batch, seq_len, F) 
        y: (batch, seq_len, V)
        '''
        # re-init hidden so that changing batch size isn't a problem
        h0 = self.init_hidden(len(X))
        y, self.hidden = self.lstm(X.transpose(0, 1), h0)
#         else:
#             hidden = self.repackage_hidden(self.hidden)
#             y, self.hidden = self.lstm(X.transpose(0, 1), hidden) 

        y = self.linear(y).transpose(0, 1)
        y = self.classifier(y)
        return y

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers*self.num_dirs, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers*self.num_dirs, batch_size, self.hidden_dim))

    # TODO: implement truncated BPTT by repackaging every k steps
    def repackage_hidden(self, hidden):
        '''
        from https://github.com/pytorch/examples/blob/master/word_language_model/main.py 
        Wraps hidden states in new Tensors, to detach them from their history.
        '''
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)

def train(datadir, jsons, tok_file):
    bsize = 16

    train_set = dataset.ESPnetBucketDataset(os.path.join(datadir, jsons['train']),
                                            os.path.join(datadir, tok_file),
                                            bucket_dir='/scratch/asr_tmp/buckets',
                                            num_buckets=10)
    dev_set = dataset.ESPnetBucketDataset(os.path.join(datadir, jsons['val']),
                                          os.path.join(datadir, tok_file),
                                          bucket_dir='/scratch/asr_tmp/buckets',
                                          num_buckets=10)
    bucket_train_loader = torch.utils.data.DataLoader(
                            train_set, 
                            batch_sampler=dataset.BucketBatchSampler(
                                shuffle=True, 
                                batch_size=bsize, 
                                utt2idx=train_set.utt2idx, 
                                buckets=train_set.buckets),
                            collate_fn=dataset.collate)
    bucket_dev_loader = torch.utils.data.DataLoader(
                            dev_set,
                            batch_sampler=dataset.BucketBatchSampler(
                                shuffle=True,
                                batch_size=bsize,
                                utt2idx=dev_set.utt2idx,
                                buckets=dev_set.buckets),
                            collate_fn=dataset.collate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LSTM(num_layers=3)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
    model.to(device)

    n_iter = len(train_set)
    n_epoch = 30
    train_loss = []
    dev_loss = []

    for epoch in range(n_epoch):
        losses = []
        model.train()
        for data in tqdm.tqdm(bucket_train_loader):
            optimizer.zero_grad()
            log_probs = model(data['feat'].cuda())
            loss = ctc_loss(log_probs.transpose(0, 1), data['label'].cuda(), 
                            data['feat_lens'].cuda(), data['label_lens'].cuda())
            loss.backward()

#             gradient clipping does slowdown, value taken from ESPnet
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if torch.sum(torch.isnan(model.linear.weight.grad)) > 0:
                print('Skipping training due to NaN in gradient')
                optimizer.zero_grad()
            optimizer.step()

            losses.append(float(loss))

        train_loss.append(np.mean(losses))
            
        losses = []
        for data in bucket_dev_loader:
            log_probs = model(data['feat'].cuda())
            loss = ctc_loss(log_probs.transpose(0, 1), data['label'].cuda(), 
                            data['feat_lens'].cuda(), data['label_lens'].cuda())
            losses.append(float(loss))
            log_probs = log_probs.cpu().detach().numpy()
            batch_decoded = decoder.batch_greedy_ctc_decode(log_probs, 
                                                            zero_infinity=True)
            for i in range(log_probs.shape[0]):
                batch_dec = batch_decoded[i, :]
                batch_dec = batch_dec[batch_dec != -1]
                decoded = [train_set.idx2tok[idx] for idx in batch_dec]
                print(' '.join(decoded))
        dev_loss.append(np.mean(losses))

    model_dir = os.path.join(datadir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))

# TODO: look at shubham's paper and kaldi 4-gram for language model integration
def recognize(args, datadir, jsons):
    model = LSTM(args)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    for data in tqdm.tqdm(bucket_dev_loader):
        log_probs = model(data['feat'].cuda())
        log_probs = log_probs.cpu().detach().numpy()
        batch_decoded = decoder.batch_greedy_ctc_decode(log_probs, 
                                                        zero_infinity=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='WSJ CTC character ASR model')
    args = parser.parse_args()
    main(args)
