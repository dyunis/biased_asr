import os
import argparse

import numpy as np
import torch
import tqdm

import dataset
import decoder
import utils
import models

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

    model = models.LSTM(num_layers=3)
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
    model = models.LSTM(args)
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
