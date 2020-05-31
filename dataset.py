import json
import os

import torch
import numpy as np
import kaldiio # kaldiio is faster at reading than kaldi_io
import tqdm

class ESPnetDataset(torch.utils.data.Dataset):
    '''
    a torch dataset reorganizing the json that ESPnet spits out
    '''
    def __init__(self, json_file, tok_file, transform=None):
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.json = json.load(f)['utts']
        self.idx2utt = list(self.json)
        self.utt2idx = {utt: idx for idx, utt in enumerate(self.idx2utt)}
        self.open_ark = None
        self.transform = transform
        self.lens = []

        self.idx2tok = {}
        with open(tok_file, 'r') as f:
            for line in f:
                line = line.split()
                self.idx2tok[int(line[1])] = line[0]

    def __getitem__(self, idx):
        utt_id = self.idx2utt[idx]
        ark_fn, label = (self.json[utt_id]['input'][0]['feat'], 
                         self.json[utt_id]['output'][0]['tokenid'])
        label = [int(s) for s in label.split()]
        
        feat = kaldiio.load_mat(ark_fn)
        if self.transform is not None:
            feat = self.transform(feat)

        sample = {'utt_id': utt_id, 'feat': feat, 'label': label}
        return sample

    def __len__(self):
        return len(self.idx2utt)

class ESPnetBucketDataset(ESPnetDataset):
    '''
    a torch dataset using a version of the dataset bucketed by length
    '''
    def __init__(self, json_file, tok_file, bucket_dir, num_buckets, 
                 transform=None):
        super().__init__(json_file, tok_file, transform)
        utt_ids = self.json.keys()
        feat_lens = {utt_id: self.json[utt_id]['input'][0]['shape'][0] for utt_id in utt_ids}
        self.buckets, self.utt2bucket = bucket_dataset(utt_ids, 
                                                       feat_lens,
                                                       num_buckets)
        self.num_buckets = num_buckets

def bucket_dataset(utt_ids, feat_lens, num_buckets):
    '''
    returns a dictionary mapping bucket idx to a list of utterances and a 
    map from utterance id to bucket idx

    based on ESPnet json
    '''
    buckets = {}
    utt2bucket = {}
    utt_and_len = []
    for utt_id in utt_ids:
        utt_and_len.append((utt_id, feat_lens[utt_id]))
    
    # sort the utterances according to length
    utt_and_len = sorted(utt_and_len, key=lambda tupl: tupl[1])
    utts = [tupl[0] for tupl in utt_and_len] 

    # divide the utterances into buckets
    per_bucket = len(utts) // num_buckets
    for i in range(num_buckets):
        buckets[i] = utts[i * per_bucket:(i + 1) * per_bucket]
    
        for utt in buckets[i]:
            utt2bucket[utt] = i

    # add the remainder to the last bucket
    if num_buckets * per_bucket < len(utts):
        buckets[num_buckets - 1] = buckets[num_buckets - 1] + utts[num_buckets * per_bucket:]

    return buckets, utt2bucket

def save_buckets(buckets, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    for i in buckets.keys():
        fname = os.path.join(savedir, f'bucket_{i}')
        with open(fname, 'w') as f:
            for utt in buckets[i]:
                f.write(utt + '\n')

class BucketBatchSampler(torch.utils.data.Sampler):
    '''
    returns a list of indices to be sampled, whenever possible, we want to 
    construct a permutation each epoch so that batches are sampled from within
    a single bucket (though the order of batches could be arbitrary)

    so we need to first sample a bucket idx, then sample a batch from within 
    the bucket, and continue to do this until without replacement until there 
    are no more things left to sample

    first construct a list of [batch_idx*num_batches, ..., ], then permute this
    list, then sample within a bucket

    permute all the indices in all buckets
    '''
    def __init__(self, shuffle, batch_size, utt2idx, buckets, seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.utt2idx = utt2idx
        self.num_buckets = len(buckets.keys())
        self.bucket2idx = []
        for i in buckets.keys():
            self.bucket2idx.append(np.array([self.utt2idx[utt] for utt in buckets[i]]))

        self.length = sum([len(bucket) for bucket in self.bucket2idx])
        self.init_num_batches()

        self.start_idx = [0 for i in range(self.num_buckets)]

    def init_num_batches(self):
        '''calculate number of batches in each bucket, sum together'''
        num_batches = [len(bucket) for bucket in self.bucket2idx]
        num_batches = [(num // self.batch_size + 1) for num in num_batches] # functools.map

        # populate a list with batch_nums of each bucket
        self.bucket_list = np.array([i for i in range(self.num_buckets) for j in range(num_batches[i])])
    
    def reset_epoch(self):
        '''reset the sampler every epoch, shuffle the batches'''
        self.start_idx = [0 for j in range(self.num_buckets)]
        if self.shuffle:
            self.bucket_list = np.random.permutation(self.bucket_list)
            for i in range(len(self.bucket2idx)):
                self.bucket2idx[i] = np.random.permutation(self.bucket2idx[i])

    def __iter__(self):
        self.reset_epoch()
        for i, bucket in enumerate(self.bucket_list):
            start = self.start_idx[bucket]
            batch_idxs = self.bucket2idx[bucket][start:start+self.batch_size]

            self.start_idx[bucket] = start + self.batch_size

            yield batch_idxs.tolist()

    def __len__(self):
        return len(self.bucket_list)

def collate(minibatch):
    '''
    stacks the features then pads them to equal length
    '''
    idx2utt, feats, labels = [], [], []
    for i in range(len(minibatch)):
        idx2utt.append(minibatch[i]['utt_id'])
        feats.append(torch.tensor(minibatch[i]['feat'])) 
        labels.append(torch.tensor(minibatch[i]['label'], dtype=torch.int32))
    batch = {}
    batch['utt_id'] = idx2utt
    batch['feat'] = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    batch['feat_lens'] = torch.tensor([feat.shape[0] for feat in feats], dtype=torch.int32)
    batch['label'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    batch['label_lens'] = torch.tensor([label.shape[0] for label in labels], dtype=torch.int32)
    return batch

if __name__=='__main__':
    datadir = '/scratch/asr_tmp/'
    json_file = 'dump/train_si284/deltafalse/data.json'
    savedir = '/scratch/asr_tmp/buckets'
    json_file = os.path.join(datadir, json_file)

