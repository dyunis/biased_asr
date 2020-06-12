import os
import pickle

import numpy as np
import tqdm

import gender_subset

'''
file for on-the-fly data transforms like normalization
'''

class Normalize(object):
    ''' normalize per utterance '''
    def __init__(self):
        pass
    
    def __call__(self, sample):
        feat = sample['feat']
        sample['feat'] = ((feat - np.mean(feat, axis=0)) 
                           / np.std(feat, axis=0)).astype(np.float32)
        return sample

class SpeakerNormalize(object):
    ''' normalize per speaker '''
    def __init__(self, spk2meanstd_file):
        with open(spk2meanstd_file, 'rb') as f:
            self.spk2meanstd = pickle.load(f)

    def __call__(self, sample):
        mean, std = self.spk2meanstd[sample['utt_id'][:3]]
        sample['feat'] = ((sample['feat'] - mean) / std).astype(np.float32)
        return sample

class GenderNormalize(object):
    ''' normalize per gender category '''
    def __init__(self, gender2meanstd_file, utt2gender):
        with open(gender2meanstd_file, 'rb') as f:
            gender2meanstd = pickle.load(f)

        self.utt2meanstd = {utt: gender2meanstd[utt2gender[utt]] 
                            for utt in utt2gender.keys()}

    def __call__(self, sample):
        utt, feat = sample['utt_id'], sample['feat']
        mean, std = self.utt2meanstd[utt]
        sample['feat'] = ((feat - mean) / std).astype(np.float32)
        return sample

def compute_spk_mean_std(dataset, save_file=None):
    ''' computes the mean and standard dev for features for each speaker '''
    spk2count = {}
    spk2sum = {}
    spk2sqsum = {}
    for bucket in dataset.buckets.keys():
        for utt in tqdm.tqdm(dataset.buckets[bucket]):
            idx = dataset.utt2idx[utt]
            feat = dataset[idx]['feat']
            spk = utt[:3]
            if spk not in spk2sum.keys():
                spk2count[spk] = feat.shape[0]
                spk2sum[spk] = np.sum(feat, axis=0)
                spk2sqsum[spk] = np.sum(feat ** 2, axis=0)
            else:
                spk2count[spk] += feat.shape[0]
                spk2sum[spk] += np.sum(feat, axis=0)
                spk2sqsum[spk] += np.sum(feat ** 2, axis=0)

    spk2meanstd = {}
    for spk in spk2sum.keys():
        mean = spk2sum[spk] / spk2count[spk]
        std = np.sqrt(spk2sqsum[spk] / spk2count[spk] - (mean ** 2))
        spk2meanstd[spk] = np.stack((mean, std), axis=0)

    if save_file is not None:
        safe_pickle(spk2meanstd, save_file)

    return spk2meanstd

def compute_gender_mean_std(dataset, save_file=None):
    ''' computes the mean and standard dev for features within a gender '''
    gender2count = {}
    gender2sum = {}
    gender2sqsum = {}
    for bucket in tqdm.tqdm(dataset.buckets.keys()):
        for utt in dataset.buckets[bucket]:
            idx = dataset.utt2idx[utt]
            feat = dataset[idx]['feat']
            gender = dataset.utt2gender[utt]
            if gender not in gender2count.keys():
                gender2count[gender] = feat.shape[0]
                gender2sum[gender] = np.sum(feat, axis=0)
                gender2sqsum[gender] = np.sum(feat ** 2, axis=0)
            else:
                gender2count[gender] += feat.shape[0]
                gender2sum[gender] += np.sum(feat, axis=0)
                gender2sqsum[gender] += np.sum(feat ** 2, axis=0)
                
    gender2meanstd = {}
    for gender in gender2count.keys():
        mean = gender2sum[gender] / gender2count[gender]
        std = np.sqrt(gender2sqsum[gender] / gender2count[gender] - (mean ** 2))
        gender2meanstd[gender] = np.stack((mean, std), axis=0)

    if save_file is not None:
        safe_pickle(gender2meanstd, save_file)

    return gender2meanstd

def safe_pickle(obj, save_file):
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    if os.path.exists(save_file):
        raise OSError(f'Path to save {save_file} already exists')

    with open(save_file, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

if __name__=='__main__':
    datadir = '/share/data/speech/Data/dyunis/data/wsj_espnet'
    tok_file = 'lang_1char/train_si284_units.txt'
    load_dir = 'buckets/5050'

    jsons = {'train': 'dump/train_si284/deltafalse/data.json',
             'dev': 'dump/test_dev93/deltafalse/data.json',
             'test': 'dump/test_eval92/deltafalse/data.json'}
    spk2genders = {'train': 'train_si284/spk2gender',
                   'dev': 'test_dev93/spk2gender',
                   'test': 'test_eval92/spk2gender'}

    for split in jsons.keys():
        if split == 'train':
            gender_dataset = gender_subset.ESPnetGenderBucketDataset(
                             os.path.join(datadir, jsons[split]),
                             os.path.join(datadir, tok_file),
                             spk2gender_file=os.path.join(datadir, spk2genders[split]),
                             load_dir=os.path.join(datadir, load_dir),
                             num_buckets=10)
        else:
            gender_dataset = gender_subset.ESPnetGenderBucketDataset(
                             os.path.join(datadir, jsons[split]),
                             os.path.join(datadir, tok_file),
                             spk2gender_file=os.path.join(datadir, spk2genders[split]),
                             num_buckets=10)
            

        spk2meanstd = compute_spk_mean_std(
                        gender_dataset,
                        save_file=os.path.join(datadir, load_dir,
                                               f'{split}_stats/spk2meanstd.pkl'))

        gender2meanstd = compute_gender_mean_std(
                            gender_dataset, 
                            save_file=os.path.join(datadir, load_dir,
                                                   f'{split}_stats/gndr2meanstd.pkl'))

#     spk_norm = SpeakerNormalize('/scratch/asr_tmp/buckets/5050_spk2meanstd.pkl')
#     gender_norm = GenderNormalize(
#                       '/scratch/asr_tmp/buckets/5050/stats/gndr2meanstd.pkl',
#                       balanced_set.utt2gender)
