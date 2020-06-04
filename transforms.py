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
        sample['feat'] = (feat - np.mean(feat, axis=0)) / np.std(feat, axis=0)
        return sample

class SpeakerNormalize(object):
    ''' normalize per speaker '''
    def __init__(self, spk2meanstd_file):
        with open(spk2meanstd_file, 'rb') as f:
            self.spk2meanstd = pickle.load(f)

    def __call__(self, sample):
        mean, std = self.spk2meanstd[sample['utt_id'][:3]]
        sample['feat'] = (sample['feat'] - mean) / std
        return sample

class GenderNormalize(object):
    ''' normalize per gender category '''
    def __init__(self, utt2gender, gender2meanstd_file):
        with open(gender2meanstd_file, 'rb') as f:
            gender2meanstd = pickle.load(f)

        self.utt2meanstd = {utt: gender2meanstd[utt2gender[utt]] for utt in utt2gender.keys()}

    def __call__(self, sample):
        utt, feat = sample['utt_id'], sample['feat']
        mean = self.utt2meanstd[utt][0, :]
        std = self.utt2meanstd[utt][1, :]
        sample['feat'] = (feat - mean) / std
        return sample

def compute_spk_mean_std(dataset, save_file=None):
    ''' computes the mean and standard dev for features for each speaker '''
    spk2count = {}
    spk2sum = {}
    spk2sqsum = {}
    for sample in tqdm.tqdm(dataset):
        spk, feat = sample['utt_id'][:3], sample['feat']
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
    for sample in tqdm.tqdm(dataset):
        utt, feat = sample['utt_id'], sample['feat']
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
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    datadir = '/scratch/asr_tmp/'
    json_file = 'dump/train_si284/deltafalse/data.json'
    tok_file = 'lang_1char/train_si284_units.txt'
    spk2gender_file = 'train_si284/spk2gender'
    balanced_set = gender_subset.ESPnetGenderBucketDataset(
                    os.path.join(datadir, json_file),
                    os.path.join(datadir, tok_file),
                    spk2gender_file=os.path.join(datadir, spk2gender_file),
                    load_dir='/scratch/asr_tmp/buckets/5050',
                    num_buckets=10)
#     spk2meanstd = compute_spk_mean_std(balanced_set, save_file='/scratch/asr_tmp/buckets/5050_spk2meanstd.pkl')
    gender2meanstd = compute_gender_mean_std(balanced_set, save_file='/scratch/asr_tmp/buckets/5050_gndr2meanstd.pkl')
#     spk_norm = SpeakerNormalize('/scratch/asr_tmp/buckets/5050_spk2meanstd.pkl')
    gender_norm = GenderNormalize(balanced_set.utt2gender,
                                  '/scratch/asr_tmp/buckets/5050_gndr2meanstd.pkl')
