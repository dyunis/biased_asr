import json
import os
import random

import numpy as np

import dataset

class ESPnetGenderBucketDataset(dataset.ESPnetDataset):
    '''
    same as above, but with gender subsetting
    '''
    def __init__(self, json_file, tok_file, spk2gender_file, load_dir=None, 
                 save_dir=None, num_buckets=10, gender_frac=[0.5, 0.5], hrs=5, 
                 transform=None):
        super().__init__(json_file, tok_file, transform)
        utt_ids = self.json.keys()
        feat_lens = {utt_id: self.json[utt_id]['input'][0]['shape'][0] for utt_id in utt_ids}
        if load_dir is not None:
            self.buckets, self.utt2bucket = dataset.load_buckets(load_dir, 
                                                                 num_buckets)
        else:
            new_utt_ids, self.utt2gender = gender_subset(utt_ids, feat_lens, 
                                                      spk2gender_file,
                                                      gender_frac, hrs)
            self.buckets, self.utt2bucket = dataset.bucket_dataset(new_utt_ids, 
                                                                   feat_lens,
                                                                   num_buckets)
            if save_dir is not None:
                dataset.save_buckets(save_dir, self.buckets)

        self.num_buckets = num_buckets

def gender_subset(utt_ids, feat_lens, spk2gender_fn, gender_frac=(0.5, 0.5), 
                  hrs=5):
    '''
    take a list of utterance ids, a file for speaker to gender map, a tuple of
    the fraction of data across gender classes, and the total amount of data in
    hours

    returns a list of utterance ids to be used as the new training set
    '''
    assert np.isclose(np.sum(gender_frac), 1.0), 'gender_frac must add up to 1'

    spk2gender = {}
    with open(spk2gender_fn, 'r') as f:
        for line in f:
            line = line.split()
            spk2gender[line[0]] = line[1]
        
    gender_s = {'f': 3600 * hrs * gender_frac[0], 'm': 3600 * hrs * gender_frac[1]}
    utt2gender = {utt: spk2gender[utt[:3]] for utt in utt_ids}

#     permute list of utterances (random.shuffle is in-place)
    utt_ids = random.sample(utt_ids, len(utt_ids))

#     add utterances to each list until (near) correct number of hours in each
    new_utt_ids = []
    idx = 0
    while gender_s['f'] > 0 and gender_s['m'] > 0:
        gender = utt2gender[utt_ids[idx]]
        if gender_s[gender] <= 0:
            continue

        new_utt_ids.append(utt_ids[idx])
        gender_s[gender] -= seconds_per_utt(feat_lens[utt_ids[idx]])
        idx += 1

    return new_utt_ids, utt2gender

def seconds_per_utt(length):
    # I'm not sure where in the ESPnet code these numbers are set, but I am sure
    # they are the right numbers 
    #
    # n_fft = 400, n_shift = 160 in espnet/utils/compute-fbank-feats.py
    frame_len = 0.025 # taken empirically by comparing utterance length to the frame number
    frame_shift = 0.01 # same logic as above

    return (length - 1) * frame_shift + frame_len

if __name__=='__main__':
    datadir = '/scratch/asr_tmp/'
    json_file = 'dump/train_si284/deltafalse/data.json'
    tok_file = 'lang_1char/train_si284_units.txt'
    spk2gender_file = 'train_si284/spk2gender'
    balanced_set = ESPnetGenderBucketDataset(os.path.join(datadir, json_file),
                                             os.path.join(datadir, tok_file),
                                             spk2gender_file=os.path.join(datadir, spk2gender_file),
#                                              save_dir='/scratch/asr_tmp/buckets',
                                             load_dir='/scratch/asr_tmp/buckets',
                                             num_buckets=10,
                                             gender_frac=(0.5, 0.5),
                                             hrs=5)

