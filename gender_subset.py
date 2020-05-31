import json
import os
import random

import numpy as np

import dataset

class ESPnetGenderBucketDataset(dataset.ESPnetDataset):
    '''
    same as above, but with gender subsetting
    '''
    def __init__(self, json_file, tok_file, bucket_dir, num_buckets, 
                 utt2gender_file, gender_frac=[0.5, 0.5], hrs=5, 
                 transform=None):
        super().__init__(json_file, tok_file, transform)
        utt_ids = self.json.keys()
        feat_lens = {utt_id: self.json[utt_id]['input'][0]['shape'][0] for utt_id in utt_ids}
        new_utt_ids = gender_subset.gender_subset(utt_ids, feat_lens, 
                                                  utt2gender_file,
                                                  gender_frac, hrs)
        self.buckets, self.utt2bucket = dataset.bucket_dataset(utt_ids, 
                                                               feat_lens,
                                                               num_buckets)
        self.num_buckets = num_buckets

def gender_subset(utt_ids, feat_lens, spk2gender_fn, gender_frac, hrs):
    '''
    take a list of utterance ids, a file for speaker to gender map, a split of
    the fraction of data across gender classes, and the total amount of data in
    hours

    returns a list of utterance ids to be used as the new training set
    '''
    assert np.sum(gender_frac) == 1.0, 'Fraction of each gender must total 1'

    spk2gender = {}
    with open(spk2gender_fn, 'r') as f:
        for line in f:
            line = line.split()
            spk2gender[line[0]] = line[1]
        
    gender_hrs = hrs * np.array(gender_frac)
    gender_frames = 16000 * gender_hrs
    gender_frames = [floor(num) for num in gender_frames]
    gender_frames = {'f': gender_frames[0], 'm': gender_frames[1]} 
    utt2gender = {utt: spk2gender[utt[:2]] for utt in utts}

#     permute list of utterances (random.shuffle is in-place)
    utt_ids = random.sample(utt_ids, len(utt_ids))

#     add utterances to each list until (near) correct number of hours in each
    new_utt_ids = []
    idx = 0
    while gender_frames['f'] > 0 and gender_frames['m'] > 0:
        gender = utt2gender[utt_ids[idx]]
        if gender_frames[gender] <= 0:
            continue

        new_utt_ids.append(utt_ids[idx])
        gender_frames[gender] -= feat_lens[utt_ids[idx]]
        idx += 1

    return new_utt_ids

if __name__=='__main__':
    # write test code that actually checks the total number of hours is approx
    # correct, and that the split is approx correct
    pass
