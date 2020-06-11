import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch

import models
import dataset
import gender_subset
import decoder

def get_preds_v_labels(datadir, expdir, save_file):
    model = models.LSTM(num_layers=3, hidden_dim=512, bidirectional=True)

    wts = glob.glob(os.path.join(expdir, '*.pt'))

    gender_dataset = gender_subset.ESPnetGenderBucketDataset(
                         os.path.join(datadir, 
                                      'dump/test_dev93/deltafalse/data.json'),
                         os.path.join(datadir,
                                      'lang_1char/train_si284_units.txt'),
                         os.path.join(datadir,
                                      'test_dev93/spk2gender'),
                         num_buckets=10)

    lines = {}
    for wt in wts:
        model.load_state_dict(torch.load(wt))
        device = torch.device('cpu')
        model.to(device)

        counter = [0, 0]
        idxs = [0, 0]
        for i in range(200, len(gender_dataset)):
            if sum(counter) >= 2:
                break
            data = gender_dataset[i]

            if gender_dataset.utt2gender[data['utt_id']] == 'f' and counter[0] == 0:
                idxs[0] = i
                counter[0] = 1
            elif gender_dataset.utt2gender[data['utt_id']] == 'm' and counter[1] == 0:
                idxs[1] = i
                counter[1] = 1
            else:
                continue

        lines[wt] = {}
        lines[wt]['f'] = []
        lines[wt]['m'] = []
        for i, idx in enumerate(idxs):
            data = gender_dataset[idx]
            feat = data['feat'].copy()[None, ...]
            
            log_probs, embed = model(torch.tensor(feat))
            log_probs = log_probs.detach().numpy()
            labels = np.array(data['label'])
            
            preds, to_remove = decoder.batch_greedy_ctc_decode(
                                   log_probs,
                                   zero_infinity=True)
            preds = preds[preds != -1]

            pred_words = decoder.compute_words(preds, gender_dataset.idx2tok)
            label_words = decoder.compute_words(labels, gender_dataset.idx2tok)
            
            gndr = 'f' if i == 0 else 'm'
            lines[wt][gndr].append(f'{gndr}:')
            lines[wt][gndr].append(f'Predicted:\n{" ".join(pred_words)}')
            lines[wt][gndr].append(f'Label:\n{" ".join(label_words)}')

    with open(save_file, 'w') as f:
        for wt in lines.keys():
            f.write(f'{wt}\n')
            for key in lines[wt].keys():
                for line in lines[wt][key]:
                    f.write(line + '\n')
                f.write('\n')
            f.write('\n')

def make_barplots(er, fer, mer, ylabel, ymin, ymax, save_file):
    n_groups = 10

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22

    rects = plt.bar(index, er, bar_width, label='Combined')
    rects_f = plt.bar(index + bar_width, fer, bar_width, label='Female')
    rects_m = plt.bar(index + 2*bar_width, mer, bar_width, label='Male')

    plt.ylabel(ylabel)
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin, ymax, 0.05))

    plt.xlabel('Experiment')
    xlabels = ['50/50', '20/80', '50/50 utt', '20/80 utt', '50/50 spk',
               '20/80 spk', '50/50 gndr', '20/80 gndr', '50/50 discrim',
               '20/80 discrim']
    plt.xticks(index + bar_width, xlabels, rotation=45)

    plt.title(f'{ylabel} by experiment and gender')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_file)
    plt.clf()

if __name__=='__main__':
#     dev cer 
    er = [0.389, 0.491, 0.392, 0.476, 0.387, 0.463, 0.393, 0.486, 0, 0]
    mer = [0.377, 0.482, 0.373, 0.462, 0.370, 0.453, 0.378, 0.473, 0, 0]
    fer = [0.401, 0.501, 0.412, 0.491, 0.406, 0.473, 0.408, 0.498, 0, 0] 

#     dev wer
#     er = [0.900, 0.996, 0.933, 0.979, 0.916, 0.991, 0.899, 1.012, 0, 0]
#     fer = [0.896, 1.007, 0.920, 0.972, 0.903, 0.992, 0.894, 1.031, 0, 0]
#     mer = [0.905, 0.984, 0.946, 0.985, 0.930, 0.990, 0.906, 0.992, 0, 0]

#     test cer

#     test wer

#     make_barplots(er, mer, fer, 'CER', 0, 0.55, 'cer.png')

    datadir = '/scratch/asr_tmp/'
    expdir = '/scratch/asr_tmp/exps/2080_1e-4'
    save_file = 'model_preds.txt'
    get_preds_v_labels(datadir, expdir, save_file)
