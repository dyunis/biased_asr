import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

import models
import dataset
import gender_subset

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

        counter = (0, 0)
        idxs = (0, 0)
        for i in range(len(gender_dataset)):
            if sum(counter) >= 2:
                break
            data = gender_dataset[i]

            if utt2gender[data['utt_id']] == 'f' and counter[0] == 0:
                idxs[0] = i
                counter[0] = 1
            elif utt2gender[data['utt_id']] == 'm' and counter[1] == 0:
                idxs[1] = i
                counter[1] = 1
            else:
                continue

        lines[wt] = {}
        lines[wt]['f'] = []
        lines[wt]['m'] = []
        for i, idx in enumerate(idxs):
            data = gender_dataset[idx]
            log_probs, embed = model(data['feat'])
            log_probs = log_probs.detach().numpy()
            labels = data['label'].detach().numpy()
            
            preds, to_remove = decoder.batch_greedy_ctc_decode(
                                   log_probs,
                                   zero_infinity=True)
            preds = preds[preds != -1]

            pred_words = decoder.compute_words(preds, gender_dataset.idx2char)
            label_words = decoder.compute_words(labels, gender_dataset.idx2char)
            
            gndr = 'f' if i == 0 else 'm'
            lines[wt][gndr].append(f'{gndr}:')
            lines[wt][gndr].append(f'Predicted:\n{pred_words}')
            lines[wt][gndr].append(f'Label:\n{label_words}')

    with open(save_file, 'w') as f:
        for wt in lines.keys():
            f.write(f'{wt}\n')
            for key in lines[wt].keys():
                for line in lines[wt][key]:
                    f.write(line + '\n')
                f.write('\n')
            f.write('\n')

def make_barplots(ylabel, ymin, ymax, save_file):
    n_groups = 10
#     data to plot
    f_er = np.ones(10) * 0.2
    m_er = np.ones(10) * 0.3
    er = np.ones(10) * 0.25

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22

    rects = plt.bar(index, er, bar_width, label='Combined')
    rects_f = plt.bar(index + bar_width, f_er, bar_width, label='Female')
    rects_m = plt.bar(index + 2*bar_width, f_er, bar_width, label='Male')

    plt.ylabel(ylabel)
    plt.ylim(0, ylimit)
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
    make_barplots('CER', 0, 0.5, 'cer.png')
