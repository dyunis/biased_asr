import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

import models
import dataset
import gender_subset
import decoder
import models_gender

def make_preds_labels(datadir, expdir, save_file):
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

def make_tsne(datadir, expdir, save_file, title, adversarial=False, 
               test=False, mean=True):
    if adversarial:
        model = models_gender.LSTM_gender(num_layers=3)
    else:
        model = models.LSTM(num_layers=3)

    model_file = os.path.join(expdir, 'best.pt')
    model.load_state_dict(torch.load(model_file))

    if test:
        split='eval92'
    else:
        split='dev93'

    gender_dataset = gender_subset.ESPnetGenderBucketDataset(
                         os.path.join(datadir, 
                                      f'dump/test_{split}/deltafalse/data.json'),
                         os.path.join(datadir,
                                      'lang_1char/train_si284_units.txt'),
                         os.path.join(datadir,
                                      f'test_{split}/spk2gender'),
                         num_buckets=10)

#     since pushkar uses whole sequence to predict gender, and that's too much to
#     keep in memory, take a mean over all frame outputs from the model
    embeds = np.zeros((len(gender_dataset), 1024))
    genders = np.zeros(len(gender_dataset), dtype=np.int)
    for i in range(len(gender_dataset)):
        data = gender_dataset[i]
        feat = data['feat'].copy()[None, ...]

        if adversarial:
            y, gen_y, embed = model(torch.tensor(feat))
        else:
            _, embed = model(torch.tensor(feat))

        embed = embed.detach().numpy()[0]
        if mean:
            embeds[i, :] = np.mean(embed, axis=0)
        else:
            embeds[i, :] = embed[-1, :]

        utt = data['utt_id']
        genders[i] = 0 if gender_dataset.utt2gender[utt] == 'f' else 1

    tsne = TSNE(n_components=2, metric='cosine')
    tsne_embeds = tsne.fit_transform(embeds)

    f = tsne_embeds[genders == 0, :]
    m = tsne_embeds[genders == 1, :]
    plt.scatter(f[:, 0], f[:, 1], label='Female')
    plt.scatter(m[:, 0], m[:, 1], label='Male')

    plt.legend(loc='upper right')
    plt.title(f't-SNE of Female and Male embeddings for {title}')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_file)
    plt.clf()

def make_barplots(er, fer, mer, ylabel, ymin, ymax, inc, save_file):
    n_groups = 10

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22

    rects = plt.bar(index, fer, bar_width, label='Female')
    rects_f = plt.bar(index + bar_width, mer, bar_width, label='Male')
    rects_m = plt.bar(index + 2*bar_width, er, bar_width, label='Combined')

    plt.ylabel(ylabel)
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin, ymax, inc))

    plt.xlabel('Experiment')
    xlabels = ['50/50', '20/80', '50/50 utt', '20/80 utt', '50/50 spk',
               '20/80 spk', '50/50 gndr', '20/80 gndr', '50/50 adv',
               '20/80 adv']
    plt.xticks(index + bar_width, xlabels, rotation=45)

    plt.title(f'{ylabel} by experiment and gender')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_file)
    plt.clf()

if __name__=='__main__':
#     dev cer 
    dcer = [0.409, 0.388, 0.380, 0.536, 0.429, 0.404, 0.405, 0.523, 0.368, 0.452]
    dcfer = [0.400, 0.394, 0.363, 0.536, 0.413, 0.416, 0.390, 0.524, 0.361, 0.444]
    dcmer = [0.419, 0.381, 0.397, 0.536, 0.445, 0.391, 0.422, 0.522, 0.375, 0.461]

#     dev wer
    dwer = [0.975, 0.915, 0.888, 1.048, 0.933, 0.892, 0.938, 0.969, 0.884, 0.950]
    dwfer = [0.969, 0.929, 0.886, 1.056, 0.931, 0.897, 0.941, 0.974, 0.870, 0.957]
    dwmer = [0.981, 0.900, 0.900, 1.039, 0.935, 0.886, 0.934, 0.963, 0.899, 0.944]

#     test cer
    tcer = [0.384, 0.358, 0.380, 0.529, 0.450, 0.407, 0.376, 0.504, 0.332, 0.432]
    tcfer = [0.394, 0.384, 0.376, 0.537, 0.436, 0.427, 0.378, 0.535, 0.348, 0.446]
    tcmer = [0.378, 0.342, 0.382, 0.525, 0.458, 0.396, 0.375, 0.487, 0.323, 0.423]

#     test wer
    twer = [0.964, 0.892, 0.905, 1.118, 0.959, 0.909, 0.922, 0.967, 0.853, 0.945]
    twfer = [0.971, 0.900, 0.890, 1.086, 0.942, 0.915, 0.918, 0.966, 0.853, 0.944]
    twmer = [0.960, 0.887, 0.914, 1.137, 0.969, 0.906, 0.924, 0.968, 0.853, 0.946]

#     make_barplots(dcer, dcfer, dcmer, 'Development CER', 0, 0.55, 0.05, 'figs/dev_cer.png')
#     make_barplots(dwer, dwfer, dwmer, 'Development WER', 0.85, 1.06, 0.02, 'figs/dev_wer.png')
#     make_barplots(tcer, tcfer, tcmer, 'Test CER', 0, 0.55, 0.05, 'figs/test_cer.png')
#     make_barplots(twer, twfer, twmer, 'Test WER', 0.84, 1.15, 0.02, 'figs/test_wer.png')

    datadir = '/scratch/asr_tmp/'
    expdir = '/share/data/speech/Data/dyunis/exps/speech_class/5050_1e-4'
#     save_file = 'model_preds.txt'
#     make_preds_labels(datadir, expdir, save_file)

#     expdirs = glob.glob('/share/data/speech/Data/dyunis/exps/speech_class/*')
    expdirs = ['5050_1e-4', '2080_1e-4', '5050_1e4_gender_mepoch45', '2080_1e4_gender_mepoch45']
    expdirs = [os.path.join('/share/data/speech/Data/dyunis/exps/speech_class', path) for path in expdirs]
    
    dset = ''
    adv = ''
    fn_adv = ''
    fn_dset = ''
    for expdir in expdirs:
        if '5050' in expdir:
            dset = '50/50'
            fn_dset = '5050'
        else:
            dset = '20/80'
            fn_dset = '2080'
        
        if 'mepoch' in expdir:
            adv = ' (adversarial)'
            fn_adv = '_adv'
            adversarial = True
        else:
            adv = ''
            fn_adv = ''
            adversarial = False

        make_tsne(datadir, expdir, f'figs/tsne_{fn_dset}{fn_adv}_mean.png',
              f'{dset} dataset{adv}', test=True, adversarial=adversarial)
        make_tsne(datadir, expdir, f'figs/tsne_{fn_dset}{fn_adv}_last.png',
              f'{dset} dataset{adv}', test=True, mean=False, adversarial=adversarial)
