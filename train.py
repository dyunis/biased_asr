import os
import argparse
import logging

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

import dataset
import decoder
import utils
import models
import gender_subset
import transforms

# TODO:
# automatic resuming of experiments based on last model file in model_dir
# beam search decoding
# integrate arpa language models

def main(args):
    jsons = {'train': 'dump/train_si284/deltafalse/data.json',
             'dev': 'dump/test_dev93/deltafalse/data.json',
             'test': 'dump/test_eval92/deltafalse/data.json'}
    spk2genders = {'train': 'train_si284/spk2gender',
                   'dev': 'test_dev93/spk2gender',
                   'test': 'test_eval92/spk2gender'}

    args.bucket_load_dir = 'buckets/5050'

    utils.safe_copytree(args.data_root, args.temp_root)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    logging.basicConfig(filename=os.path.join(args.model_dir, args.log_file),
                        filemode='a', level=logging.INFO)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.determ
    np.random.seed(args.seed)

    if not args.eval_only:
        train(args, jsons, spk2genders)

    evaluate(args, jsons, spk2genders)
    
    if args.cleanup:
        utils.safe_rmtree(args.temp_root)

def train(args, jsons, spk2genders):
    train_set, bucket_train_loader = make_dataset_dataloader(args, jsons,
                                                             spk2genders,
                                                             split='train')

    dev_set, bucket_dev_loader = make_dataset_dataloader(args, jsons,
                                                         spk2genders,
                                                         split='dev')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.LSTM(num_layers=args.n_layers, hidden_dim=args.hidden_dim,
                        bidirectional=args.bidir)
    if args.model_dir is not None:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
                        
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                 betas=(0.9, 0.98))
    model.to(device)

    train_loss = []
    dev_loss, dev_cer, dev_wer = [], [], []
    best_dev_loss = np.inf

    for epoch in range(args.n_epochs):
        losses = []
        model.train()
        for data in tqdm.tqdm(bucket_train_loader):
            optimizer.zero_grad()
            log_probs, embed = model(data['feat'].cuda())
            loss = ctc_loss(log_probs.transpose(0, 1), data['label'].cuda(), 
                            data['feat_lens'].cuda(), data['label_lens'].cuda())
            loss.backward()

#             gradient clipping does slowdown, value taken from ESPnet
            torch.nn.utils.clip_grad_value_(model.parameters(), args.val_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_clip)
            if torch.sum(torch.isnan(model.linear.weight.grad)) > 0:
                logging.info('Skipping training due to NaN in gradient')
                optimizer.zero_grad()
            optimizer.step()

            losses.append(float(loss))

        train_loss.append(np.mean(losses))
            
        losses, cer, wer = [], [], []
        model.eval()
        for data in bucket_dev_loader:
            log_probs, embed = model(data['feat'].cuda())
            loss = ctc_loss(log_probs.transpose(0, 1), data['label'].cuda(), 
                            data['feat_lens'].cuda(), data['label_lens'].cuda())

            losses.append(float(loss))
            batch_cer, batch_wer = decoder.compute_cer_wer(
                                       log_probs.cpu().detach().numpy(), 
                                       data['label'].cpu().detach().numpy(),
                                       train_set.idx2tok)
            cer.extend(batch_cer)
            wer.extend(batch_wer)

        dev_loss.append(np.mean(losses))
        dev_cer.append(np.mean(cer))
        dev_wer.append(np.mean(wer))

#         log a single female and male utterance prediction vs label
        logging.info(f'Epoch: {epoch}')
        logging.info('----')
        logging.info(f'training loss: {train_loss[-1]}')
        logging.info(f'dev loss: {dev_loss[-1]}')
        logging.info(f'dev CER: {dev_cer[-1]}')
        logging.info(f'dev WER: {dev_wer[-1]}')
        logging.info('    ')

        if args.model_dir is not None:
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.model_dir, 
                                                            f'{epoch}.pt'))
            if epoch == args.n_epochs - 1:
                torch.save(model.state_dict(), os.path.join(args.model_dir, 
                                                            f'{epoch}.pt'))

            if dev_loss[-1] < best_dev_loss:
                torch.save(model.state_dict(), os.path.join(args.model_dir, 
                                                            'best.pt'))
                best_dev_loss = dev_loss[-1]

    make_epoch_plot(train_loss, 'train loss', os.path.join(args.temp_root,
                                                           args.model_dir,
                                                           'train_loss.png'))
    make_epoch_plot(dev_loss, 'dev loss', os.path.join(args.temp_root,
                                                       args.model_dir,
                                                       'dev_loss.png'))
    make_epoch_plot(dev_cer, 'dev CER', os.path.join(args.temp_root,
                                                     args.model_dir,
                                                     'dev_cer.png'))
    make_epoch_plot(dev_wer, 'dev WER', os.path.join(args.temp_root,
                                                     args.model_dir,
                                                     'dev_wer.png'))

def make_epoch_plot(x, x_name, save_file):
    plt.plot(np.arange(len(x)), x)
    plt.title(f'{x_name} vs epoch')
    plt.savefig(save_file)
    plt.clf()

# TODO: look at shubham's paper and kaldi 4-gram for language model integration
def evaluate(args, jsons, spk2genders):
    model = models.LSTM(num_layers=args.n_layers, hidden_dim=args.hidden_dim,
                        bidirectional=args.bidir)

    model.load_state_dict(torch.load(os.path.join(args.temp_root, 
                                                  args.model_dir,
                                                  'best.pt')))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    splits = ['train', 'dev', 'test']
    for split in splits:
        evaluate_split(args, jsons, spk2genders, model, split=split)

def make_dataset_dataloader(args, jsons, spk2genders, split='train'):
    if split == 'train':
        load_dir = os.path.join(args.temp_root, args.bucket_load_dir)
    else:
        load_dir = None

    gender_dataset = gender_subset.ESPnetGenderBucketDataset(
                     os.path.join(args.temp_root, jsons[split]),
                     os.path.join(args.temp_root, args.tok_file),
                     os.path.join(args.temp_root, spk2genders[split]),
                     load_dir=load_dir,
                     num_buckets=args.n_buckets)

    if args.normalize == 'utt':
        gender_dataset.transform = transforms.Normalize()
    elif args.normalize == 'spk':
        gender_dataset.transform = transforms.SpeakerNormalize(
                                       os.path.join(args.temp_root,
                                                    args.bucket_load_dir,
                                                    f'{split}_stats/spk2meanstd.pkl'))
    elif args.normalize == 'gndr':
        gender_dataset.transform = transforms.GenderNormalize(
                                       os.path.join(args.temp_root, 
                                                    args.bucket_load_dir, 
                                                    f'{split}_stats/gndr2meanstd.pkl'),
                                       gender_dataset.utt2gender)

    gender_dataloader = torch.utils.data.DataLoader(
                            gender_dataset, 
                            batch_sampler=dataset.BucketBatchSampler(
                                shuffle=True, 
                                batch_size=args.batch_size, 
                                utt2idx=gender_dataset.utt2idx, 
                                buckets=gender_dataset.buckets),
                            collate_fn=dataset.collate)

    return gender_dataset, gender_dataloader


def evaluate_split(args, jsons, spk2genders, model, split='train'):
    gender_dataset, gender_dataloader = make_dataset_dataloader(args, jsons,
                                                                spk2genders,
                                                                split=split)

    stats = evaluate_dataset(model, gender_dataset, gender_dataloader)
    
    logging.info(f'Final results on {split}')
    logging.info('----')
    for key in stats.keys():
        logging.info(f'{key}: {stats[key]}')
    logging.info('    ')

def evaluate_dataset(model, gender_dataset, gender_dataloader):
    cer, wer, m_cer, m_wer, f_cer, f_wer = [], [], [], [], [], []
    model.eval()
    for data in tqdm.tqdm(gender_dataloader):
        log_probs, embed = model(data['feat'].cuda())
        log_probs = log_probs.cpu().detach().numpy()
        labels = data['label'].cpu().detach().numpy()

        idx2gender = np.array([gender_dataset.utt2gender[utt]=='f' 
                               for utt in data['utt_id']])
        m_log_probs = log_probs[idx2gender == 0, :, :]
        m_labels = labels[idx2gender == 0, :]
        f_log_probs = log_probs[idx2gender == 1, :, :]
        f_labels = labels[idx2gender == 1, :]

        f_batch_cer, f_batch_wer = decoder.compute_cer_wer(
                                       f_log_probs,
                                       f_labels,
                                       gender_dataset.idx2tok)
        m_batch_cer, m_batch_wer = decoder.compute_cer_wer(
                                       m_log_probs,
                                       m_labels,
                                       gender_dataset.idx2tok)
        batch_cer, batch_wer = decoder.compute_cer_wer(
                                   log_probs, 
                                   labels,
                                   gender_dataset.idx2tok)

        f_cer.extend(f_batch_cer)
        f_wer.extend(f_batch_wer)
        m_cer.extend(m_batch_cer)
        m_wer.extend(m_batch_wer)
        cer.extend(batch_cer)
        wer.extend(batch_wer)

    stats = {}
    stats['cer'] = np.mean(cer)
    stats['wer'] = np.mean(wer)
    stats['f_cer'] = np.mean(f_cer)
    stats['f_wer'] = np.mean(f_wer)
    stats['m_cer'] = np.mean(m_cer)
    stats['m_wer'] = np.mean(m_wer)

    return stats

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='WSJ CTC character ASR model')

    parser.add_argument('--eval_only',
                        action='store_true',
                        help='skip training step',
                        default=False)
    parser.add_argument('--data_root',
                        type=str,
                        help='data directory',
                        default='/share/data/speech/Data/dyunis/data/wsj_espnet')
    parser.add_argument('--temp_root',
                        type=str,
                        help='temporary data directory',
                        default='/scratch/asr_tmp')
    parser.add_argument('--tok_file', 
                        type=str,
                        help='token file (idx -> token)',
                        default='lang_1char/train_si284_units.txt')
    parser.add_argument('--log_file',
                        type=str,
                        help='file to log to',
                        default='log.log')
    parser.add_argument('--model_dir',
                        type=str,
                        help='directory to save models',
                        default='models')
    parser.add_argument('--bucket_save_dir',
                        type=str,
                        help='directory to save data buckets')
    parser.add_argument('--bucket_load_dir', 
                        type=str,
                        help='directory to load buckets from')
    parser.add_argument('--n_buckets', 
                        type=int, 
                        help='number of buckets to split dataset into',
                        default=10)

    parser.add_argument('--hidden_dim',
                        type=int,
                        help='hidden dimension of the RNN acoustic model',
                        default=512)
    parser.add_argument('--n_layers',
                        type=int,
                        help='number of layers of RNN acoustic model',
                        default=3)
    parser.add_argument('--bidir',
                        type=bool,
                        help='whether acoustic model is bidirectional',
                        default=True)

    parser.add_argument('--n_epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        help='minibatch size for training',
                        default=16)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for training',
                        default=1e-4)
    parser.add_argument('--norm_clip',
                        type=float,
                        help='value to clip gradient norm to',
                        default=10.0)
    parser.add_argument('--val_clip',
                        type=float,
                        help='value to clip gradient values to',
                        default=10.0)
    parser.add_argument('--save_interval',
                        type=int,
                        help='save model every so many epochs',
                        default=10)

    parser.add_argument('--seed',
                        type=int,
                        help='seed for random number generators',
                        default=0)
    parser.add_argument('--cleanup',
                        action='store_true',
                        help='clean up temporary data at the end',
                        default=False)
    parser.add_argument('--determ',
                        type=bool,
                        help='deterministic behavior for cuda',
                        default=True)

    parser.add_argument('--prop_female',
                        type=float,
                        help='proportion of female speakers in dataset',
                        default=0.5)
    parser.add_argument('--n_hrs',
                        type=float,
                        help='number of hours of subsampled data to use',
                        default=5.0)
    parser.add_argument('--normalize',
                        type=str,
                        help='type of normalization [utt, spk, gndr] to use',
                        default=None)

    args = parser.parse_args()
    main(args)
