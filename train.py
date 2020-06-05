import os
import argparse

import numpy as np
import torch
import tqdm

import dataset
import decoder
import utils
import models
import gender_subset
import transforms

# TODO:
# avg WER evaluation per gender
# plotting and logging of loss, CER, WER
# automatic resuming of experiments based on last model file in model_dir
# beam search decoding
# integrate arpa language models

def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.determ
    np.random.seed(args.seed)

    jsons = {'train': 'dump/train_si284/deltafalse/data.json',
             'val': 'dump/test_dev93/deltafalse/data.json',
             'test': 'dump/test_eval92/deltafalse/data.json'}
    args.bucket_load_dir = 'buckets/5050'


    utils.safe_copytree(args.data_root, args.temp_root)
    train(args, jsons)
    
    if args.cleanup:
        utils.safe_rmtree(args.temp_root)

def train(args, jsons):
    spk_normalize = transforms.SpeakerNormalize(
                        os.path.join(args.temp_root, args.bucket_load_dir, 
                                     'stats/spk2meanstd.pkl'))
    utt_normalize = transforms.Normalize()
                                
    train_set = gender_subset.ESPnetGenderBucketDataset(
                    os.path.join(args.temp_root, jsons['train']),
                    os.path.join(args.temp_root, args.tok_file),
                    os.path.join(args.temp_root, args.spk2gender_file),
                    load_dir=os.path.join(args.temp_root, args.bucket_load_dir),
                    transform=None,
                    num_buckets=args.n_buckets)
    gndr_normalize = transforms.GenderNormalize(
                         os.path.join(args.temp_root, args.bucket_load_dir, 
                                      'stats/gndr2meanstd.pkl'),
                         train_set.utt2gender)
    train_set.transform = gndr_normalize

    dev_set = dataset.ESPnetBucketDataset(
                os.path.join(args.temp_root, jsons['val']),
                os.path.join(args.temp_root, args.tok_file),
                transform=utt_normalize,
                num_buckets=args.n_buckets)

    bucket_train_loader = torch.utils.data.DataLoader(
                            train_set, 
                            batch_sampler=dataset.BucketBatchSampler(
                                shuffle=True, 
                                batch_size=args.batch_size, 
                                utt2idx=train_set.utt2idx, 
                                buckets=train_set.buckets),
                            collate_fn=dataset.collate)
    bucket_dev_loader = torch.utils.data.DataLoader(
                            dev_set,
                            batch_sampler=dataset.BucketBatchSampler(
                                shuffle=True,
                                batch_size=args.batch_size,
                                utt2idx=dev_set.utt2idx,
                                buckets=dev_set.buckets),
                            collate_fn=dataset.collate)

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
    dev_loss = []
    best_dev_loss = np.inf

    for epoch in range(args.n_epochs):
        losses = []
        model.train()
        for data in tqdm.tqdm(bucket_train_loader):
            optimizer.zero_grad()
            log_probs = model(data['feat'].cuda())
            loss = ctc_loss(log_probs.transpose(0, 1), data['label'].cuda(), 
                            data['feat_lens'].cuda(), data['label_lens'].cuda())
            loss.backward()

#             gradient clipping does slowdown, value taken from ESPnet
            torch.nn.utils.clip_grad_value_(model.parameters(), args.val_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_clip)
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
            batch_decoded, to_remove = decoder.batch_greedy_ctc_decode(
                                        log_probs, 
                                        zero_infinity=True)

            for i in range(log_probs.shape[0]):
                batch_dec = batch_decoded[i, :]
                batch_dec = batch_dec[batch_dec != to_remove]
                pred_words = decoder.compute_words(
                                    batch_dec, 
                                    train_set.idx2tok)

                label_chars = data['label'][i].cpu().detach().numpy()
                label_chars = label_chars[label_chars != 0] # remove padding
                label_words = decoder.compute_words(
                                label_chars,
                                train_set.idx2tok)

                char_errors = decoder.levenshtein(batch_dec, label_chars)
                print('predicted:', ' '.join(pred_words))
                print('label:', ' '.join(label_words))
                word_errors = decoder.levenshtein(pred_words, label_words)
                print(f'CER: {char_errors / len(label_chars)}')
                print(f'WER: {word_errors / len(label_words)}')

        dev_loss.append(np.mean(losses))

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
                

# TODO: actually do test and dev-set recognition with avg WER (per gender)
# TODO: look at shubham's paper and kaldi 4-gram for language model integration
def recognize(args, datadir, jsons):
    model = models.LSTM(num_layers=args.n_layers, hidden_dim=args.hidden_dim,
                        bidirectional=args.bidir)

    model.load_state_dict(torch.load(os.path.join(args.temp_root, 
                                                  args.model_dir,
                                                  'best.pt')))
    model.eval()

    for data in tqdm.tqdm(bucket_dev_loader):
        log_probs = model(data['feat'].cuda())
        log_probs = log_probs.cpu().detach().numpy()
        batch_decoded = decoder.batch_greedy_ctc_decode(log_probs, 
                                                        zero_infinity=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='WSJ CTC character ASR model')

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
                        type=bool,
                        help='clean up temporary data at the end',
                        default=False)
    parser.add_argument('--determ',
                        type=bool,
                        help='deterministic behavior for cuda',
                        default=True)

    parser.add_argument('--spk2gender_file',
                        type=str,
                        help='spk2gender file path from kaldi',
                        default='train_si284/spk2gender')
    parser.add_argument('--prop_female',
                        type=float,
                        help='proportion of female speakers in dataset',
                        default=0.5)
    parser.add_argument('--n_hrs',
                        type=float,
                        help='number of hours of subsampled data to use',
                        default=5.0)

    args = parser.parse_args()
    main(args)
