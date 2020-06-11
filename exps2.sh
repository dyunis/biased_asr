#!/usr/bin/bash

# no normalization
python train.py --model_dir=/scratch/asr_tmp/exps/2080_1e-4 --bucket_load_dir=buckets/2080

# utterance normalization
python train.py --model_dir=/scratch/asr_tmp/exps/2080_utt_1e-4 --bucket_load_dir=buckets/2080 --normalize=utt

# speaker normalization
python train.py --model_dir=/scratch/asr_tmp/exps/2080_spk_1e-4 --bucket_load_dir=buckets/2080 --normalize=spk

# gender normalization
python train.py --model_dir=/scratch/asr_tmp/exps/2080_gndr_1e-4 --bucket_load_dir=buckets/2080 --normalize=gndr

