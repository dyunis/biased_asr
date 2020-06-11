#!/usr/bin/bash

# no normalization
python train.py --model_dir=/scratch/asr_tmp/exps/5050_1e-4 --bucket_load_dir=buckets/5050
python train.py --model_dir=/scratch/asr_tmp/exps/5050_1e-3 --bucket_load_dir=buckets/5050 --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/5050_1e-5 --bucket_load_dir=buckets/5050 --learning_rate=0.00001

python train.py --model_dir=/scratch/asr_tmp/exps/2080_1e-4 --bucket_load_dir=buckets/2080
python train.py --model_dir=/scratch/asr_tmp/exps/2080_1e-3 --bucket_load_dir=buckets/2080 --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/2080_1e-5 --bucket_load_dir=buckets/2080 --learning_rate=0.00001

# utterance normalization
python train.py --model_dir=/scratch/asr_tmp/exps/5050_utt_1e-4 --bucket_load_dir=buckets/5050 --normalize=utt
python train.py --model_dir=/scratch/asr_tmp/exps/5050_utt_1e-3 --bucket_load_dir=buckets/5050 --normalize=utt --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/5050_utt_1e-5 --bucket_load_dir=buckets/5050 --normalize=utt --learning_rate=0.00001

python train.py --model_dir=/scratch/asr_tmp/exps/2080_utt_1e-4 --bucket_load_dir=buckets/2080 --normalize=utt
python train.py --model_dir=/scratch/asr_tmp/exps/2080_utt_1e-3 --bucket_load_dir=buckets/2080 --normalize=utt --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/2080_utt_1e-5 --bucket_load_dir=buckets/2080 --normalize=utt --learning_rate=0.00001

# speaker normalization
python train.py --model_dir=/scratch/asr_tmp/exps/5050_spk_1e-4 --bucket_load_dir=buckets/5050 --normalize=spk
python train.py --model_dir=/scratch/asr_tmp/exps/5050_spk_1e-3 --bucket_load_dir=buckets/5050 --normalize=spk --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/5050_spk_1e-5 --bucket_load_dir=buckets/5050 --normalize=spk --learning_rate=0.00001

python train.py --model_dir=/scratch/asr_tmp/exps/2080_spk_1e-4 --bucket_load_dir=buckets/2080 --normalize=spk
python train.py --model_dir=/scratch/asr_tmp/exps/2080_spk_1e-3 --bucket_load_dir=buckets/2080 --normalize=spk --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/2080_spk_1e-5 --bucket_load_dir=buckets/2080 --normalize=spk --learning_rate=0.00001

# gender normalization
python train.py --model_dir=/scratch/asr_tmp/exps/5050_gndr_1e-4 --bucket_load_dir=buckets/5050 --normalize=gndr
python train.py --model_dir=/scratch/asr_tmp/exps/5050_gndr_1e-3 --bucket_load_dir=buckets/5050 --normalize=gndr --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/5050_gndr_1e-5 --bucket_load_dir=buckets/5050 --normalize=gndr --learning_rate=0.00001

python train.py --model_dir=/scratch/asr_tmp/exps/2080_gndr_1e-4 --bucket_load_dir=buckets/2080 --normalize=gndr
python train.py --model_dir=/scratch/asr_tmp/exps/2080_gndr_1e-3 --bucket_load_dir=buckets/2080 --normalize=gndr --learning_rate=0.001
python train.py --model_dir=/scratch/asr_tmp/exps/2080_gndr_1e-5 --bucket_load_dir=buckets/2080 --normalize=gndr --learning_rate=0.00001

