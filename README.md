## Investigating bias in ASR

TTIC Speech Technologies course project 2020
David Yunis and Pushkar Shukla

### Dependencies:
This code tested on `python 3.8.1` with the following external packages:
```
torch 1.5.0
numpy 1.18.1
kaldiio 2.15.1
scikit-learn 0.22.2
matplotlib 3.1.3
```

### Files:
- `train.py`: the main training loop and evaluation
- `train_gender.py`: gender discriminator experiments
- `dataset.py`: the file with all the torch datasets and samplers, I'm using 
  a json from a package called ESPnet that does all the kaldi preprocessing (of
  transcripts and for filterbanks from wav files)
- `models.py`: models for ASR, right now it's just an LSTM
- `models_gender.py`: LSTM and gender discriminator
- `decoder.py`: a greedy decoder taking in the log probabilities of a sequence
  and outputting the most probable indices
- `gender_subset.py`: the file for taking WSJ and producing a dataset with 
  certain fractions of gender data, this data already exists in the dataset
  directory
- `transforms.py`: a file for the normalizations
- `exps{1,2,3}.sh`: scripts for running experiments
- `make_figs.py`: code for all the plots in the report
- `utils.py`: some small tools I use for debugging and optimizing, kinda 
  irrelevant

### Dataset
my treated copy of WSJ is located at `/share/data/speech/Data/dyunis/data/wsj_espnet`,
copy to scratch before starting training (this is done in the training code 
already with `safe_cptree`)

see `main()` in `train.py` for more path info

the balanced and imbalanced datasets are inside the WSJ directory at 
`buckets/5050` and `buckets/2080`, where the first number is 
the proportion of female speakers, and the second is the proportion of male

### Running the code

```
python train.py --bucket_load_dir buckets/5050 --model_dir=./models
```
should be all it takes to run the code (for the speech group), it will save 
all models and outputs in a directory called `models`.
