## Investigating bias in ASR

### Files:
- `train.py`: the main training loop (and a skeleton for test-set evaluation)
  - currently encountering NaNs with full WSJ training, but I want to finish 
    coding our subsampled dataset first
- `dataset.py`: the file with all the torch datasets and samplers, I'm using 
  a json from a package called ESPnet that does all the kaldi preprocessing (of
  transcripts and for filterbanks from wav files)
- `models.py`: models for ASR, right now it's just an LSTM
- `decoder.py`: a greedy decoder taking in the log probabilities of a sequence
  and outputting the most probable indices
- `gender_subset.py`: the file for taking WSJ and producing a dataset with 
  certain fractions of gender data, but this data is already generated in the
  dataset directory
- `utils.py`: some small tools I use for debugging and optimizing, kinda 
  irrelevant

### Dataset
my treated copy of WSJ is located at `/share/data/speech/Data/dyunis/data/wsj_espnet`,
copy to scratch before starting training (this is done in the training code 
already with `safe_cptree`)

see `main()` in `train.py` for more path info

the balanced and imbalanced datasets are inside the WSJ directory at 
`buckets/5050`, `buckets/8020` and `buckets/2080`, where the first number is 
the proportion of female speakers, and the second is the proportion of male

### TODO:
- David:
  - [x] finish up dataset splitting code (in `gender_subset.py`)
  - [x] code up WER evaluation (in `decoder.py`)
  - [x] normalizations (as transforms in `transforms.py`)
  - [ ] argument parsing
  - [ ] logging and plotting for each experiment to record loss, CER, and WER
  - [ ] saving models via best dev results
- Pushkar:
  - adversarial regularizer (ideally in `models.py`, take a look at the training
    code in `train.py` and `dataset.py`)

