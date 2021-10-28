# SBCSAE-preprocess
Preprocessing scripts for the [Santa Barbara Corpus of Spoken American English (SBCSAE)](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus).

## Instructions

### Installation
- Python >= 3.6
- Install sox on your OS
- Install dependencies
    ```
    pip install -r requirements.txt
    ```

### Download Data
Download corpus data to `<data dir>`.
```
./download.sh <data dir>
```
The data directory should look like
```
<data dir>/
├── mp3
│   ├── 01.mp3
│   ├── 02.mp3
│   ├── ...
│   └── 60.mp3
└── trn
    ├── SBC001.trn
    ├── SBC002.trn
    ├── ...
    └── SBC060.trn
```

### Process Data
- Converts and segments `.mp3` files to `.wav` files.
- Transforms `.trn` files into transcriptions for ASR training and evaluation.

```
python3 preprocess.py --trn <data dir>/trn --mp3 <data dir>/mp3 --wav <data dir>/wav --tsv <transcript dir>
```
After processing, there will be three transcription files `train.tsv`, `dev.tsv`, and `test.tsv`. Some examples in the `train.tsv` file is shown as follows
```
SBC009_000000-001316.wav    AM I DOING THAT RIGHT SO FAR
SBC042_020175-021461.wav    ALL I NEED IS YOUR SIGNATURE SO I CAN PLAY THE VOLLEYBALL
SBC042_081976-083246.wav    WELL I GUESS THAT NOTE
```
The `.wav` files are stored in `<data dir>/wav` while the transciptions are in `<transcript dir>`. The naming criterion of `.wav` files is 
```
SBC<.mp3 index>_<start sec x 100>-<end sec x 100>.wav
```
E.g., the `SBC009_000000-001316.wav` file is the 0.00 sec to the 13.16 sec of `09.mp3`. Note that we remove utterances longer than 15 seconds.

## Reference
Du Bois, John W., Wallace L. Chafe, Charles Meyer, Sandra A. Thompson, Robert Englebretson, and Nii Martey. 2000-2005. Santa Barbara corpus of spoken American English, Parts 1-4. Philadelphia: Linguistic Data Consortium.

## Citation
TBA
