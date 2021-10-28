#!/bin/bash

data_dir=$1  # directory to save data

mkdir -p ${data_dir}
mkdir -p ${data_dir}/mp3
mkdir -p ${data_dir}/trn

# Download audio data (.wav files)
wget -r -nH --cut-dirs=2 -np \
    --reject="index.html*","*.wav" \
    -P ${data_dir}/mp3 \
    https://media.talkbank.org/ca/SBCSAE/ || exit 1;

# Download transcript files
for id in {001..060}
do
    wget https://www.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7/files/sitefiles/research/SBC/SBC${id}.trn \
        -P ${data_dir}/trn || exit 1;
done
