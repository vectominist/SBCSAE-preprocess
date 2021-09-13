import argparse
import os
from os.path import join

import csv
import unicodedata
from string import punctuation
import re

from tqdm import tqdm
import torch
import torchaudio
from librosa import resample

MIN_SEC = 0.1
MAX_SEC = 15.0
TARGET_SR = 16000  # targeted sample rate for audio files

TRAIN_ID = list(range(1, 47))
DEV_ID = list(range(47, 54))
TEST_ID = list(range(54, 61))

rm_punctuation = '.,-%?!@+~_*#&\"'
translator = str.maketrans(
    rm_punctuation + '’', ' ' * len(rm_punctuation) + '\'')

vocab = set(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ\' '))


def normalization(sent: str):
    if sent.find('[') >= 0 or sent.find(']') >= 0 or sent.find('<') >= 0 or sent.find('>') >= 0:
        # having brackets means there is overlapping with other speakers
        # having < or > means unsure transcriptions
        return ''

    # remove parentheses and everything inside them
    sent = re.sub(r'\([^)]*\)', '', sent)
    if sent.find('(') >= 0 or sent.find(')') >= 0:
        return ''

    # remove all "="
    sent = sent.replace('=', '')
    # remove punctuations
    sent = sent.translate(translator)
    # remove repeated whitespaces
    sent = re.sub(' +', ' ', sent).strip()

    sent = sent.upper() + ' '
    if sent.find('XX') >= 0 or sent.find(' X ') >= 0:
        # unknown words
        return ''

    if any(s not in vocab for s in list(sent)):
        return ''

    return sent.strip()


def read_trn(path: str, name_base: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        data = []
        # for line in tqdm(fp):
        for line in fp:
            line = line.strip('\t').strip('\n').split('\t')
            if len(line) == 2 and len(line[0].split(' ')) >= 2:
                # <t1> <t2> [speaker]\t<sentence>
                time_begin, time_end = line[0].split(' ')[:2]
                sentence = line[1]
            elif len(line) == 3 and len(line[0].split(' ')) == 2:
                # <t1> <t2>\t[speaker]\t<sentence>
                time_begin, time_end = line[0].split(' ')
                sentence = line[2]
            elif len(line) == 4:
                try:
                    time_begin, time_end = float(line[0]), float(line[1])
                except:
                    time_begin, time_end = line[0].split(' ')[:2]

                sentence = line[3]
            else:
                continue

            time_begin, time_end = float(time_begin), float(time_end)
            if time_end - time_begin > MAX_SEC or time_end - time_begin < MIN_SEC:
                continue

            sentence = normalization(sentence)

            if len(sentence) <= 1:
                continue

            data.append(
                {
                    't_begin': time_begin,
                    't_end': time_end,
                    'sentence': sentence,
                    'name': '{}_{:06d}-{:06d}.wav'.format(
                        name_base, int(time_begin * 100), int(time_end * 100))
                }
            )

        data = sorted(data, key=lambda x: x['t_end'])
        data_new = []
        # remove overlapped clips
        for i in range(len(data)):
            if i == 0:
                data_new.append(data[i])
                continue

            j = len(data_new) - 1
            overlapped = False
            while j >= 0:
                if data[i]['t_begin'] < data_new[j]['t_end']:
                    data_new.pop()
                    overlapped = True
                    j -= 1
                else:
                    break

            if not overlapped:
                data_new.append(data[i])

        total_secs = 0
        for d in data_new:
            total_secs += d['t_end'] - d['t_begin']

        print('Found {} utterances of total {:.2f} hours.'
              .format(len(data_new), total_secs / 3600.))

        return data_new, total_secs


def process_mp3(file: str, data: list, wav_dir: str):
    wav, sample_rate = torchaudio.load(file)
    if wav.shape[0] > 1:
        wav = wav.mean(0).unsqueeze(0)
    if sample_rate != TARGET_SR:
        # downsample to desired sample rate
        wav = resample(
            wav.squeeze(0).numpy(), sample_rate,
            TARGET_SR, res_type='kaiser_best')
        wav = torch.FloatTensor(wav).unsqueeze(0)

    for d in tqdm(data):
        t_begin = int(d['t_begin'] * TARGET_SR)
        t_end = int(d['t_end'] * TARGET_SR)
        wav_clip = wav[:, t_begin: t_end + 1]
        clip_name = join(wav_dir, d['name'])
        torchaudio.save(clip_name, wav_clip, TARGET_SR)


def write_tsv(data, out_path):
    with open(out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(['path', 'sentence'])
        for d in data:
            writer.writerow([d['name'], d['sentence']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', type=str,
                        help='Directory of the trn files.')
    parser.add_argument('--mp3', type=str, default='none',
                        help='Directory of the mp3 files.')
    parser.add_argument('--wav', type=str, default='none',
                        help='Directory to save the wav files.')
    parser.add_argument('--tsv', type=str, default='none',
                        help='Directory to save the tsv files.')
    args = parser.parse_args()

    if args.wav != 'none':
        os.makedirs(args.wav, exist_ok=True)
    if args.tsv != 'none':
        os.makedirs(args.tsv, exist_ok=True)

    for s, partition_id in [('train', TRAIN_ID), ('dev', DEV_ID), ('test', TEST_ID)]:
        print(s + ' set')
        data_list = []
        total_len = 0.
        for i in partition_id:
            name = 'SBC{:03}'.format(i)
            print(name)

            data, tot_len = read_trn(
                join(args.trn, name + '.trn'), name)
            data_list += data
            total_len += tot_len

            if args.wav != 'none' and args.mp3 != 'none':
                print('Splitting audio files')
                process_mp3(
                    join(args.mp3, str(i).zfill(2) + '.mp3'), data, args.wav)

        print(' == Found {} utterances of total {:.2f} hours from split {} =='
              .format(len(data_list), total_len / 3600., s))

        if args.tsv != 'none':
            data_list = sorted(
                data_list, key=lambda x: x['t_end'] - x['t_begin'], reverse=True)
            write_tsv(data_list, join(args.tsv, s + '.tsv'))


if __name__ == '__main__':
    main()
