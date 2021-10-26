# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:21:06 2020

@author: T530
"""
import argparse
from collections import defaultdict
import os
import pickle

import torch
import torchaudio
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mfcc', help='number of mfcc features')

    return parser.parse_known_args()

ARGS, unused = parse_args()
speech_dict = dict()
data_path = "speech_16_/"

for file_name in tqdm(os.listdir(data_path)):
    wav_path = os.path.join(data_path, file_name)
    waveform, sample_rate = torchaudio.load_wav(wav_path, normalization=True)
    n_mfcc = int(ARGS.n_mfcc)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate, n_mfcc=n_mfcc, log_mels=True)
    mfcc = torch.transpose(mfcc_transform(waveform)[0],0,1)

    speech_dict[file_name.replace(".wav", "")] = mfcc

with open('mfcc_features_'+str(n_mfcc)+'_.pkl', 'wb') as f:
    pickle.dump(speech_dict, f)
