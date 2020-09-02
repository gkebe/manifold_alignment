import argparse
from collections import defaultdict
import os
import pickle

import torch
import torchaudio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path to wav files')
    parser.add_argument('--out_file', help='path to output')
    
    return parser.parse_known_args()

def main():
    ARGS, unused = parse_args()    

    data = defaultdict(list)

    for wav in os.listdir(ARGS.data_dir):
        object_name = '_'.join(wav.split('_')[:-2])
        instance_name = '_'.join(wav.split('_')[:-1])

        wav_path = os.path.join(ARGS.data_dir, wav)
        waveform, sample_rate = torchaudio.load_wav(wav_path, normalization=True)

        mfcc_transform = torchaudio.transforms.MFCC(sample_rate, log_mels=True)
        mfcc = mfcc_transform(waveform)
  
        data['object_name'].append(object_name)
        data['instance_name'].append(instance_name)
        data['mfcc'].append(mfcc)
      
        print(f'{instance_name}, {type(mfcc)}, {mfcc.size()}')

    data = dict(data)
    with open(ARGS.out_file, 'wb') as fout:
        pickle.dump(data, fout)

if __name__ == '__main__':
    main()
