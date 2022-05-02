import librosa
import soundfile as sf
import os
from tqdm import tqdm
import argparse

sr = 16000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_dir', help='speech directory', required=True)
    parser.add_argument('--output_dir', help='output directory', required=True)

    return parser.parse_known_args()

args, unused = parse_args()

directory = args.speech_dir
output_dir = args.output_dir

for filename in tqdm(os.listdir(directory)):
    #print(os.path.join(directory, filename))
    y, s = librosa.load(os.path.join(directory, filename), sr=sr)
#    print(s)
#    print(sr)
    sf.write(os.path.join(output_dir, filename), y, s)
