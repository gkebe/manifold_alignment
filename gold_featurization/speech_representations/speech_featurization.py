import argparse
from collections import defaultdict
import os
import pickle

import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel

import argparse
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_dir', help='speech directory', required=True)
    parser.add_argument('--model', help='pretrained model', required=True)
    parser.add_argument('--output', help='output file name (.pkl file with embeddings)', required=True)

    return parser.parse_known_args()

ARGS, unused = parse_args()
if ARGS.model == "wav2vec2":
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()

elif ARGS.model == "wavlm":
    processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").cuda()

model.eval()
speech_dict = dict()
data_path = ARGS.speech_dir


for file_name in tqdm(os.listdir(data_path)):
    wav_path = os.path.join(data_path, file_name)
    speech, _ = sf.read(wav_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.cuda()
    with torch.no_grad():
       hidden_states = model(input_values, output_hidden_states=True).hidden_states
    hidden_states = torch.cat([i for i in hidden_states][-4:]).transpose(0,1).contiguous().view(-1, 3072)
    features = torch.mean(hidden_states, dim=0).view(-1)
    speech_dict[file_name.replace(".wav", "")] = features.detach().cpu()

with open(ARGS.output, 'wb') as f:
    pickle.dump(speech_dict, f)