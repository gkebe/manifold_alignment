import argparse
from collections import defaultdict
import os
import pickle

import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
speech_dict = dict()
data_path = "./speech_16/"


for file_name in tqdm(os.listdir(data_path)):
    wav_path = os.path.join(data_path, file_name)
    speech, _ = sf.read(wav_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.cuda()
    with torch.no_grad():
       hidden_states = model(input_values, output_hidden_states=True).hidden_states
    hidden_states = torch.cat([i for i in hidden_states][-4:]).transpose(0,1).contiguous().view(-1, 3072)
    features = torch.mean(hidden_states, dim=0).view(-1)
    speech_dict[file_name.replace(".wav", "")] = features.detach().cpu()

with open('wav2vec2_mean_features.pkl', 'wb') as f:
    pickle.dump(speech_dict, f)

