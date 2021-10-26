import argparse
from collections import defaultdict
import os
import pickle

import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

import soundfile as sf

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").cuda()
model.eval()
speech_dict = dict()
data_path = "./speech_16/"


for file_name in tqdm(os.listdir(data_path)):
    wav_path = os.path.join(data_path, file_name)
    speech, _ = sf.read(wav_path)
    input_values = tokenizer(speech, return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1

    # retrieve logits
    with torch.no_grad():
      logits = model(input_values.cuda()).logits.detach().cpu()
  
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    speech_dict[file_name.replace(".wav", "")] = transcription
    

with open('wav2vec2_transcriptions.pkl', 'wb') as f:
    pickle.dump(speech_dict, f)

