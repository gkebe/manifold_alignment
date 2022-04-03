import argparse
from collections import defaultdict
import os
import pickle

import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, WavLMForCTC

import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_dir', help='speech directory', required=True)
    parser.add_argument('--model', help='pretrained model', required=True)
    parser.add_argument('--output', help='output file name (.pkl file with transcriptions)', required=True)

    return parser.parse_known_args()

ARGS, unused = parse_args()

if ARGS.model == "wav2vec2":
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").cuda()

elif ARGS.model == "wavlm":
    tokenizer = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").cuda()

model.eval()
data_path = ARGS.speech_dir
speech_dict = dict()

for file_name in tqdm(os.listdir(data_path)):
    wav_path = os.path.join(data_path, file_name)
    speech, _ = sf.read(wav_path)
    input_values = tokenizer(speech, return_tensors="pt", padding="longest",
                             sampling_rate=16000).input_values  # Batch size 1

    # retrieve logits
    with torch.no_grad():
        logits = model(input_values.cuda()).logits.detach().cpu()

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    speech_dict[file_name.replace(".wav", "")] = transcription

with open(ARGS.output, 'wb') as f:
    pickle.dump(speech_dict, f)