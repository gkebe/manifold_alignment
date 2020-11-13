import argparse
import os
import pickle
import random
from collections import defaultdict

import scipy
import scipy.spatial
import torch
import torch.nn.functional as F

from datasets import GLData
from rownet import RowNet
import speech_featurize as sf
import text_featurize as tf
import vision_featurize as vf
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', help='wav file or textual description.', required=True)
    parser.add_argument('--rgb', help='path to rgb file', required=True)
    parser.add_argument('--depth', help='path to depth file', required=True)
    parser.add_argument('--language_type', help='decoar, vq-wav2vec or transcription', required=True)
    parser.add_argument('--experiment_name', help='name of experiment to test')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='embedded_dim')

    return parser.parse_known_args()

def infer(language, rgb, depth, language_type, experiment_name, gpu_num, embedded_dim):

    print(language_type)
    if language_type == "decoar":
        language_data = sf.decoar_featurize(wav_file=language, gpu_num=gpu_num)
    elif language_type == "vq-wav2vec":
        language_data = sf.vq_wav2vec_featurize(wav_file=language, gpu_num=gpu_num)
    elif language_type == "transcription":
        language_data = tf.bert_embedding(sentence=language)

    vision_data = vf.vision_featurize(rgb=rgb, depth=depth, gpu_num=gpu_num)

    # BERT dimension
    language_dim = list(language_data.size())[0]
    # Eitel dimension
    vision_dim = list(vision_data.size())[0]

    results_dir = f'./output/{experiment_name}'
    train_results_dir = os.path.join(results_dir, 'train_results/')

    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(vision_data)
    language_data = language_data.to(device)
    vision_data = vision_data.to(device)
    print(vision_data)
    language_model = RowNet(language_dim, embedded_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)
    # language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt')))
    # vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt')))
    language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt'), map_location=device))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt'), map_location=device))
    language_model.to(device)
    vision_model.to(device)
    language_model.eval()
    vision_model.eval()

    print(next(language_model.parameters()).device, language_data.get_device(), next(vision_model.parameters()).device, vision_data.get_device())
    embedded_vision = vision_model(vision_data).cpu().detach().numpy()
    embedded_language = language_model(language_data).cpu().detach().numpy()
    dist = scipy.spatial.distance.cosine(embedded_vision, embedded_language)

    return dist

def main():
    ARGS, unused = parse_args()

    dist = infer(
        language = ARGS.language,
        rgb = ARGS.rgb,
        depth =ARGS.depth,
        language_type=ARGS.language_type,
        experiment_name=ARGS.experiment_name,
        gpu_num=ARGS.gpu_num,
        embedded_dim=ARGS.embedded_dim
    )

    print(dist)

if __name__ == '__main__':
    main()
