import argparse
import os
import pickle
import random
import datetime

import scipy
import scipy.spatial
import torch

from datasets import GLData
from lstm import LSTM
from rnn import RNN
from rownet import RowNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='name of experiment to evaluate')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--embedded_dim', type=int, default=1024,
        help='embedded_dim')

    return parser.parse_known_args()

def evaluate(experiment, test_path, sample_size, gpu_num, embedded_dim):
    with open(test_path, 'rb') as fin:
        test_data = pickle.load(fin)

    speech_test_data = [(s, i) for s, _, _, i in test_data]
    vision_test_data = [(v, i) for _, v, _, i in test_data]
    instance_names = [i for _, _, _, i in test_data]

    print(f'num speech: {len(speech_test_data)}')
    print(f'num vision: {len(vision_test_data)}')

    vision_dim = list(vision_test_data[0][0].size())[0]

    results_dir = f'./output/{experiment}'
    train_results_dir = os.path.join(results_dir, 'train_results/')

    print(f'is_available(): {torch.cuda.is_available()}')
    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    speech_model = LSTM(
        input_size=40,
        output_size=embedded_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        device=device
    )
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)
    speech_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt'), map_location=device))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt'), map_location=device))
    speech_model.to(device)
    vision_model.to(device)
    speech_model.eval()
    vision_model.eval()

    print(f'Starting evaluation: {datetime.datetime.now().time()}')

    speech2speech_fout = open(os.path.join(results_dir, 'speech2speech.txt'), 'w')
    speech2speech_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
    for speech_index, speech in enumerate(speech_test_data):
        positive_indices = [i for i, name in enumerate(instance_names) if speech[1] == name]
        negative_indices = random.sample([i for i, name in enumerate(instance_names) if speech[1] != name], len(positive_indices)) 
        positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
        negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

        speech_data = speech[0].to(device)
        # TODO: NEEDS TO BE HANDLED WHEN CREATING FEATURES
        speech_data = speech_data.permute(0, 2, 1)
        embedded_speech = speech_model(speech_data).cpu().detach().numpy()

        for i in positive_indices:
            pos_speech = speech_test_data[i]
            pos_speech_data = pos_speech[0].to(device)
            pos_speech_data = pos_speech_data.permute(0, 2, 1)
            embedded_pos_speech = speech_model(pos_speech_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_speech, embedded_pos_speech)
            speech2speech_fout.write(f'{speech[1]},{pos_speech[1]},p,{dist}\n')

        for i in negative_indices: 
            neg_speech = speech_test_data[i]
            neg_speech_data = neg_speech[0].to(device)
            neg_speech_data = neg_speech_data.permute(0, 2, 1)
            embedded_neg_speech = speech_model(neg_speech_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_speech, embedded_neg_speech)
            speech2speech_fout.write(f'{speech[1]},{neg_speech[1]},n,{dist}\n')        
    speech2speech_fout.close()
    print(f'Wrote speech2speech: {datetime.datetime.now().time()}')

    vision2vision_fout = open(os.path.join(results_dir, 'vision2vision.txt'), 'w')
    vision2vision_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
    for vision_index, vision in enumerate(vision_test_data):
        positive_indices = [i for i, name in enumerate(instance_names) if vision[1] == name]
        negative_indices = random.sample([i for i, name in enumerate(instance_names) if vision[1] != name], len(positive_indices))
        positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
        negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))
        
        vision_data = vision[0].to(device)
        embedded_vision = vision_model(vision_data).cpu().detach().numpy()

        for i in positive_indices:
            pos_vision = vision_test_data[i]
            pos_vision_data = pos_vision[0].to(device)
            embedded_pos_vision = vision_model(pos_vision_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_vision)
            vision2vision_fout.write(f'{vision[1]},{pos_vision[1]},p,{dist}\n')

        for i in negative_indices:
            neg_vision = vision_test_data[i]
            neg_vision_data = neg_vision[0].to(device)
            embedded_neg_vision = vision_model(neg_vision_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_vision)
            vision2vision_fout.write(f'{vision[1]},{neg_vision[1]},n,{dist}\n')
    vision2vision_fout.close()
    print(f'Wrote vision2vision: {datetime.datetime.now().time()}')

    vision2speech_fout = open(os.path.join(results_dir, 'vision2speech.txt'), 'w')
    vision2speech_fout.write('vision_instance,speech_instance,p/n,embedded_distance\n')
    for vision_index, vision in enumerate(vision_test_data):
        positive_indices = [i for i, name in enumerate(instance_names) if vision[1] == name]
        negative_indices = random.sample([i for i, name in enumerate(instance_names) if vision[1] != name], len(positive_indices))
        positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
        negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))
        
        vision_data = vision[0].to(device)
        embedded_vision = vision_model(vision_data).cpu().detach().numpy()

        for i in positive_indices:
            pos_speech = speech_test_data[i]
            pos_speech_data = pos_speech[0].to(device)
            pos_speech_data = pos_speech_data.permute(0, 2, 1)
            embedded_pos_speech = speech_model(pos_speech_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_speech, embedded_pos_speech)
            vision2speech_fout.write(f'{vision[1]},{pos_speech[1]},p,{dist}\n')

        for i in negative_indices:
            neg_speech = speech_test_data[i]
            neg_speech_data = neg_speech[0].to(device)
            neg_speech_data = neg_speech_data.permute(0, 2, 1)
            embedded_neg_speech = speech_model(neg_speech_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_speech, embedded_neg_speech)
            vision2speech_fout.write(f'{vision[1]},{neg_speech[1]},n,{dist}\n')
    vision2speech_fout.close()
    print('Wrote vision2speech')

def main():
    ARGS, unused = parse_args()
    
    evaluate(
        ARGS.experiment,
        ARGS.test_data,
        ARGS.gpu_num,
        ARGS.embedded_dim,
    )

if __name__ == '__main__':
    main()
