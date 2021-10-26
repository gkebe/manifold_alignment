import argparse
import os
import pickle
import random
import datetime

import scipy
import scipy.spatial
import torch

from datasets import GLData
from rownet import RowNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='name of experiment to evaluate')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--sample_size', type=int, default=0,
        help='number of pos/neg samples for each instance')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--embedded_dim', type=int, default=1024,
        help='embedded_dim')

    return parser.parse_known_args()

def evaluate(experiment, test_path, sample_size, gpu_num, embedded_dim):
    with open(test_path, 'rb') as fin:
        test_data = pickle.load(fin)

    language_test_data = [(l, i) for l, _, _, i, _ in test_data]
    vision_test_data = [(v, i) for _, v, _, i, _ in test_data]
    instance_names = [i for _, _, _, i, _ in test_data]

    print(f'num language: {len(language_test_data)}')
    print(f'num vision: {len(vision_test_data)}')

    language_dim = list(language_test_data[0][0].size())[0]
    vision_dim = list(vision_test_data[0][0].size())[0]

    results_dir = f'./output/{experiment}'
    train_results_dir = os.path.join(results_dir, 'train_results/')

    print(f'is_available(): {torch.cuda.is_available()}')
    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    language_model = RowNet(language_dim, embedded_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)
    language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt'), map_location=device))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt'), map_location=device))
    language_model.to(device)
    vision_model.to(device)
    language_model.eval()
    vision_model.eval()

    print(f'Starting evaluation: {datetime.datetime.now().time()}')

    language2language_fout = open(os.path.join(results_dir, 'language2language.txt'), 'w')
    language2language_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
    for language_index, language in enumerate(language_test_data):
        positive_indices = [i for i, name in enumerate(instance_names) if language[1] == name]
        negative_indices = random.sample([i for i, name in enumerate(instance_names) if language[1] != name], len(positive_indices))
        if sample_size:
            positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
            negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

        language_data = language[0].to(device)
        embedded_language = language_model(language_data).cpu().detach().numpy()

        for i in positive_indices:
            pos_language = language_test_data[i]
            pos_language_data = pos_language[0].to(device)
            embedded_pos_language = language_model(pos_language_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_language, embedded_pos_language)
            language2language_fout.write(f'{language[1]},{pos_language[1]},p,{dist}\n')

        for i in negative_indices: 
            neg_language = language_test_data[i]
            neg_language_data = neg_language[0].to(device)
            embedded_neg_language = language_model(neg_language_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_language, embedded_neg_language)
            language2language_fout.write(f'{language[1]},{neg_language[1]},n,{dist}\n')        
    language2language_fout.close()
    print(f'Wrote language2language: {datetime.datetime.now().time()}')

    vision2vision_fout = open(os.path.join(results_dir, 'vision2vision.txt'), 'w')
    vision2vision_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
    for vision_index, vision in enumerate(vision_test_data):
        positive_indices = [i for i, name in enumerate(instance_names) if vision[1] == name]
        negative_indices = random.sample([i for i, name in enumerate(instance_names) if vision[1] != name], len(positive_indices))
        if sample_size:
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

    vision2language_fout = open(os.path.join(results_dir, 'vision2language.txt'), 'w')
    vision2language_fout.write('vision_instance,language_instance,p/n,embedded_distance\n')
    for vision_index, vision in enumerate(vision_test_data):
        positive_indices = [i for i, name in enumerate(instance_names) if vision[1] == name]
        negative_indices = random.sample([i for i, name in enumerate(instance_names) if vision[1] != name], len(positive_indices))
        if sample_size:
            positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
            negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))
        
        vision_data = vision[0].to(device)
        embedded_vision = vision_model(vision_data).cpu().detach().numpy()

        for i in positive_indices:
            pos_language = language_test_data[i]
            pos_language_data = pos_language[0].to(device)
            embedded_pos_language = language_model(pos_language_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_language)
            vision2language_fout.write(f'{vision[1]},{pos_language[1]},p,{dist}\n')

        for i in negative_indices:
            neg_language = language_test_data[i]
            neg_language_data = neg_language[0].to(device)
            embedded_neg_language = language_model(neg_language_data).cpu().detach().numpy()
            dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_language)
            vision2language_fout.write(f'{vision[1]},{neg_language[1]},n,{dist}\n')
    vision2language_fout.close()
    print(f'Wrote vision2language: {datetime.datetime.now().time()}')

def main():
    ARGS, unused = parse_args()
    
    evaluate(
        ARGS.experiment,
        ARGS.test_data,
        ARGS.sample_size,
        ARGS.gpu_num,
        ARGS.embedded_dim,
    )

if __name__ == '__main__':
    main()
