import argparse
import os
import pickle
import random
from collections import defaultdict

import scipy
import scipy.spatial
import torch
import torch.nn.functional as F

from split_data_pre_loader import gl_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='RandomExperiment',
        help='name of experiment to test')
    parser.add_argument('--data_path', type=str,
        help='path to testing data')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='embedded_dim')

    return parser.parse_known_args()

class RowNet(torch.nn.Module):
    def __init__(self, input_size, embed_dim=1024):
        # Language (BERT): 3072, Vision+Depth (ResNet152): 2048 * 2.
        super(RowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, input_size)
        self.fc3 = torch.nn.Linear(input_size, embed_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=.2)
        x = self.fc3(x)

        return x

def embed(experiment_name, data_path, gpu_num, embedded_dim):

    with open(data_path, 'rb') as fin:
        data = pickle.load(fin)

    language_data_ = [(data["language_data"][i], data["object_names"][i], data["instance_names"][i]) for i in range(len(data["instance_names"]))]
    vision_data_ = [(data["vision_data"][i], data["object_names"][i], data["instance_names"][i]) for i in range(len(data["instance_names"]))]

    # BERT dimension
    language_dim = list(language_data_[0][0].size())[0]
    # Eitel dimension
    vision_dim = list(vision_data_[0][0].size())[0]

    results_dir = f'./output/{experiment_name}'
    train_results_dir = os.path.join(results_dir, 'train_results/')

    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)


    language_model = RowNet(language_dim, embed_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embed_dim=embedded_dim)
    language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt')))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt')))
    language_model.to(device)
    vision_model.to(device)
    language_model.eval()
    vision_model.eval()

    objects_vision = list()
    objects_language = list()

    instances_vision = list()
    instances_language = list()

    embedded_visions = list()
    embedded_languages = list()

    for vision_index, vision in enumerate(vision_data_):
        
        vision_data = vision[0].to(device)
        vision_object = vision[1]
        vision_instance = vision[2]
        
        embedded_vision = vision_model(vision_data).cpu().detach()
        instances_vision.append(vision_instance)
        objects_vision.append(vision_object)
        embedded_visions.append(embedded_vision)

    for language_index, language in enumerate(language_data_):
        language_data = language[0].to(device)
        language_object = language[1]
        language_instance = language[2]

        embedded_language = language_model(language_data).cpu().detach()
        instances_language.append(language_instance)
        objects_language.append(language_object)
        embedded_languages.append(embedded_language)

    vision_dict = {"instance_names": instances_vision, "object_names": objects_vision, "embedded_vectors":embedded_visions}
    language_dict = {"instance_names": instances_language, "object_names": objects_language, "embedded_vectors":embedded_languages}

    return vision_dict, language_dict

def main():
    ARGS, unused = parse_args()

    v_embeddings, l_embeddings = embed(
        ARGS.experiment_name,
        ARGS.data_path,
        ARGS.gpu_num,
        ARGS.embedded_dim
    )

    with open("output/"+ARGS.experiment_name+"_l_embeddings.pkl", 'wb') as fout:
        pickle.dump(l_embeddings, fout)

    with open("output/"+ARGS.experiment_name+"_v_embeddings.pkl", 'wb') as fout:
        pickle.dump(v_embeddings, fout)

    print(f'Wrote two files\n\t{"output/"+ARGS.experiment_name+"_l_embeddings"}\n\t{"output/"+ARGS.experiment_name+"_v_embeddings"}')


if __name__ == '__main__':
    main()
