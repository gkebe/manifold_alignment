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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='name of experiment to test')
    parser.add_argument('--test_data_path', help='path to testing data')
    parser.add_argument('--pos_neg_examples_file', default=None,
        help='path to examples pkl')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='embedded_dim')

    return parser.parse_known_args()

def evaluate(experiment_name, test_data_path, pos_neg_examples_file, gpu_num, embedded_dim):

    pn_fout = open('./pn_eval_output.txt', 'w')
    rand_fout = open('./rand_eval_output.txt', 'w')
    with open(test_data_path, 'rb') as fin:
        test_data = pickle.load(fin)

    language_test_data = [(l, i) for l, _, _, i in test_data]
    vision_test_data = [(v, i) for _, v, _, i in test_data]

    # BERT dimension
    language_dim = list(language_test_data[0][0].size())[0]
    # Eitel dimension
    vision_dim = list(vision_test_data[0][0].size())[0]

    results_dir = f'./output/{experiment_name}'
    train_results_dir = os.path.join(results_dir, 'train_results/')

    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    with open(pos_neg_examples_file, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)

    language_model = RowNet(language_dim, embed_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embed_dim=embedded_dim)
    # language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt')))
    # vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt')))
    language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt'), map_location=device))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt'), map_location=device))
    language_model.to(device)
    vision_model.to(device)
    language_model.eval()
    vision_model.eval()

    # Vision to Language
    reciprocal_sum_euclid_pos_neg = 0
    reciprocal_sum_cosine_pos_neg = 0
    reciprocal_sum_euclid_random = 0
    reciprocal_sum_cosine_random = 0
    v_to_l_pn_counts = defaultdict(int)
    v_to_l_rand_counts = defaultdict(int)
    for vision_index, vision in enumerate(vision_test_data):
        pn_fout.write(f'V->L {language_test_data[vision_index][1]} ')
        rand_fout.write(f'V->L {language_test_data[vision_index][1]} ')

        euclidean_distances = []
        cosine_distances = []
        vision_data = vision[0].to(device)
        embedded_vision = vision_model(vision_data).cpu().detach().numpy()

        #
        # Pos/Neg MRR evaluation
        #

        euclidean_distances_pos_neg = []
        cosine_distances_pos_neg = []

        pos_index = pos_neg_examples[vision_index][0]
        neg_index = pos_neg_examples[vision_index][1]

        pn_fout.write(f'{language_test_data[pos_index][1]} {language_test_data[neg_index][1]} ')

        language_target = language_test_data[vision_index][0].to(device)
        language_pos = language_test_data[pos_index][0].to(device)
        language_neg = language_test_data[neg_index][0].to(device)
        embedded_language_target = language_model(language_target).cpu().detach().numpy()
        embedded_language_pos = language_model(language_pos).cpu().detach().numpy()
        embedded_language_neg = language_model(language_neg).cpu().detach().numpy()

        euclidean_distances_pos_neg.append(('target', scipy.spatial.distance.euclidean(embedded_vision, embedded_language_target)))
        euclidean_distances_pos_neg.append(('pos', scipy.spatial.distance.euclidean(embedded_vision, embedded_language_pos)))
        euclidean_distances_pos_neg.append(('neg', scipy.spatial.distance.euclidean(embedded_vision, embedded_language_neg)))

        cosine_distances_pos_neg.append(('target', scipy.spatial.distance.cosine(embedded_vision, embedded_language_target)))
        cosine_distances_pos_neg.append(('pos', scipy.spatial.distance.cosine(embedded_vision, embedded_language_pos)))
        cosine_distances_pos_neg.append(('neg', scipy.spatial.distance.cosine(embedded_vision, embedded_language_neg)))

        euclidean_distances_pos_neg.sort(key=lambda x: x[1])
        cosine_distances_pos_neg.sort(key=lambda x: x[1])

        # TODO: Possibly need to write out ranks of all tested items

        # find rank of closest vision that matches the instance name
        euclid_rank_pos_neg = 3
        for i, (key, distance) in enumerate(euclidean_distances_pos_neg, start=1):
            if key == 'target':
                euclid_rank_pos_neg = i
                break

        cosine_rank_pos_neg = 3
        for i, (key, distance) in enumerate(cosine_distances_pos_neg, start=1):
            if key == 'target':
                cosine_rank_pos_neg = i
                break

        pn_fout.write(f'{euclid_rank_pos_neg} {cosine_rank_pos_neg}\n')

        reciprocal_sum_euclid_pos_neg += 1 / euclid_rank_pos_neg
        reciprocal_sum_cosine_pos_neg += 1 / cosine_rank_pos_neg

        v_to_l_pn_counts[euclid_rank_pos_neg] += 1

        print(f'Pos/Neg {vision[1]}, e: {euclid_rank_pos_neg}, c:{cosine_rank_pos_neg}')

        euclidean_distances_random = []
        cosine_distances_random = []

        random_indexes = random.sample(range(len(language_test_data)), 4)

        language_target = language_test_data[vision_index][0].to(device)
        embedded_language_target = language_model(language_target).cpu().detach().numpy()
        euclidean_distances_random.append(('target', scipy.spatial.distance.euclidean(embedded_vision, embedded_language_target)))
        cosine_distances_random.append(('target', scipy.spatial.distance.cosine(embedded_vision, embedded_language_target)))

        for i in random_indexes:
            rand_fout.write(f'{language_test_data[i][1]} ')
            language_data = language_test_data[i][0].to(device)
            embedded_language = language_model(language_data).cpu().detach().numpy()
            euclidean_distances_random.append(('random', scipy.spatial.distance.euclidean(embedded_vision, embedded_language)))
            cosine_distances_random.append(('random', scipy.spatial.distance.cosine(embedded_vision, embedded_language)))

        euclidean_distances_random.sort(key=lambda x: x[1])
        cosine_distances_random.sort(key=lambda x: x[1])

        # find rank of closest vision that matches the instance name
        euclid_rank_random = 5
        for i, (key, distance) in enumerate(euclidean_distances_random, start=1):
            if key == 'target':
                euclid_rank_random = i
                break

        cosine_rank_random = 5
        for i, (key, distance) in enumerate(cosine_distances_random, start=1):
            if key == 'target':
                cosine_rank_random = i
                break

        rand_fout.write(f'{euclid_rank_random} {cosine_rank_random}\n')

        reciprocal_sum_euclid_random += 1 / euclid_rank_random
        reciprocal_sum_cosine_random += 1 / cosine_rank_random

        v_to_l_rand_counts[euclid_rank_random] += 1

        print(f'Random {vision[1]}, e: {euclid_rank_random}, c:{cosine_rank_random}')

    euclid_mean_reciprocal_rank_pos_neg = reciprocal_sum_euclid_pos_neg / len(language_test_data)
    cosine_mean_reciprocal_rank_pos_neg = reciprocal_sum_cosine_pos_neg / len(language_test_data)

    euclid_mean_reciprocal_rank_random = reciprocal_sum_euclid_random / len(language_test_data)
    cosine_mean_reciprocal_rank_random = reciprocal_sum_cosine_random / len(language_test_data)

    v_to_l_pos_neg, v_to_l_random = euclid_mean_reciprocal_rank_pos_neg, euclid_mean_reciprocal_rank_random

    # Language to Vision
    reciprocal_sum_euclid_pos_neg = 0
    reciprocal_sum_cosine_pos_neg = 0
    reciprocal_sum_euclid_random = 0
    reciprocal_sum_cosine_random = 0
    l_to_v_pn_counts = defaultdict(int)
    l_to_v_rand_counts = defaultdict(int)
    for language_index, language in enumerate(language_test_data):
        pn_fout.write(f'L->V {vision_test_data[language_index][1]} ')
        rand_fout.write(f'L->V {vision_test_data[language_index][1]} ')

        euclidean_distances = []
        cosine_distances = []
        language_data = language[0].to(device)
        embedded_language = language_model(language_data).cpu().detach().numpy()

        #
        # Pos/Neg MRR evaluation
        #

        euclidean_distances_pos_neg = []
        cosine_distances_pos_neg = []

        pos_index = pos_neg_examples[language_index][0]
        neg_index = pos_neg_examples[language_index][1]
  
        pn_fout.write(f'{vision_test_data[pos_index][1]} {vision_test_data[neg_index][1]} ')

        vision_target = vision_test_data[language_index][0].to(device)
        vision_pos = vision_test_data[pos_index][0].to(device)
        vision_neg = vision_test_data[neg_index][0].to(device)
        embedded_vision_target = vision_model(vision_target).cpu().detach().numpy()
        embedded_vision_pos = vision_model(vision_pos).cpu().detach().numpy()
        embedded_vision_neg = vision_model(vision_neg).cpu().detach().numpy()

        euclidean_distances_pos_neg.append(('target', scipy.spatial.distance.euclidean(embedded_language, embedded_vision_target)))
        euclidean_distances_pos_neg.append(('pos', scipy.spatial.distance.euclidean(embedded_language, embedded_vision_pos)))
        euclidean_distances_pos_neg.append(('neg', scipy.spatial.distance.euclidean(embedded_language, embedded_vision_neg)))

        cosine_distances_pos_neg.append(('target', scipy.spatial.distance.cosine(embedded_language, embedded_vision_target)))
        cosine_distances_pos_neg.append(('pos', scipy.spatial.distance.cosine(embedded_language, embedded_vision_pos)))
        cosine_distances_pos_neg.append(('neg', scipy.spatial.distance.cosine(embedded_language, embedded_vision_neg)))

        euclidean_distances_pos_neg.sort(key=lambda x: x[1])
        cosine_distances_pos_neg.sort(key=lambda x: x[1])

        # find rank of closest vision that matches the instance name
        euclid_rank_pos_neg = 3
        for i, (key, distance) in enumerate(euclidean_distances_pos_neg, start=1):
            if key == 'target':
                euclid_rank_pos_neg = i
                break

        cosine_rank_pos_neg = 3
        for i, (key, distance) in enumerate(cosine_distances_pos_neg, start=1):
            if key == 'target':
                cosine_rank_pos_neg = i
                break

        pn_fout.write(f'{euclid_rank_pos_neg} {cosine_rank_pos_neg}\n')

        reciprocal_sum_euclid_pos_neg += 1 / euclid_rank_pos_neg
        reciprocal_sum_cosine_pos_neg += 1 / cosine_rank_pos_neg

        l_to_v_pn_counts[euclid_rank_pos_neg] += 1

        print(f'Pos/Neg {language[1]}, e: {euclid_rank_pos_neg}, c:{cosine_rank_pos_neg}')

        euclidean_distances_random = []
        cosine_distances_random = []

        random_indexes = random.sample(range(len(language_test_data)), 4)

        vision_target = vision_test_data[language_index][0].to(device)
        embedded_vision_target = vision_model(vision_target).cpu().detach().numpy()
        euclidean_distances_random.append(('target', scipy.spatial.distance.euclidean(embedded_language, embedded_vision_target)))
        cosine_distances_random.append(('target', scipy.spatial.distance.cosine(embedded_language, embedded_vision_target)))

        for i in random_indexes:
            rand_fout.write(f'{language_test_data[i][1]} ')
            vision_data = vision_test_data[i][0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()
            euclidean_distances_random.append(('random', scipy.spatial.distance.euclidean(embedded_language, embedded_vision)))
            cosine_distances_random.append(('random', scipy.spatial.distance.cosine(embedded_language, embedded_vision)))

        euclidean_distances_random.sort(key=lambda x: x[1])
        cosine_distances_random.sort(key=lambda x: x[1])

        # find rank of closest vision that matches the instance name
        euclid_rank_random = 5
        for i, (key, distance) in enumerate(euclidean_distances_random, start=1):
            if key == 'target':
                euclid_rank_random = i
                break

        cosine_rank_random = 5
        for i, (key, distance) in enumerate(cosine_distances_random, start=1):
            if key == 'target':
                cosine_rank_random = i
                break

        rand_fout.write(f'{euclid_rank_random} {cosine_rank_random}\n')

        reciprocal_sum_euclid_random += 1 / euclid_rank_random
        reciprocal_sum_cosine_random += 1 / cosine_rank_random

        l_to_v_rand_counts[euclid_rank_random] += 1

        print(f'Random {language[1]}, e: {euclid_rank_random}, c:{cosine_rank_random}')

    euclid_mean_reciprocal_rank_pos_neg = reciprocal_sum_euclid_pos_neg / len(language_test_data)
    cosine_mean_reciprocal_rank_pos_neg = reciprocal_sum_cosine_pos_neg / len(language_test_data)

    euclid_mean_reciprocal_rank_random = reciprocal_sum_euclid_random / len(language_test_data)
    cosine_mean_reciprocal_rank_random = reciprocal_sum_cosine_random / len(language_test_data)

    l_to_v_pos_neg, l_to_v_random = euclid_mean_reciprocal_rank_pos_neg, euclid_mean_reciprocal_rank_random

    print(f'v_to_l_pn_counts: {v_to_l_pn_counts}')
    print(f'v_to_l_rand_counts: {v_to_l_rand_counts}')
    print(f'l_to_v_pn_counts: {l_to_v_pn_counts}')
    print(f'l_to_v_rand_counts: {l_to_v_rand_counts}')

    pn_fout.close()
    rand_fout.close()

    return v_to_l_pos_neg, v_to_l_random, l_to_v_pos_neg, l_to_v_random

def main():
    ARGS, unused = parse_args()

    v_to_l_pos_neg, v_to_l_random, l_to_v_pos_neg, l_to_v_random = evaluate(
        ARGS.experiment_name,
        ARGS.test_data_path,
        ARGS.pos_neg_examples_file,
        ARGS.gpu_num,
        ARGS.embedded_dim
    )

    print(f'V -> L p/n: {v_to_l_pos_neg}')
    print(f'V -> L rand: {v_to_l_random}')
    print(f'L -> V p/n: {l_to_v_pos_neg}')
    print(f'L -> V rand: {l_to_v_random}')

if __name__ == '__main__':
    main()
