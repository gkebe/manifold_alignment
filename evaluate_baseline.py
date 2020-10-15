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

def evaluate(test_data_path, pos_neg_examples_file):

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



    with open(pos_neg_examples_file, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)


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

        euclid_rank_pos_neg = random.choice([1,2,3])
        cosine_rank_pos_neg = random.choice([1,2,3])
        pn_fout.write(f'{euclid_rank_pos_neg} {cosine_rank_pos_neg}\n')

        reciprocal_sum_euclid_pos_neg += 1 / euclid_rank_pos_neg
        reciprocal_sum_cosine_pos_neg += 1 / cosine_rank_pos_neg

        euclid_rank_random = random.choice([1,2,3,4,5])
        cosine_rank_random = random.choice([1,2,3,4,5])
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
        euclid_rank_pos_neg = random.choice([1,2,3])
        cosine_rank_pos_neg = random.choice([1,2,3])

        pn_fout.write(f'{euclid_rank_pos_neg} {cosine_rank_pos_neg}\n')

        reciprocal_sum_euclid_pos_neg += 1 / euclid_rank_pos_neg
        reciprocal_sum_cosine_pos_neg += 1 / cosine_rank_pos_neg

        euclid_rank_random = random.choice([1,2,3,4,5])
        cosine_rank_random = random.choice([1,2,3,4,5])

        rand_fout.write(f'{euclid_rank_random} {cosine_rank_random}\n')

        reciprocal_sum_euclid_random += 1 / euclid_rank_random
        reciprocal_sum_cosine_random += 1 / cosine_rank_random

        l_to_v_rand_counts[euclid_rank_random] += 1

        print(f'Random {language[1]}, e: {euclid_rank_random}, c:{cosine_rank_random}')

    euclid_mean_reciprocal_rank_pos_neg = reciprocal_sum_euclid_pos_neg / len(language_test_data)
    cosine_mean_reciprocal_rank_pos_neg = reciprocal_sum_cosine_pos_neg / len(language_test_data)

    euclid_mean_reciprocal_rank_random = reciprocal_sum_euclid_random / len(language_test_data)
    cosine_mean_reciprocal_rank_random = reciprocal_sum_cosine_random / len(language_test_data)

    l_to_v_pos_neg, l_to_v_random = cosine_mean_reciprocal_rank_pos_neg, cosine_mean_reciprocal_rank_random

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
        ARGS.test_data_path,
        ARGS.pos_neg_examples_file
    )

    print(f'V -> L p/n: {v_to_l_pos_neg}')
    print(f'V -> L rand: {v_to_l_random}')
    print(f'L -> V p/n: {l_to_v_pos_neg}')
    print(f'L -> V rand: {l_to_v_random}')

if __name__ == '__main__':
    main()
