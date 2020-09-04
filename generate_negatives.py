import argparse
import pickle
import random

import scipy
import scipy.spatial
import torch
import numpy as np

from datasets import GLData

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='/home/iral/data_processing/uw_data_complete.pkl',
        help='path to GL data dict')
    parser.add_argument('--out_file', default='/home/iral/data_processing/uw_negatives.pkl',
        help='path to output file')

    return parser.parse_known_args()

def get_negative_language(l, data, n=10):
    cosines = []
    for i, v in enumerate(data):
        cosines.append((i, scipy.spatial.distance.cosine(l, v)))
    cosines.sort(key=lambda x: x[1])

    negative = random.choice(cosines[:n])

    return negative[0], data[negative[0]]

def get_pos_neg_examples(language, language_data, object_names):
    """
    language: the language to find negatives of
    language_data: the set of all language features
    object_names: the set of all object_names
    n: the top n negatives to choose from
    k: sample size to return

    returns the indices in language_data of the negatives as a list
    """
    positive_index = 0
    for i, vector in enumerate(language_data):
        # .data[0] necessary to compare on cuda
        if torch.equal(language, vector):
            positive_index = i
    index = positive_index
    while (positive_index == index):
        positive_index = np.random.choice(
            [i for i, x in enumerate(object_names) if x == object_names[index]])
    negative_label = np.random.choice(list(set(object_names) - set([object_names[index]])))
    negative_index = np.random.choice([i for i, x in enumerate(object_names) if x == negative_label])
    # choose randomly for top n negative examples
    # this returns indexes
    return positive_index, negative_index

def main():
    ARGS, unused = parse_args()

    print(f'Reading from {ARGS.data_file}')
    print(f'Writing to {ARGS.out_file}')

    negatives = []

    with open(ARGS.data_file, 'rb') as fin:
        data = pickle.load(fin)
    
    language_data, object_names = [l for l, _, _, _ in data], [o for _, _, o, _ in data]
    for i, language in enumerate(language_data):
        if i % 100 == 0:
            print(f'Calculating {i}/{len(language_data)}')
        negatives.append(get_pos_neg_examples(language, language_data, object_names))

    with open(ARGS.out_file, 'wb') as fout:
        pickle.dump(negatives, fout)

    print(f'Wrote negatives to file {ARGS.out_file}')

if __name__ == '__main__':
    main()
