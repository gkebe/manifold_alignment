import argparse
import pickle
import random

import scipy
import scipy.spatial
import torch

from manifold_alignment.datasets import GLData

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

def get_pos_neg_examples(language, language_data):
    """
    language: the language to find negatives of
    language_data: the set of all language features
    n: the top n negatives to choose from
    k: sample size to return

    returns the indices in language_data of the negatives as a list
    """
    cosines = []
    for i, vector in enumerate(language_data):
        # .data[0] necessary to compare on cuda
        if torch.equal(language, vector):
            continue
        cosines.append((i, scipy.spatial.distance.cosine(language, vector)))
    cosines.sort(key=lambda x: x[1])

    # choose randomly for top n negative examples
    # this returns indexes
    positive_index = cosines[-1][0]
    negative_index = cosines[0][0]

    return positive_index, negative_index

def main():
    ARGS, unused = parse_args()

    print(f'Reading from {ARGS.data_file}')
    print(f'Writing to {ARGS.out_file}')

    negatives = []

    with open(ARGS.data_file, 'rb') as fin:
        data = pickle.load(fin)
        language_data = [l for l, _, _, _ in data]
        for i, language in enumerate(language_data):
            if i % 100 == 0:
                print(f'Calculating {i}/{len(language_data)}')
            negatives.append(get_pos_neg_examples(language, language_data))

    with open(ARGS.out_file, 'wb') as fout:
        pickle.dump(negatives, fout)

    print(f'Wrote negatives to file {ARGS.out_file}')

if __name__ == '__main__':
    main()
