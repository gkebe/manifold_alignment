import argparse
import os
import pickle
import random
from collections import defaultdict

import scipy
import scipy.spatial
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from datasets import GLData
from rownet import RowNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='name of experiment to test')
    parser.add_argument('--test_data_path', help='path to testing data')
    parser.add_argument('--threshold', required=True, type=float,
        help='embedded_dim')
    parser.add_argument('--pos_neg_examples_file', default=None,
        help='path to examples pkl')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='embedded_dim')

    return parser.parse_known_args()

def evaluate(experiment_name, test_data_path, pos_neg_examples_file, gpu_num, embedded_dim, threshold):

    pn_fout = open('./pn_eval_output.txt', 'w')
    rand_fout = open('./rand_eval_output.txt', 'w')
    with open(test_data_path, 'rb') as fin:
        test_data = pickle.load(fin)

    language_test_data = [(l, i) for l, _, _, i, _ in test_data]
    vision_test_data = [(v, i) for _, v, _, i, _ in test_data]

    # BERT dimension
    language_dim = list(language_test_data[0][0].size())[0]
    # Eitel dimension
    vision_dim = list(vision_test_data[0][0].size())[0]

    results_dir = f'./output/{experiment_name}'
    train_results_dir = os.path.join(results_dir, 'train_results/')
    v2l = os.path.join(results_dir, 'vision2language_test_epoch_299.txt')
    y_true = {}
    distances = {}
    y_pred = {}
    with open(v2l, 'r') as fin:
        headers = fin.readline() 
        for line in fin:
            instance_1, instance_2, pn, dist = line.strip().split(',')
            if instance_1 not in y_true:
                y_true[instance_1] = []
                y_pred[instance_1] = []
                distances[instance_1] = []

            y_true[instance_1].append(True if pn == 'p' else False)
            distances[instance_1].append(float(dist))

    normalized_distances = {k:[d / 2 for d in v] for k,v in distances}
    for k in normalized_distances:
        for nd in normalized_distances[k]:
          if nd < threshold:
            y_pred[k].append(True)
          else:
            y_pred[k].append(False)
        print(k)
        p, r, f, s = precision_recall_fscore_support(y_true[k], y_pred[k], average='binary', zero_division=1)
        print(f"F1 score: {f}")
        print()


def main():
    ARGS, unused = parse_args()

    evaluate(
        ARGS.experiment_name,
        ARGS.test_data_path,
        ARGS.pos_neg_examples_file,
        ARGS.gpu_num,
        ARGS.embedded_dim,
        threshold=ARGS.threshold
    )

if __name__ == '__main__':
    main()
