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
import numpy as np

from datasets import GLData
from rownet import RowNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='name of experiment to test')
    parser.add_argument('--threshold', required=True, type=float,
        help='embedded_dim')

    return parser.parse_known_args()

def evaluate(experiment_name, threshold):
   fs = dict()
   for i in range(1,6):    
        results_dir = f'./output/{experiment_name}_{i}_bias'
        train_results_dir = os.path.join(results_dir, 'train_results/')
        v2l = os.path.join(results_dir, 'vision2language_test_epoch_220.txt')
        y_true = {}
        distances = {}
        y_pred = {}
        with open(v2l, 'r') as fin:
            headers = fin.readline() 
            for line in fin:
                instance_1, instance_2, pn, dist = line.strip().split(',')
                if instance_1[:-2] not in y_true:
                    y_true[instance_1[:-2]] = []
                    y_pred[instance_1[:-2]] = []
                    distances[instance_1[:-2]] = []

                y_true[instance_1[:-2]].append(True if pn == 'p' else False)
                distances[instance_1[:-2]].append(float(dist))

        normalized_distances = {k:[d / 2 for d in v] for k,v in distances.items()}
    
        for k in normalized_distances:
            for nd in normalized_distances[k]:
                if nd < threshold:
                   y_pred[k].append(True)
                else:
                   y_pred[k].append(False)
            
            p, r, f, s = precision_recall_fscore_support(y_true[k], y_pred[k], average='binary', zero_division=1)
            if k not in fs:
                fs[k] = []
            fs[k].append(f)
   print("\n".join([f"{k}" for k,v in fs.items()]))
   print("\n".join([f"{np.mean(v)}" for k,v in fs.items()]))
def main():
    ARGS, unused = parse_args()

    evaluate(
        ARGS.experiment_name,
        threshold=ARGS.threshold
    )

if __name__ == '__main__':
    main()

