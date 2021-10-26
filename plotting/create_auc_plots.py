import argparse
import os
import statistics
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment name')
    parser.add_argument('--n', type=int, default=20,
        help='number of bins 0 to 1 for threshold')
    parser.add_argument('--epoch', type=int, default=299,
        help='Epoch to plot')
    parser.add_argument('--test', dest='test', action='store_true',
        help="Use test split instead of dev split.")

    return parser.parse_known_args()

def create_plot(n, file_path, fout, title):
    distances = []
    y_true = []
    precision = []
    recall = []
    f1 = []

    with open(file_path, 'r') as fin:
        headers = fin.readline() 
        for line in fin:
            instance_1, instance_2, pn, dist = line.strip().split(',')

            y_true.append(1 if pn == 'p' else 0)
            distances.append(float(dist))

    normalized_distances = [1 - (d / 2) for d in distances]
    fpr, tpr, thresholds = roc_curve(y_true, normalized_distances)
    auc = roc_auc_score(y_true, normalized_distances)    
    pickle.dump([thresholds, fpr, tpr, auc], open(fout, "wb"))

def main():
    ARGS, unused = parse_args()

    results_dir = f'../output/{ARGS.experiment}'
    split = "dev"
    if ARGS.test:
        split = "test"
    v2l = os.path.join(results_dir, f'vision2language_{split}_epoch_{ARGS.epoch}.txt')
    v2l_out = os.path.join(results_dir, f'{ARGS.experiment}_auc.pkl')
    
    create_plot(ARGS.n, v2l, v2l_out, 'Vision to Language Precision/Recall/F1 by Threshold')

if __name__ == '__main__':
    main()
