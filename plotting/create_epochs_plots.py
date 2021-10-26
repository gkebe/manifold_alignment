# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:14:47 2020

@author: T530
"""

import argparse
import glob, os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import pickle
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment name')
    parser.add_argument('--threshold', type=float, default=0.45,
        help='threshold between 0 and 1')
    parser.add_argument('--test', dest='test', action='store_true',
        help="Use test split instead of dev split.")
    return parser.parse_known_args()
def create_plot(threshold, file_path, fout, title, split):
    files = []
    for epoch_ in range(len(glob.glob(file_path + f"/vision2language_{split}_*.txt"))):
        epoch_file = file_path + f"/vision2language_{split}_epoch_{epoch_}.txt"
        files.append(os.path.join(epoch_file))
    distances = []
    precision = []
    recall = []
    f1 = []
    for epoch_file in files:
        print(epoch_file)
        distances = []
        y_true = []
        with open(epoch_file, 'r') as fin:
            headers = fin.readline()
            for line in fin:
                instance_1, instance_2, pn, dist = line.strip().split(',')

                y_true.append(True if pn == 'p' else False)
                distances.append(float(dist))

        normalized_distances = [d/2 for d in distances]
        print(f'min n_dist = {min(normalized_distances)}; max n_dist = {max(normalized_distances)}')
        print(f'Calculating for threshold = {threshold}')
        y_pred = []
        for nd in normalized_distances:
            if nd < threshold:
                y_pred.append(True)
            else:
                y_pred.append(False)

        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print(f'p: {p}, r: {r}, f: {f}')
        precision.append(p)
        recall.append(r)
        f1.append(f)

    epochs = [i+1 for i in range(len(files))]
    print(epochs)
    print(precision)
    print(recall)
    print(f1)
    pickle.dump([epochs, precision, recall, f1], open(fout, "wb"))
    # p_line = plt.plot(epochs, precision, 'b', label='Precision')
    # r_line = plt.plot(epochs, recall, 'r', label='Recall')
    # f_line = plt.plot(epochs, f1, 'm', label='F1-Score')
    # plt.title(title)
    # plt.xlabel('Epochs')
    # plt.ylabel('Precision/Recall/F1')
    # plt.legend()
    #
    # plt.savefig(fout+"_epochs.png")
def main():
    ARGS, unused = parse_args()

    results_dir = f'../output/{ARGS.experiment}'
    split = "dev"
    if ARGS.test:
        split = "test"
    #l2l = os.path.join(results_dir, 'language2language.txt')
    #l2l_out = os.path.join(results_dir, 'l2l.png')
    #v2v = os.path.join(results_dir, 'vision2vision.txt')
    #v2v_out = os.path.join(results_dir, 'v2v.png')
    v2l_out = os.path.join(results_dir, f'{ARGS.experiment}_epochs.pkl')

    #create_plot(l2l, l2l_out, 'Language to Language Embedded Cosine Distance')
    #create_plot(v2v, v2v_out, 'Vision to Vision Embedded Cosine Distance')
    create_plot(ARGS.threshold, results_dir, v2l_out, 'Vision to Language Precision/Recall/F1 by Epoch', split)

if __name__ == '__main__':
    main()
