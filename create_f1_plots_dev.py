import argparse
import os
import statistics

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment name')
    parser.add_argument('--n', type=int, default=20,
        help='number of bins 0 to 1 for threshold')

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

            y_true.append(True if pn == 'p' else False)
            distances.append(float(dist))

    normalized_distances = [d / max(distances) for d in distances]
    print(f'min n_dist = {min(normalized_distances)}; max n_dist = {max(normalized_distances)}')
    thresholds = [t / n for t in range(n + 1)]
    for threshold in thresholds:
        print(f'Calculating for threshold = {threshold}')
        y_pred = []
        for nd in normalized_distances:
            if nd < threshold:
                y_pred.append(True)
            else:
                y_pred.append(False)
        
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=1)
        print(f'p: {p}, r: {r}, f: {f}, s:{s}, t:{len(y_pred)}')
        precision.append(p)
        recall.append(r)
        f1.append(f)

    print(thresholds)
    print(precision)
    print(recall)
    print(f1)
    p_line = plt.plot(thresholds, precision, 'b', label='Precision')
    r_line = plt.plot(thresholds, recall, 'r', label='Recall')
    f_line = plt.plot(thresholds, f1, 'm', label='F1-Score')
    plt.title(title)
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall/F1')
    plt.legend()    

    plt.savefig(fout)

def main():
    ARGS, unused = parse_args()

    results_dir = f'./output/{ARGS.experiment}'

    #l2l = os.path.join(results_dir, 'language2language.txt')
    #l2l_out = os.path.join(results_dir, 'l2l.png')
    #v2v = os.path.join(results_dir, 'vision2vision.txt')
    #v2v_out = os.path.join(results_dir, 'v2v.png')
    v2l = os.path.join(results_dir, 'vision2language_dev_epoch299.txt')
    v2l_out = os.path.join(results_dir, 'v2l_p_r_f1.png')

    #create_plot(l2l, l2l_out, 'Language to Language Embedded Cosine Distance')
    #create_plot(v2v, v2v_out, 'Vision to Vision Embedded Cosine Distance')
    create_plot(ARGS.n, v2l, v2l_out, 'Vision to Language Precision/Recall/F1 by Threshold')

if __name__ == '__main__':
    main()
