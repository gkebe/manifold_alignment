import argparse
import os
import statistics

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment name')
    parser.add_argument('--epoch', type=int, default=299,
        help='Epoch to plot')
    parser.add_argument('--test', dest='test', action='store_true',
        help="Use test split instead of dev split.")
    return parser.parse_known_args()

def create_plot(file_path, fout, title):
    pos_distances = []
    neg_distances = []
    with open(file_path, 'r') as fin:
        headers = fin.readline()
        for line in fin:
            instance_1, instance_2, pn, dist = line.strip().split(',')
            if pn == 'p' and dist != 0.0:
                pos_distances.append(float(dist))
            else:
                neg_distances.append(float(dist))

    print(f'pos mean: {statistics.mean(pos_distances)}')
    print(f'pos stdev: {statistics.stdev(pos_distances)}')
    print(f'neg mean: {statistics.mean(neg_distances)}')
    print(f'neg stdev: {statistics.stdev(neg_distances)}')

    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.xlabel('Cosine Distance')
    plt.ylabel('Instance Class')

    ax.boxplot(
        [pos_distances, neg_distances],
        labels=['P', 'N'],
        vert=False
    )
    
    fig.savefig(fout)

def main():
    ARGS, unused = parse_args()

    results_dir = f'../output/{ARGS.experiment}'

    split = "dev"
    if ARGS.test:
        split = "test"

    l2l = os.path.join(results_dir, f'language2language_{split}_epoch_{ARGS.epoch}.txt')
    l2l_out = os.path.join(results_dir, 'l2l.png')
    v2v = os.path.join(results_dir, f'vision2vision_{split}_epoch_{ARGS.epoch}.txt')
    v2v_out = os.path.join(results_dir, 'v2v.png')
    v2l = os.path.join(results_dir, f'vision2language_{split}_epoch_{ARGS.epoch}.txt')
    v2l_out = os.path.join(results_dir, 'v2l.png')

    create_plot(l2l, l2l_out, 'Language to Language Embedded Cosine Distance')
    create_plot(v2v, v2v_out, 'Vision to Vision Embedded Cosine Distance')
    create_plot(v2l, v2l_out, 'Vision to Language Embedded Cosine Distance')

if __name__ == '__main__':
    main()
