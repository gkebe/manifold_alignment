import argparse
from collections import defaultdict
import os
import pickle
import random

import scipy
import scipy.spatial
import torch

from datasets import GLData
from lstm import LSTM
from rnn import RNN
from rownet import RowNet
from sklearn.metrics import precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='name of experiment to evaluate')
    parser.add_argument('--threshold', required=True, type=float,
        help='embedded_dim')
    parser.add_argument('--test_data_path', help='path to test data')
    parser.add_argument('--pos_neg_examples_file',
        help='path to examples pkl')
    parser.add_argument('--num_layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--embedded_dim', type=int, default=1024,
        help='embedded_dim')
    parser.add_argument('--awe', type=int, default=32,
        help='awe')
    parser.add_argument('--epoch', type=int, default=299,
        help='F1 at epoch')
    return parser.parse_known_args()

def evaluate(experiment, test_path, pos_neg_examples, num_layers, gpu_num, embedded_dim, awe, threshold, epoch):
    pn_fout = open('./pn_eval_output.txt', 'w')
    rand_fout = open('./rand_eval_output.txt', 'w')

    with open(pos_neg_examples, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)

    with open(test_path, 'rb') as fin:
        test_data = pickle.load(fin)

    speech_test_data = [(s, i) for s, _, _, i, _ in test_data]
    vision_test_data = [(v, i) for _, v, _, i, _ in test_data]
    instance_names = [i for _, _, _, i, _ in test_data]
    object_names = [o for _, _, o, _, _ in test_data]

    print(len(speech_test_data))
    print(len(vision_test_data))

    vision_dim = list(vision_test_data[0][0].size())[0]

    results_dir = f'./output/{experiment}'
    train_results_dir = os.path.join(results_dir, 'train_results/')
    v2l = os.path.join(results_dir, f'vision2language_test_epoch_{epoch}.txt')
    y_true = []
    distances = []
    y_pred = []
    with open(v2l, 'r') as fin:
        headers = fin.readline() 
        for line in fin:
            instance_1, instance_2, pn, dist = line.strip().split(',')

            y_true.append(True if pn == 'p' else False)
            distances.append(float(dist))

    normalized_distances = [d / 2 for d in distances]
    for nd in normalized_distances:
      if nd < threshold:
        y_pred.append(True)
      else:
        y_pred.append(False)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=1)
    print(f"F1 score: {f}")

    print(f'is_available(): {torch.cuda.is_available()}')
    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    speech_model = LSTM(
        input_size=13,
        output_size=embedded_dim,
        hidden_dim=64,
        num_layers=num_layers,
        dropout=0.0,
        device=device,
        awe=awe
    )
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)

    speech_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'speech_model.pt'), map_location=device))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'vision_model.pt'), map_location=device))
    speech_model.to(device)
    vision_model.to(device)
    speech_model.eval()
    vision_model.eval()

    # Speech to Vision
    reciprocal_sum_pn = 0
    reciprocal_sum_rand = 0
    s_to_v_pn_counts = defaultdict(int)
    s_to_v_rand_counts = defaultdict(int)
    for speech_index, speech in enumerate(speech_test_data):
        pn_fout.write(f'S->V,')
        rand_fout.write(f'S->V,')

        speech_data = speech[0].to(device)
        # speech_data = speech_data.permute(0, 2, 1)
        embedded_speech = speech_model(speech_data).cpu().detach().numpy()

        #
        # Pos/Neg MRR evaluation
        #

        cosine_distances_pn = []

        pos_index = pos_neg_examples[speech_index][0]
        neg_index = pos_neg_examples[speech_index][1]

        vision_target = vision_test_data[speech_index][0].to(device)
        vision_pos = vision_test_data[pos_index][0].to(device)
        vision_neg = vision_test_data[neg_index][0].to(device)
        embedded_vision_target = vision_model(vision_target).cpu().detach().numpy()
        embedded_vision_pos = vision_model(vision_pos).cpu().detach().numpy()
        embedded_vision_neg = vision_model(vision_neg).cpu().detach().numpy()

        cosine_distances_pn.append(('target', scipy.spatial.distance.cosine(embedded_speech, embedded_vision_target), speech[1]))
        cosine_distances_pn.append(('pos', scipy.spatial.distance.cosine(embedded_speech, embedded_vision_pos), vision_test_data[pos_index][1]))
        cosine_distances_pn.append(('neg', scipy.spatial.distance.cosine(embedded_speech, embedded_vision_neg), vision_test_data[neg_index][1]))

        cosine_distances_pn.sort(key=lambda x: x[1])
        rank_pn = len(cosine_distances_pn)
        for i, (key, distance, instance_name) in enumerate(cosine_distances_pn, start=1):
            pn_fout.write(f'{instance_name},{distance},')
            if key == 'target':
                rank_pn = i

        pn_fout.write(f'{rank_pn}\n')

        reciprocal_sum_pn += 1 / rank_pn
        s_to_v_pn_counts[rank_pn] += 1

        #
        # Random MRR evaluation
        #

        cosine_distances_rand = []
        random_indexes = random.sample(range(len(vision_test_data)), 4)

        cosine_distances_rand.append(('target', scipy.spatial.distance.cosine(embedded_speech, embedded_vision_target), speech[1]))

        for i in random_indexes:
            vision_data = vision_test_data[i][0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()
            cosine_distances_rand.append(('random', scipy.spatial.distance.cosine(embedded_speech, embedded_vision), vision_test_data[i][1]))

        cosine_distances_rand.sort(key=lambda x: x[1])
        rank_rand = len(cosine_distances_rand)
        for i, (key, distance, instance_name) in enumerate(cosine_distances_rand, start=1):
            rand_fout.write(f'{instance_name},{distance},')
            if key == 'target':
                rank_rand = i

        rand_fout.write(f'{rank_rand}\n')

        reciprocal_sum_rand += 1 / rank_rand
        s_to_v_rand_counts[rank_rand] += 1

        s_to_v_mrr_pn = reciprocal_sum_pn / len(vision_test_data)
        s_to_v_mrr_rand = reciprocal_sum_rand / len(vision_test_data)

#    print(f'V->S counts pn: {dict(v_to_s_pn_counts)}')
#    print(f'V->S counts rand: {dict(v_to_s_rand_counts)}')
#    print(f'S->V counts pn: {dict(s_to_v_pn_counts)}')
#    print(f'S->V counts rand: {dict(s_to_v_rand_counts)}')
    pn_fout.close()
    rand_fout.close()

    return s_to_v_mrr_pn, s_to_v_mrr_rand

def main():
    ARGS, unused = parse_args()

    s_to_v_mrr_pn, s_to_v_mrr_rand = evaluate(
        ARGS.experiment,
        ARGS.test_data_path,
        ARGS.pos_neg_examples_file,
        ARGS.num_layers,
        ARGS.gpu_num,
        ARGS.embedded_dim,
        ARGS.awe,
        ARGS.threshold,
        ARGS.epoch
    )

    print(f'Triplet MRR: {s_to_v_mrr_pn}')
    print(f'Subset MRR: {s_to_v_mrr_rand}')

if __name__ == '__main__':
    main()