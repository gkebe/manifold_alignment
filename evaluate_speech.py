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
from attention import Combiner, SmarterAttentionNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='name of experiment to evaluate')
    parser.add_argument('--test_data_path', help='path to test data')
    parser.add_argument('--pos_neg_examples_file',
        help='path to examples pkl')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--embedded_dim', type=int, default=1024,
        help='embedded_dim')
    parser.add_argument("--lstm",
                        action='store_true',
                        help="Whether to use a lstm.")
    parser.add_argument('--num_layers', type=int, default=1,
        help='number of lstm hidden layers')
    parser.add_argument("--mean_pooling",
                        action='store_true',
                        help="Whether to use mean pooling on the lstm's output.")

    return parser.parse_known_args()

def evaluate(experiment, test_path, pos_neg_examples, num_layers, gpu_num, embedded_dim, lstm=False, mean_pooling=False):
    pn_fout = open('./pn_eval_output.txt', 'w')
    rand_fout = open('./rand_eval_output.txt', 'w')

    with open(pos_neg_examples, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)

    with open(test_path, 'rb') as fin:
        test_data = pickle.load(fin)

    speech_test_data = [(s, i) for s, _, _, i in test_data]
    vision_test_data = [(v, i) for _, v, _, i in test_data]
    instance_names = [i for _, _, _, i in test_data]
    object_names = [o for _, _, o, _ in test_data]

    print(len(speech_test_data))
    print(len(vision_test_data))

    vision_dim = list(vision_test_data[0][0].size())[0]

    results_dir = f'./output/{experiment}'
    train_results_dir = os.path.join(results_dir, 'train_results/')

    print(f'is_available(): {torch.cuda.is_available()}')
    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if lstm:
        speech_model = LSTM(
            input_size=list(speech_test_data[0][0].size())[1],
            output_size=embedded_dim,
            hidden_dim=list(speech_test_data[0][0].size())[1],
            num_layers=num_layers,
            mean_pooling=mean_pooling,
            device=device,

        )
    else:
#        speech_model = Combiner(list(speech_test_data[0][0].size())[1], embedded_dim)
        speech_model = SmarterAttentionNet(list(speech_test_data[0][0].size())[1], embedded_dim, embedded_dim)
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)

    speech_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt'), map_location=device))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt'), map_location=device))
    speech_model.to(device)
    vision_model.to(device)
    speech_model.eval()
    vision_model.eval()

    # Vision to Speech
    reciprocal_sum_pn = 0
    reciprocal_sum_rand = 0
    v_to_s_pn_counts = defaultdict(int)
    v_to_s_rand_counts = defaultdict(int)
    for vision_index, vision in enumerate(vision_test_data):
        pn_fout.write(f'V->S,')
        rand_fout.write(f'V->S,')

        vision_data = vision[0].to(device)
        embedded_vision = vision_model(vision_data).cpu().detach().numpy()

        #
        # Pos/Neg MRR evaluation
        #

        cosine_distances_pn = []

        pos_index = pos_neg_examples[vision_index][0]
        neg_index = pos_neg_examples[vision_index][1]

        speech_target = torch.unsqueeze(speech_test_data[vision_index][0], 0).to(device)
        #print(speech_target.size())
        speech_pos = torch.unsqueeze(speech_test_data[pos_index][0], 0).to(device)
        speech_neg = torch.unsqueeze(speech_test_data[neg_index][0], 0).to(device)

        # TODO: THIS SHOULD BE HANDLED WHEN CREATING THE FEATURES
        # speech_target = speech_target.permute(0, 2, 1)
        # speech_pos = speech_pos.permute(0, 2, 1)
        # speech_neg = speech_neg.permute(0, 2, 1)

        embedded_speech_target = speech_model(speech_target).cpu().detach().numpy()
        embedded_speech_pos = speech_model(speech_pos).cpu().detach().numpy()
        embedded_speech_neg = speech_model(speech_neg).cpu().detach().numpy()

        cosine_distances_pn.append(('target', scipy.spatial.distance.cosine(embedded_vision, embedded_speech_target), vision[1]))
        cosine_distances_pn.append(('pos', scipy.spatial.distance.cosine(embedded_vision, embedded_speech_pos), speech_test_data[pos_index][1]))
        cosine_distances_pn.append(('neg', scipy.spatial.distance.cosine(embedded_vision, embedded_speech_neg), speech_test_data[neg_index][1]))

        cosine_distances_pn.sort(key=lambda x: x[1])
        rank_pn = len(cosine_distances_pn)
        for i, (key, distance, instance_name) in enumerate(cosine_distances_pn, start=1):
            pn_fout.write(f'{instance_name},{distance},')
            if key == 'target':
                rank_pn = i

        pn_fout.write(f'{rank_pn}\n')

        reciprocal_sum_pn += 1 / rank_pn
        v_to_s_pn_counts[rank_pn] += 1

        #
        # Random MRR evaluation
        #

        cosine_distances_rand = []
        random_indexes = random.sample(range(len(speech_test_data)), 4)

        cosine_distances_rand.append(('target', scipy.spatial.distance.cosine(embedded_vision, embedded_speech_target), vision[1]))

        for i in random_indexes:
            speech_data = torch.unsqueeze(speech_test_data[i][0], 0).to(device)
            # TODOtorch. unsqueeze (input, dim): SHOULD BE HANDLED WHEN CREATING FEATURES
            # speech_data = speech_data.permute(0, 2, 1)
            embedded_speech = speech_model(speech_data).cpu().detach().numpy()
            cosine_distances_rand.append(('random', scipy.spatial.distance.cosine(embedded_vision, embedded_speech), speech_test_data[i][1]))

        cosine_distances_rand.sort(key=lambda x: x[1])
        rank_rand = len(cosine_distances_rand)
        for i, (key, distance, instance_name) in enumerate(cosine_distances_rand, start=1):
            rand_fout.write(f'{instance_name},{distance},')
            if key == 'target':
                rank_rand = i

        rand_fout.write(f'{rank_rand}\n')

        reciprocal_sum_rand += 1 / rank_rand
        v_to_s_rand_counts[rank_rand] += 1


        v_to_s_mrr_pn = reciprocal_sum_pn / len(speech_test_data)
        v_to_s_mrr_rand = reciprocal_sum_rand / len(speech_test_data)

    # Speech to Vision
    reciprocal_sum_pn = 0
    reciprocal_sum_rand = 0
    s_to_v_pn_counts = defaultdict(int)
    s_to_v_rand_counts = defaultdict(int)
    for speech_index, speech in enumerate(speech_test_data):
        pn_fout.write(f'S->V,')
        rand_fout.write(f'S->V,')

        speech_data = torch.unsqueeze(speech[0], 0).to(device)
        # TODO: NEEDS TO BE HANDLED WHEN CREATING FEATURES
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

    print(f'V->S counts pn: {dict(v_to_s_pn_counts)}')
    print(f'V->S counts rand: {dict(v_to_s_rand_counts)}')
    print(f'S->V counts pn: {dict(s_to_v_pn_counts)}')
    print(f'S->V counts rand: {dict(s_to_v_rand_counts)}')
    pn_fout.close()
    rand_fout.close()

    return v_to_s_mrr_pn, v_to_s_mrr_rand, s_to_v_mrr_pn, s_to_v_mrr_rand

def main():
    ARGS, unused = parse_args()
    
    v_to_s_mrr_pn, v_to_s_mrr_rand, s_to_v_mrr_pn, s_to_v_mrr_rand = evaluate(
        experiment=ARGS.experiment,
        test_path=ARGS.test_data_path,
        pos_neg_examples=ARGS.pos_neg_examples_file,
        num_layers=ARGS.num_layers,
        gpu_num=ARGS.gpu_num,
        embedded_dim=ARGS.embedded_dim,
        lstm=ARGS.lstm,
        mean_pooling=ARGS.mean_pooling
    )

    print(f'V->S p/n: {v_to_s_mrr_pn}')
    print(f'V->S rand: {v_to_s_mrr_rand}')
    print(f'S->V p/n: {s_to_v_mrr_pn}')
    print(f'S->V rand: {s_to_v_mrr_rand}')

if __name__ == '__main__':
    main()
