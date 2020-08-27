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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='name of experiment to evaluate')
    parser.add_argument('--test_data_path', help='path to test data')
    parser.add_argument('--pos_neg_examples_file',
        help='path to examples pkl')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--embedded_dim', type=int, default=1024,
        help='embedded_dim')

    return parser.parse_known_args()

def evaluate(experiment, test_path, pos_neg_examples, gpu_num, embedded_dim):
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

    speech_model = LSTM(
        input_size=40,
        output_size=embedded_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        device=device
    )
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
        break
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

        speech_target = speech_test_data[vision_index][0].to(device)
        #print(speech_target.size())
        speech_pos = speech_test_data[pos_index][0].to(device)
        speech_neg = speech_test_data[neg_index][0].to(device)

        # TODO: THIS SHOULD BE HANDLED WHEN CREATING THE FEATURES
        speech_target = speech_target.permute(0, 2, 1)
        speech_pos = speech_pos.permute(0, 2, 1)
        speech_neg = speech_neg.permute(0, 2, 1)

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
            speech_data = speech_test_data[i][0].to(device)
            # TODO: SHOULD BE HANDLED WHEN CREATING FEATURES
            speech_data = speech_data.permute(0, 2, 1)
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
        break
        pn_fout.write(f'S->V,')
        rand_fout.write(f'S->V,')

        speech_data = speech[0].to(device)
        # TODO: NEEDS TO BE HANDLED WHEN CREATING FEATURES
        speech_data = speech_data.permute(0, 2, 1)
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

    speech2speech_fout = open('speech2speech.txt', 'w')
    speech2speech_fout.write('instance_name_1,instance_name_2,embedded_distance\n')
    for speech_index_i, speech_i in enumerate(speech_test_data):
        for speech_index_j, speech_j in enumerate(speech_test_data):
            speech_data_i = speech_i[0].to(device)
            # TODO: NEEDS TO BE HANDLED WHEN CREATING FEATURES
            speech_data_i = speech_data_i.permute(0, 2, 1)
            embedded_speech_i = speech_model(speech_data_i).cpu().detach().numpy()
            
            speech_data_j = speech_j[0].to(device)
            speech_data_j = speech_data_j.permute(0, 2, 1)
            embedded_speech_j = speech_model(speech_data_j).cpu().detach().numpy()

            dist = scipy.spatial.distance.cosine(embedded_speech_i, embedded_speech_j)
            
            speech2speech_fout.write(f'{instance_names[speech_index_i]},{instance_names[speech_index_j]},{dist}\n')
    speech2speech_fout.close()

    vision2vision_fout = open('vision2vision.txt', 'w')
    vision2vision_fout.write('instance_name_1,instance_name_2,embedded_distance\n')
    for vision_index_i, vision_i in enumerate(vision_test_data):
        for vision_index_j, vision_j in enumerate(vision_test_data):
            vision_data_i = vision_i[0].to(device) 
            embedded_vision_i = vision_model(vision_data_i).cpu().detach().numpy()

            vision_data_j = vision_j[0].to(device)
            embedded_vision_j = vision_model(vision_data_j).cpu().detach().numpy()
            
            dist = scipy.spatial.distance.cosine(embedded_vision_i, embedded_vision_j)

            vision2vision_fout.write(f'{instance_names[vision_index_i]},{instance_names[vision_index_j]},{dist}\n')
    vision2vision_fout.close()

    vision2speech_fout = open('vision2speech.txt', 'w')
    vision2speech_fout.write('vision_instance,speech_instance,embedded_distance\n')
    for vision_index, vision in enumerate(vision_test_data):
        for speech_index, speech in enumerate(speech_test_data):
            vision_data = vision[0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()

            speech_data = speech[0].to(device)
            speech_data = speech_data.permute(0, 2, 1)
            embedded_speech = speech_model(speech_data).spu().detach().numpy()

            dist = scipy.spatial.distance.cosine(embedded_vision, embedded_speech)

            vision2speech_fout.write(f'{instance_names[vision_index]},{instance_names[speech_index]},{dist}\n')
    vision2speech_fout.close()

    return v_to_s_mrr_pn, v_to_s_mrr_rand, s_to_v_mrr_pn, s_to_v_mrr_rand

def main():
    ARGS, unused = parse_args()
    
    v_to_s_mrr_pn, v_to_s_mrr_rand, s_to_v_mrr_pn, s_to_v_mrr_rand = evaluate(
        ARGS.experiment,
        ARGS.test_data_path,
        ARGS.pos_neg_examples_file,
        ARGS.gpu_num,
        ARGS.embedded_dim,
    )

    print(f'V->S p/n: {v_to_s_mrr_pn}')
    print(f'V->S rand: {v_to_s_mrr_rand}')
    print(f'S->V p/n: {s_to_v_mrr_pn}')
    print(f'S->V rand: {s_to_v_mrr_rand}')

if __name__ == '__main__':
    main()
