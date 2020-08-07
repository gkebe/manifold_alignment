import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import torch
import matplotlib.pyplot as plt
from datasets import GLData
from datasets import gl_loaders
from metrics import corr_between, knn, mean_reciprocal_rank, object_identification_task_classifier
from models import RowNet
from utils import get_pos_neg_examples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='RandomExperiment',
        help='name of experiment to test')
    parser.add_argument('--test_data_path', type=str,
        help='path to testing data')
    parser.add_argument('--train_data_path', type=str, default=None,
        help='path to testing data')
    parser.add_argument('--pos_neg_examples_file', default=None,
        help='path to examples pkl')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--seed', default=None,
        help='random seed for train test split')

    return parser.parse_known_args()

def test_vision_triplet(anchor, positive, negative):
    positive_distance = torch.dist(anchor, positive).item()
    negative_distance = torch.dist(anchor, negative).item()

    return positive_distance < negative_distance

def test_language_triplet(anchor, positive, negative):
    cos = torch.nn.CosineSimilarity()
    positive_distance = cos(anchor, positive).item()
    negative_distance = cos(anchor, negative).item()

    return positive_distance < negative_distance

def test(experiment_name, test_data_path, gpu_num, train_data_path=None, pos_neg_examples_file=None, margin=0.4, seed=None):
    gpu_num = int(gpu_num)
    margin = float(margin)

    # BERT dimension
    language_dim = 3072
    # Eitel dimension
    vision_dim = 4096
    embedded_dim = 1024

    # Setup the results and device.
    results_dir = f'./{experiment_name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_results_dir = os.path.join(results_dir, 'train_results/')
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)

    test_results_dir = os.path.join(results_dir, 'test_results/')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)

    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    with open(os.path.join(results_dir, 'hyperparams_test.txt'), 'w') as f:
        f.write('Command used to run: python \n')
        f.write(f'ARGS: {sys.argv}\n')
        f.write(f'device in use: {device}\n')
        f.write(f'--experiment_name {experiment_name}\n')
        f.write(f'--seed {seed}\n')

    train_data, test_data = gl_loaders(test_data_path, seed=seed)

    language_train_data = [l for l, _, _, _ in train_data]
    vision_train_data = [v for _, v, _, _ in train_data]
    language_test_data = [l for l, _, _, _ in test_data]
    vision_test_data = [v for _, v, _, _ in test_data]

    if pos_neg_examples_file is None:
        print('Calculating examples from scratch...')
        pos_neg_examples = []
        for anchor_language, _, _, _ in test_data:
            pos_neg_examples.append(get_pos_neg_examples(anchor_language, language_test_data))
        with open(f'{experiment_name}_test_examples.pkl', 'wb') as fout:
            pickle.dump(pos_neg_examples, fout)
    else:
        print(f'Pulling examples from file {pos_neg_examples_file}...')
        with open(pos_neg_examples_file, 'rb') as fin:
            pos_neg_examples = pickle.load(fin)

    language_model = RowNet(language_dim, embed_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embed_dim=embedded_dim)

    # Finish model setup.
    print(f'pulling models from {train_results_dir}')
    language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt')))
    vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt')))
    language_model.to(device)
    vision_model.to(device)

    # Put models into evaluation mode.
    language_model.eval()
    vision_model.eval()

        # Iterate through the train data.
    language_train = []
    vision_train = []

    print("Computing embeddings of train data to calculate threshhold for distance")
    for language, vision, object_name, instance_name in train_data:
        positive_data = language.to(device)
        anchor_data = vision.to(device)
        language_train.append(language_model(positive_data).cpu().detach().numpy())
        vision_train.append(vision_model(anchor_data).cpu().detach().numpy())
    print("Finished Computing embeddings of train data")

    language_train = np.concatenate(language_train, axis=0)
    vision_train = np.concatenate(vision_train, axis=0)

    # Test data
    # For accumulating predictions to check embedding visually using test set.
    language_embeddings = []
    vision_embeddings = []
    labels = []
    instance_data = []

    language_tests = []
    vision_tests = []
    # Iterate through the test data.
    print("Computing embeddings of test data...")
    for i, (language, vision, object_name, instance_name) in enumerate(test_data):
        language = language.to(device)
        vision = vision.to(device)
        instance_data.extend(instance_name)
        labels.extend(object_name)
        language_embeddings.append(language_model(language).cpu().detach().numpy())
        vision_embeddings.append(vision_model(vision).cpu().detach().numpy())

        positive_language = language_test_data[pos_neg_examples[i][0]].to(device)
        negative_language = language_test_data[pos_neg_examples[i][1]].to(device)
        positive_vision = vision_test_data[pos_neg_examples[i][0]].to(device)
        negative_vision = vision_test_data[pos_neg_examples[i][1]].to(device)

        language_tests.append(test_language_triplet(language_model(language), language_model(positive_language), language_model(negative_language)))
        vision_tests.append(test_vision_triplet(vision_model(vision), vision_model(positive_vision), vision_model(negative_vision)))
    print("Finished computing embeddings for test data")

    # Convert string labels to ints.
    labelencoder = LabelEncoder()
    labelencoder.fit(labels)
    encoded_labels = labelencoder.transform(labels)

    # Concatenate predictions.
    language_embeddings = np.concatenate(language_embeddings, axis=0)
    vision_embeddings = np.concatenate(vision_embeddings, axis=0)
    ab = np.concatenate((language_embeddings, vision_embeddings), axis=0)

    # Given language -> nearest vision embeddings
    ground_truth, predicted, distance = object_identification_task_classifier(
        language_embeddings,
        vision_embeddings,
        encoded_labels,
        language_train,
        vision_train,
        lamb_std=1,
        cosine=False
    )

    precisions = []
    recalls = []
    f1s = []
    precisions_binary = []
    recalls_binary = []
    f1s_binary = []
    #print(classification_report(oit_res[i], 1/np.arange(1,len(oit_res[i])+1) > 0.01))
    for i in range(len(ground_truth)):
        p, r, f, s = precision_recall_fscore_support(ground_truth[i], predicted[i], warn_for=(), average='micro')
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        p, r, f, s = precision_recall_fscore_support(ground_truth[i], predicted[i], warn_for=(), average='binary')
        precisions_binary.append(p)
        recalls_binary.append(r)
        f1s_binary.append(f)

    # TODO: output to file
    print('\n ')
    print(experiment_name+'_'+str(embedded_dim))
    print('MRR,    KNN,    Corr,   Mean F1,    Mean F1 (pos only)')
    print('%.3g & %.3g & %.3g & %.3g & %.3g' % (
        mean_reciprocal_rank(language_embeddings, vision_embeddings, encoded_labels, cosine=False),
        knn(language_embeddings, vision_embeddings, encoded_labels, k=5, cosine=False),
        corr_between(language_embeddings, vision_embeddings, cosine=False), np.mean(f1s), np.mean(f1s_binary))
    )

    language_accuracy = language_tests.count(True) / len(language_tests)
    vision_accuracy = vision_tests.count(True) / len(vision_tests)

    print(f'Language: {language_tests.count(True)}/{len(language_tests)} = {language_accuracy}')
    print(f'Vision: {vision_tests.count(True)}/{len(vision_tests)} = {vision_accuracy}')

if __name__ == '__main__':
    ARGS, unused = parse_args()

    test(
        ARGS.experiment_name,
        ARGS.test_data_path,
        ARGS.gpu_num,
        train_data_path=ARGS.train_data_path,
        pos_neg_examples_file=ARGS.pos_neg_examples_file,
        seed=ARGS.seed,
    )
