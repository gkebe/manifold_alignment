import argparse
import csv
import os
import random
import statistics
import sys
import umap

import numpy as np
from numpy.linalg import norm
import seaborn as sns
sns.set_context('poster')
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import roc_curve, classification_report, precision_recall_fscore_support
import torch
import matplotlib.pyplot as plt

from datasets import gl_loaders
from metrics import corr_between, rank_from, rank_from_class, knn, mean_reciprocal_rank, object_identification_task_rank, object_identification_task_classifier
from models import RowNet
from utils import setup_dirs, setup_device, save_embeddings, load_embeddings

# TODO: move to end of train loop, or to test.py
# Computing F1, recall, and precision on test data at each epoch
#with torch.no_grad():
#    for language, vision, object_name, instance_name in test_loader:
#        language = language.to(device)
#        vision = vision.to(device)
#        test_label_name = object_name

#    p, r, f, s = precision_recall_fscore_support(ground_truth_test, predictions_test, average='micro')
#    F1_test.append(f)
#    precision_test.append(p)
#    recall_test.append(r)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='RandomExperiment',
        help='name of experiment to test')
    parser.add_argument('--test_data_path', type=str,
        help='path to testing data')
    parser.add_argument('--train_data_path', type=str, default=None,
        help='path to testing data')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--seed', default=None,
        help='random seed for train test split')

    return parser.parse_known_args()

def test(experiment_name, test_data_path, gpu_num, train_data_path=None, margin=0.4, seed=None):
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

    train_data, test_data = gl_loaders(test_data_path, seed=seed)

    if train_data_path is not None:
        train_data, _ = gl_loaders(train_data_path, seed=seed)

    language_model = RowNet(language_dim, embed_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embed_dim=embedded_dim)

    # Finish model setup.
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

    train_distances = []
    print("Computing embeddings of train data to calculate threshhold for distance")
    for language, vision, object_name, instance_name in train_data:
        language_data = language.to(device)
        vision_data = vision.to(device)

        embedded_language = language_model(language_data)
        embedded_vision = vision_model(vision_data)

        train_distances.append(torch.dist(embedded_language, embedded_vision).item())

        #language_train.append(language_model(positive_data).cpu().detach().numpy())
        #vision_train.append(vision_model(anchor_data).cpu().detach().numpy())

    print("Finished Computing embeddings of train data")
    print(f'{statistics.mean(train_distances)}, {statistics.stdev(train_distances)}')

    mean = statistics.mean(train_distances)
    std = statistics.stdev(train_distances)
    upper_bound = mean + std

    print(f'upper_b: {upper_bound}')

    y_pred = []
    y_true = []
    for i, (language, _, object_name1, instance_name1) in enumerate(test_data):
        print(f'calculating {i} out of {len(test_data)}')
        lanuage_data = language.to(device)

        for _, vision, object_name2, instance_name2 in test_data:
            vision_data = vision.to(device)

            if object_name1 == object_name2:
            #if instance_name1 == instance_name2:
                y_true.append(1)
            else:
                y_true.append(0)

            dist = torch.dist(language_model(language_data), vision_model(vision_data)).item()

            if dist < upper_bound:
                y_pred.append(1)
            else:
                y_pred.append(0)

            #print(f'{type(lower_bound)} {type(upper_bound)} {type(dist)}')
            #print(f'{lower_bound} {upper_bound} {dist}')

    print(f'y_true: 1:{y_true.count(1)}, 0:{y_true.count(0)}')
    print(f'y_pred: 1:{y_pred.count(1)}, 0:{y_pred.count(0)}')

    micro_p, micro_r, micro_f1, micro_s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    binary_p, binary_r, binary_f1, binary_s = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print('micro_p\tmicro_r\tmicro_f1\tmicro_s')
    print(micro_p, micro_r, micro_f1, micro_s)

    print('binary_p\tbinary_r\tbinary_f1\tbinary_s')
    print(binary_p, binary_r, binary_f1, binary_s)

    return

    language_train = np.concatenate(language_train, axis=0)
    vision_train = np.concatenate(vision_train, axis=0)

    # Test data
    # For accumulating predictions to check embedding visually using test set.
    language_embeddings = []
    vision_embeddings = []
    labels = []
    instance_data = []

    # Iterate through the test data.
    print("Computing embeddings of test data...")
    for language, vision, object_name, instance_name in test_data:
        language = language.to(device)
        vision = vision.to(device)

        instance_data.extend(instance_name)
        labels.extend(object_name)

        language_embeddings.append(language_model(language).cpu().detach().numpy())
        vision_embeddings.append(vision_model(vision).cpu().detach().numpy())

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

    return

    # TODO: !???!!??1/

    a = language_embeddings
    b = vision_embeddings

    plt.figure(figsize=(14,7))
    for i in range(len(ground_truth)):
        fpr, tpr, thres = roc_curve(ground_truth[i], [1-e for e in distance[i]], drop_intermediate=True)
        plt.plot(fpr,tpr,alpha=0.08,color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(test_results_dir + '_' + str(embedded_dim)+'_ROC.svg')

    # Pick a pair, plot distance in A vs distance in B. Should be correlated.
    a_dists = []
    b_dists = []
    for _ in range(3000):
        i1 = random.randrange(len(a))
        i2 = random.randrange(len(a))
        a_dists.append(euclidean(a[i1], a[i2]))
        b_dists.append(euclidean(b[i1], b[i2]))
    #     a_dists.append(cosine(a[i1], a[i2]))
    #     b_dists.append(cosine(b[i1], b[i2]))

    # Plot.
    plt.figure(figsize=(14,14))
    #plt.title('Check Distance Correlation Between Domains')
    plt.xlim([0, 3])
    plt.ylim([0, 3])
    # plt.xlim([0,max(a_dists)])
    # plt.ylim([0,max(b_dists)])
    # plt.xlabel('Distance in Domain A')
    # plt.ylabel('Distance in Domain B')
    plt.xlabel('Distance in Language Domain')
    plt.ylabel('Distance in Vision Domain')
    #plt.plot(a_dists_norm[0],b_dists_norm[0],'.')
    #plt.plot(np.arange(0,2)/20,np.arange(0,2)/20,'k-',lw=3)
    plt.plot(a_dists, b_dists, 'o', alpha=0.5)
    plt.plot(np.arange(0, 600), np.arange(0, 600), 'k--', lw=3, alpha=0.5)
    #plt.text(-0.001, -0.01, 'Corr: %.3f'%(pearsonr(a_dists,b_dists)[0]),  fontsize=20)
    plt.savefig(test_results_dir + '_'+str(embedded_dim) + '_CORR.svg')

    # Inspect embedding distances.
    clas = 5  # Base class.
    i_clas = [i for i in range(len(labels)) if labels[i].item() == clas]
    i_clas_2 = np.random.choice(i_clas, len(i_clas), replace=False)

    clas_ref = 4  # Comparison class.
    i_clas_ref = [i for i in range(len(labels)) if labels[i].item() == clas_ref]

    ac = np.array([a[i] for i in i_clas])
    bc = np.array([b[i] for i in i_clas])

    ac2 = np.array([a[i] for i in i_clas_2])
    bc2 = np.array([b[i] for i in i_clas_2])

    ac_ref = np.array([a[i] for i in i_clas_ref])
    aa_diff_ref = norm(ac[:min(len(ac), len(ac_ref))] - ac_ref[:min(len(ac),len(ac_ref))], ord=2, axis=1)

    ab_diff = norm(ac - bc2, ord=2, axis=1)
    aa_diff = norm(ac - ac2, ord=2, axis=1)
    bb_diff = norm(bc - bc2, ord=2, axis=1)

    # aa_diff_ref = [cosine(ac[:min(len(ac),len(ac_ref))][i],ac_ref[:min(len(ac),len(ac_ref))][i]) for i in range(len(ac[:min(len(ac),len(ac_ref))]))]

    # ab_diff = [cosine(ac[i],bc2[i]) for i in range(len(ac))]
    # aa_diff = [cosine(ac[i],ac2[i]) for i in range(len(ac))]
    # bb_diff = [cosine(bc[i],bc2[i]) for i in range(len(ac))]

    bins = np.linspace(0, 0.1, 100)

    plt.figure(figsize=(14,7))
    plt.hist(ab_diff, bins, alpha=0.5, label='between embeddings')
    plt.hist(aa_diff, bins, alpha=0.5, label='within embedding A')
    plt.hist(bb_diff, bins, alpha=0.5, label='within embedding B')

    plt.hist(aa_diff_ref, bins, alpha=0.5, label='embedding A, from class ' + str(clas_ref))

    plt.title('Embedding Distances - Class: '+str(clas))
    plt.xlabel('L2 Distance')
    plt.ylabel('Count')
    plt.legend()

    #labelencoder.classes_
    classes_to_keep = [36, 6, 9, 46, 15, 47, 50, 22, 26, 28]
    print(labelencoder.inverse_transform(classes_to_keep))

    ab_norm = [e for i, e in enumerate(ab) if labels[i % len(labels)]  in classes_to_keep]
    ys_norm = [e for e in labels if e in classes_to_keep]

    color_index = {list(set(ys_norm))[i]: i for i in range(len(set(ys_norm)))} #set(ys_norm)
    markers = ["o","v","^","s","*","+","x","D","h","4"]
    marker_index = {list(set(ys_norm))[i]: markers[i] for i in range(len(set(ys_norm)))}

    embedding = umap.UMAP(n_components=2).fit_transform(ab_norm) # metric='cosine'
    # Plot UMAP embedding of embeddings for all classes.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    mid = len(ys_norm)

    ax1.set_title('Language UMAP')
    for e in list(set(ys_norm)):
        x1 = [embedding[:mid, 0][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        x2 = [embedding[:mid, 1][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        ax1.scatter(x1, x2, marker=marker_index[int(e)], alpha=0.5, c=[sns.color_palette("colorblind", 10)[color_index[int(e)]]],label=labelencoder.inverse_transform([int(e)])[0])
    ax1.set_xlim([min(embedding[:,0])-4, max(embedding[:,0])+4])
    ax1.set_ylim([min(embedding[:,1])-4, max(embedding[:,1])+4])
    ax1.grid(True)
    ax1.legend(loc='upper center', bbox_to_anchor=(1.1, -0.08),fancybox=True, shadow=True, ncol=5)

    ax2.set_title('Vision UMAP')
    for e in list(set(ys_norm)):
        x1 = [embedding[mid::, 0][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        x2 = [embedding[mid::, 1][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        ax2.scatter(x1, x2, marker=marker_index[int(e)], alpha=0.5, c=[sns.color_palette("colorblind", 10)[color_index[int(e)]]])
    ax2.set_xlim([min(embedding[:,0])-4, max(embedding[:,0])+4])
    ax2.set_ylim([min(embedding[:,1])-4, max(embedding[:,1])+4])
    ax2.grid(True)

    plt.savefig(test_results_dir+'_'+str(embedded_dim)+'_UMAP_wl.svg', bbox_inches='tight')

    #sns.palplot(sns.color_palette("colorblind", 10))

if __name__ == '__main__':
    ARGS, unused = parse_args()

    test(
        ARGS.experiment_name,
        ARGS.test_data_path,
        ARGS.gpu_num,
        train_data_path=ARGS.train_data_path,
        seed=ARGS.seed,
    )
