import os
import pickle
import random

#import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch

def get_pos_neg_examples(language, language_data):
    """
    language: the language to find negatives of
    language_data: the set of all language features
    n: the top n negatives to choose from
    k: sample size to return

    returns the indices in language_data of the negatives as a list
    """
    # TODO: language could be a tensor of multiple language data,
    #   needs to grab and assemble a tensor with all negative examples
    cosines = []
    for i, vector in enumerate(language_data):
        # .data[0] necessary to compare on cuda
        if torch.equal(language, vector):
            continue
        cosines.append((i, torch.nn.functional.cosine_similarity(language, vector, 0).data[0]))
    cosines.sort(key=lambda x: x[1])

    # choose randomly for top n negative examples
    # this returns indexes
    positive_index = cosines[0][0]
    negative_index = cosines[-1][0]

    return positive_index, negative_index

def setup_dirs(experiment_name):
    """Create results and experiment directories if doesn't exist."""
    results_dir = './results/' + experiment_name + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return results_dir

def setup_device(gpu_num=0):
    """Setup device."""
    # IS there a GPU?
    device_name = f'cuda: {gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    return device

def load_embeddings(embedding_path):
    return np.load(embedding_path) if os.path.exists(embedding_path) else None

def save_embeddings(embedding_path,embedding):
    if not os.path.exists(embedding_path):
        np.save(embedding_path, embedding)

#create_plot(
#    os.path.join(train_results_dir, 'epoch_loss.pkl'),
#    'Correlation History',
#    True,
#    'epoch',
#    'Correlation (log scale)',
#    None,
#    'log',
#    None,
#    (14, 7),
#    os.path.join(train_results_dir, 'Figures/')
#)

#def create_plot(pkldata, title, xlabel, ylabel, xscale, yscale, legend, figsize, location):
#    if not os.path.exists(location):
#        os.makedirs(location)
#
#    visual = pickle.load(open(pkldata, 'rb'))
#    #Note: I always save the x label in visual[0], and the results from different methods in next indices visual[1], visual[2], ...
#    if figsize is not None:
#        plt.figure(figsize=figsize)
#
#    if legend is not None:
#        for i in range(len(visual)):
#            plt.plot(visual[i], label=legend[i])
#    else:
#        for i in range(len(visual)):
#            plt.plot(visual[i])
#
#    plt.title(title)
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    if xscale is not None:
#        plt.xscale(xscale)
#    if yscale is not None:
#        plt.yscale(yscale)
#    if legend is not None:
#        plt.legend(loc='lower right')
#    plt.savefig(f'{location}{title}.pdf')
#    plt.close('all')

def procrustes_distance(A, B):
    # Translation.
    A -= torch.mean(A, 0)
    B -= torch.mean(B, 0)

    # Scaling.
    A = torch.div(A, torch.norm(A))
    B = torch.div(B, torch.norm(B))

    # Orthogonal Procrustes.
    M = torch.t(torch.mm(torch.t(B),A))
    u, s, v = torch.svd(M)
    R = torch.mm(u, torch.t(v))
    s = torch.sum(s)
    B = s * torch.mm(B,torch.t(R))

    # Compute distance.
    dists = torch.norm(A - B, dim=1)

    return dists

class Procrustes():
    """Transformation: translate, scale, and rotate/reflect."""
    def __init__(self, A, B, trans=True, scale=True, rot=True):
        self.trans = trans
        self.scale = scale
        self.rot = rot

        if self.trans:
            # Compute translation.
            self.mean_A = np.mean(A, axis=0)
            self.mean_B = np.mean(B, axis=0)
            A = A - self.mean_A
            B = B - self.mean_B

        if self.scale:
            # Compute scaling.
            self.s_A = np.linalg.norm(A)
            self.s_B = np.linalg.norm(B)
            A = np.divide(A, self.s_A)
            B = np.divide(B, self.s_B)

        if self.rot:
            # Compute Orthogonal Procrustes.
            M = B.T.dot(A).T
            u, s, vh = np.linalg.svd(M)
            self.R = u.dot(vh)
            self.s = np.sum(s)

    def transform(self, A, B):
        if self.trans:
            A = A - self.mean_A
            B = B - self.mean_B
        if self.scale:
            A = np.divide(A, self.s_A)
            B = np.divide(B, self.s_B)
        if self.rot:
            B = B.dot(self.R.T) * self.s

        return A, B
