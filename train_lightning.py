import argparse
import os
import pickle
import random
import subprocess
import sys

import numpy as np
import scipy
import scipy.spatial
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import (DataLoader, SequentialSampler)

from datasets import gl_loaders, GLData
from losses import triplet_loss_vanilla
from rownet import RowNet
from utils import save_embeddings, load_embeddings, get_pos_neg_examples
from losses import triplet_loss_cosine_abext_marker
from triplet_loss_lightning import triplet_loss
import pytorch_lightning as pl

import datetime
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
        help='Name for output folders')
    parser.add_argument('--epochs', default=20, type=int,
        help='Number of epochs to run for')
    parser.add_argument('--train_data', default='/home/iral/data_processing/gld_data_complete.pkl',
        help='Path to training data pkl file')
    parser.add_argument('--test_data', default='/home/iral/data_processing/gld_data_complete.pkl',
        help='Path to testing data pkl file')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--pos_neg_examples_file', default=None,
        help='path to examples pkl')
    parser.add_argument('--seed', type=int, default=75,
        help='random seed for train test split')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='Dimension of embedded manifold')
    parser.add_argument('--batch_size', type=int, default=1,
       help='batch size for learning')

    return parser.parse_known_args()

def train(experiment_name, epochs, train_data_path, test_data_path, gpu_num, pos_neg_examples_file=None, margin=0.4, procrustes=0.0, seed=None, batch_size=1, embedded_dim=1024):
    """Train joint embedding networks."""

    epochs = int(epochs)
    margin = float(margin)
    gpu_num = str(gpu_num)
    procrustes = float(procrustes)
    with open(train_data_path, 'rb') as fin:
        train_data = pickle.load(fin)
    print(f'Pulling examples from file {pos_neg_examples_file}...')
    with open(pos_neg_examples_file, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)
    model = triplet_loss(train_data=train_data, pos_neg_examples=pos_neg_examples, learning_rate=0.001)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    trainer = pl.Trainer(auto_lr_find = True)

    # Run learning rate finder
    lr_finder = trainer.lr_find(model, train_dataloader=train_dataloader)

    # Results can be found in
    print(lr_finder.results)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr_finder_plot.png")

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print(new_lr)

def main():
    ARGS, unused = parse_args()

    train(
        ARGS.experiment_name,
        ARGS.epochs,
        ARGS.train_data,
        ARGS.test_data,
        ARGS.gpu_num,
        pos_neg_examples_file=ARGS.pos_neg_examples_file,
        margin=0.4,
        seed=ARGS.seed,
        batch_size=ARGS.batch_size,
        embedded_dim=ARGS.embedded_dim
    )

if __name__ == '__main__':
    main()