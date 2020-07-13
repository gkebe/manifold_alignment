import argparse
import os
import pickle
import random
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import (DataLoader, SequentialSampler)
from datasets import gl_loaders, GLData
from losses import triplet_loss_vanilla
from utils import save_embeddings, load_embeddings, get_pos_neg_examples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
        help='Name for output folders')
    parser.add_argument('--epochs', default=20, type=int,
        help='Number of epochs to run for')
    parser.add_argument('--train_data', default='/home/iral/data_processing/gld_data_complete.pkl',
        help='Path to training data pkl file')
    parser.add_argument('--gpu_num', default='0',
        help='gpu id number')
    parser.add_argument('--pos_neg_examples_file', default=None,
        help='path to examples pkl')
    parser.add_argument('--seed', type=int, default=75,
        help='random seed for train test split')
    #parser.add_argument('--embedded_dim', default=1024, type=int,
    #    help='Dimension of embedded manifold')
    parser.add_argument('--batch_size', type=int, default=1,
       help='batch size for learning')

    return parser.parse_known_args()

class RowNet(torch.nn.Module):
    def __init__(self, input_size, embed_dim=1024):
        # Language (BERT): 3072, Vision+Depth (ResNet152): 2048 * 2.
        super(RowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, input_size)
        self.fc3 = torch.nn.Linear(input_size, embed_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=.2)
        x = self.fc3(x)

        return x

# Add learning rate scheduling.
def lr_lambda(e):
    if e < 50:
        return 0.001
    elif e < 100:
        return 0.0001
    else:
        return 0.00001

def get_examples_batch(pos_neg_examples,indices,train_data):
    examples = [pos_neg_examples[i] for i in indices]
    return [train_data[i] for i in examples]


def train(experiment_name, epochs, train_data_path, gpu_num, pos_neg_examples_file=None, margin=0.4, procrustes=0.0, seed=None, batch_size=1):
    """Train joint embedding networks."""

    epochs = int(epochs)
    margin = float(margin)
    gpu_num = str(gpu_num)
    procrustes = float(procrustes)

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
        os.makedirs(os.path.join(train_results_dir))

    device_name = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    with open(os.path.join(results_dir, 'hyperparams_train.txt'), 'w') as f:
        f.write('Command used to run: python \n')
        f.write(f'ARGS: {sys.argv}\n')
        f.write(f'device in use: {device}\n')
        f.write(f'--epochs {epochs}\n')
        f.write(f'--seed {seed}\n')

    # Setup data loaders and models.
    #train_data, _ = gl_loaders(train_data_path, seed=seed)

    with open(train_data_path, 'rb') as fin:
        train_data = pickle.load(fin)

    #if test_data_path is not None:
    #    _, test_data = gl_loaders(test_data_path)

    language_train_data = [l for l, _, _, _ in train_data]
    vision_train_data = [v for _, v, _, _ in train_data]

    if pos_neg_examples_file is None:
        print('Calculating examples from scratch...')
        pos_neg_examples = []
        for anchor_language, _, _, _ in train_data:
            pos_neg_examples.append(get_pos_neg_examples(anchor_language, language_train_data))
        with open(f'{experiment_name}_train_examples.pkl', 'wb') as fout:
            pickle.dump(pos_neg_examples, fout)
    else:
        print(f'Pulling examples from file {pos_neg_examples_file}...')
        with open(pos_neg_examples_file, 'rb') as fin:
            pos_neg_examples = pickle.load(fin)

    language_model = RowNet(language_dim, embed_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embed_dim=embedded_dim)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Finish model setup
    # If we want to load pretrained models to continue training...
    if os.path.exists(os.path.join(train_results_dir, 'model_A_state.pt')) and os.path.exists(os.path.join(train_results_dir, 'model_B_state.pt')):
        print('Starting from pretrained networks.')
        print(f'Loaded model_A_state.pt and model_B_state.pt from {train_results_dir}')
        language_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt')))
        vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt')))
    else:
        print('Starting from scratch to train networks.')

    language_model.to(device)
    vision_model.to(device)

    # Initialize the optimizers and loss function.
    language_optimizer = torch.optim.Adam(language_model.parameters(), lr=0.00001)
    vision_optimizer = torch.optim.Adam(vision_model.parameters(), lr=0.00001)

    language_scheduler = torch.optim.lr_scheduler.LambdaLR(language_optimizer, lr_lambda)
    vision_scheduler = torch.optim.lr_scheduler.LambdaLR(vision_optimizer, lr_lambda)

    # Put models into training mode.
    language_model.train()
    vision_model.train()

    # Train.
    # for saving to files
    batch_loss = []
    avg_epoch_loss = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        #for i, (language, vision, object_name, instance_name) in enumerate(train_data):
            language, vision, object_name, instance_name = batch
            indices = list(range(step * 3, (step + 1) * 3))
            language_pos_examples, language_neg_examples = get_examples_batch(pos_neg_examples,indices,language_train_data)
            vision_pos_examples, vision_neg_examples = get_examples_batch(pos_neg_examples,indices,vision_train_data)
        #print(f'Training {i}...')
            # Zero the parameter gradients.
            language_optimizer.zero_grad()
            vision_optimizer.zero_grad()
            triplet_loss = torch.nn.TripletMarginLoss(margin=0.4, p=2)

            rand_int = random.randint(1, 4)

            # Determine triplets based on epoch + index (mod 4)
            if rand_int == 1:
                anchor = vision.to(device)
                positive = vision_pos_examples.to(device)
                negative = vision_neg_examples.to(device)
                loss = triplet_loss(vision_model(anchor), vision_model(positive), vision_model(negative))

            elif rand_int == 2:
                anchor = language.to(device)
                positive = language_pos_examples.to(device)
                negative = language_neg_examples.to(device)
                loss = triplet_loss(language_model(anchor), language_model(positive), language_model(negative))

            elif rand_int == 3:
                anchor = vision.to(device)
                positive = language_pos_examples.to(device)
                negative = language_neg_examples.to(device)
                loss = triplet_loss(vision_model(anchor), language_model(positive), language_model(negative))

            elif rand_int == 4:
                anchor = language.to(device)
                positive = vision_pos_examples.to(device)
                negative = vision_neg_examples.to(device)
                loss = triplet_loss(language_model(anchor), vision_model(positive), vision_model(negative))

            loss.backward()
            vision_optimizer.step()
            language_optimizer.step()

            # Forward.
            #loss = triplet_loss_vanilla(
            #    vision_data,
            #    language_data,
            #    negative,
            #    language_model,
            #    vision_model,
            #    margin=margin
            #)

            #if procrustes > 0:
            #    p_loss = procrustes_loss(anchor, positive, marker, language_model, vision_model)
            #    loss += procrustes * p_loss
            #    # mflow.log_metric(key='procurstes_loss', value=p_loss.item(), step=batch_num)

            # Save batch loss.
            batch_loss.append(loss.item())
            epoch_loss += loss.item()

            #reporting progress
            if not step % (len(train_data) // 32):
                print(f'epoch: {epoch + 1}, batch: {step + 1}, loss: {loss.item()}')

        # Save network state at each epoch.
        torch.save(language_model.state_dict(), os.path.join(train_results_dir, 'model_A_state.pt'))
        torch.save(vision_model.state_dict(), os.path.join(train_results_dir, 'model_B_state.pt'))

        # average loss over the entire epoch
        avg_epoch_loss.append(epoch_loss / len(train_data))

        # Save loss data
        with open(os.path.join(train_results_dir, 'epoch_loss.pkl'), 'wb') as fout:
            pickle.dump(avg_epoch_loss, fout)

        with open(os.path.join(train_results_dir, 'batch_loss.pkl'), 'wb') as fout:
            pickle.dump(batch_loss, fout)

        # reporting results for this epoch
        print('*********** epoch is finished ***********')
        print(f'epoch: {epoch + 1}, loss: {avg_epoch_loss[epoch]}')

        # Update learning rate schedulers.
        language_scheduler.step()
        vision_scheduler.step()

    print('Training Done!')

def main():
    ARGS, unused = parse_args()

    train(
        ARGS.experiment_name,
        ARGS.epochs,
        ARGS.train_data,
        ARGS.gpu_num,
        pos_neg_examples_file=ARGS.pos_neg_examples_file,
        margin=0.4,
        seed=ARGS.seed,
        batch_size=ARGS.batch_size
    )

if __name__ == '__main__':
    main()
