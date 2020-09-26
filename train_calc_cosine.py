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
import datetime
from torch.optim.swa_utils import AveragedModel, SWALR
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
    parser.add_argument('--lr', type=float, default=0.001,
       help='learning rate')

    return parser.parse_known_args()

# Add learning rate scheduling.

def get_examples_batch(pos_neg_examples, indices, train_data, instance_names):
    examples = [pos_neg_examples[i] for i in indices]

    return (
        torch.stack([train_data[i[0]] for i in examples]),
        torch.stack([train_data[i[1]] for i in examples]),
        [instance_names[i[0]] for i in examples][0],
        [instance_names[i[1]] for i in examples][0],
    )

def train(experiment_name, epochs, train_data_path, test_data_path, gpu_num, pos_neg_examples_file=None, margin=0.4, procrustes=0.0, seed=None, batch_size=1, embedded_dim=1024, lr=0.001):
    """Train joint embedding networks."""

    def lr_lambda(e):
        if e < 100:
            return lr
        elif e < 200:
            return lr*0.1
        else:
            return lr*0.01
    epochs = int(epochs)
    margin = float(margin)
    gpu_num = str(gpu_num)
    procrustes = float(procrustes)
    with open(train_data_path, 'rb') as fin:
        train_data = pickle.load(fin)

    #if test_data_path is not None:
    #    _, test_data = gl_loaders(test_data_path)

    language_train_data = [l for l, _, _, _ in train_data]
    vision_train_data = [v for _, v, _, _ in train_data]
    instance_names = [i for _, _, _, i in train_data]

    # BERT dimension
    language_dim = list(language_train_data[0].size())[0]
    # Eitel dimension
    vision_dim = list(vision_train_data[0].size())[0]
    test_path = test_data_path
    with open(test_path, 'rb') as fin:
        test_data = pickle.load(fin)

    language_test_data = [(l, i) for l, _, _, i in test_data]
    vision_test_data = [(v, i) for _, v, _, i in test_data]
    instance_names_test = [i for _, _, _, i in test_data]
    sample_size = 0
    # Setup the results and device.
    results_dir = f'./output/{experiment_name}'
    os.makedirs(results_dir, exist_ok=True)

    train_results_dir = os.path.join(results_dir, 'train_results/')
    os.makedirs(os.path.join(train_results_dir), exist_ok=True)

    train_fout = open(os.path.join(train_results_dir, 'train_out.txt'), 'w')

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

    language_model = RowNet(language_dim, embedded_dim=embedded_dim)
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)
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
    swa_language_model = AveragedModel(language_model)
    swa_vision_model = AveragedModel(vision_model)
    print(lr)

    # Initialize the optimizers and loss function.
    language_optimizer = torch.optim.Adam(language_model.parameters(), lr=lr)
    vision_optimizer = torch.optim.Adam(vision_model.parameters(), lr=lr)

    language_scheduler = torch.optim.lr_scheduler.LambdaLR(language_optimizer, lr_lambda)
    vision_scheduler = torch.optim.lr_scheduler.LambdaLR(vision_optimizer, lr_lambda)

#    language_scheduler = torch.optim.lr_scheduler.CyclicLR(language_optimizer, base_lr=lr, max_lr=lr*6)
#    vision_scheduler = torch.optim.lr_scheduler.CyclicLR(vision_optimizer, base_lr=lr, max_lr=lr*6)

    swa_language_scheduler = SWALR(language_optimizer, swa_lr=0.001)
    swa_vision_scheduler = SWALR(vision_optimizer, swa_lr=0.001)
    # Put models into training mode.
    language_model.train()
    vision_model.train()

    # Train.
    # for saving to files
    batch_loss = []
    avg_epoch_loss = []
    swa_start = 150
    train_fout.write('epoch,step,target,pos,neg,case,pos_dist,neg_dist,loss\n')
    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            train_fout.write(f'{epoch},{step},')
        #for i, (language, vision, object_name, instance_name) in enumerate(train_data):
            language, vision, object_name, instance_name = batch
            train_fout.write(f'{instance_name[0]},')
            indices = list(range(step * batch_size, min((step + 1) * batch_size, len(train_data))))
            language_pos_examples, language_neg_examples, language_pos_instance, language_neg_instance = get_examples_batch(pos_neg_examples,indices,language_train_data, instance_names)
            vision_pos_examples, vision_neg_examples, vision_pos_instance, vision_neg_instance = get_examples_batch(pos_neg_examples,indices,vision_train_data, instance_names)
        #print(f'Training {i}...')
            # Zero the parameter gradients.
            language_optimizer.zero_grad()
            vision_optimizer.zero_grad()

            rand_int = random.randint(1, 8)
            if rand_int == 1:
                train_fout.write(f'{vision_pos_instance},{vision_neg_instance},vvv,')
                target = vision_model(vision.to(device))
                pos = vision_model(vision_pos_examples.to(device))
                neg = vision_model(vision_neg_examples.to(device))
                marker = ["bbb"]
            elif rand_int == 2:
                train_fout.write(f'{language_pos_instance},{language_neg_instance},lll,')
                target = language_model(language.to(device))
                pos = language_model(language_pos_examples.to(device))
                neg = language_model(language_neg_examples.to(device))
                marker = ["aaa"]
            elif rand_int == 3:
                train_fout.write(f'{language_pos_instance},{language_neg_instance},vll')
                target = vision_model(vision.to(device))
                pos = language_model(language_pos_examples.to(device))
                neg = language_model(language_neg_examples.to(device))
                marker = ["baa"]
            elif rand_int == 4:
                train_fout.write(f'{vision_pos_instance},{vision_neg_instance},lvv')
                target = language_model(language.to(device))
                pos = vision_model(vision_pos_examples.to(device))
                neg = vision_model(vision_neg_examples.to(device))
                marker = ["abb"]
            elif rand_int == 5:
                train_fout.write(f'{vision_pos_instance},{language_neg_instance},vvl')
                target = vision_model(vision.to(device))
                pos = vision_model(vision_pos_examples.to(device))
                neg = language_model(language_neg_examples.to(device))
                marker = ["bba"]
            elif rand_int == 6:
                train_fout.write(f'{language_pos_instance},{vision_neg_instance},llv')
                target = language_model(language.to(device))
                pos = language_model(language_pos_examples.to(device))
                neg = vision_model(vision_neg_examples.to(device))
                marker = ["aab"]
            elif rand_int == 7:
                train_fout.write(f'{language_pos_instance},{vision_neg_instance},vlv')
                target = vision_model(vision.to(device))
                pos = language_model(language_pos_examples.to(device))
                neg = vision_model(vision_neg_examples.to(device))
                marker = ["bab"]
            elif rand_int == 8:
                train_fout.write(f'{vision_pos_instance},{language_neg_instance},lvl')
                target = language_model(language.to(device))
                pos = vision_model(vision_pos_examples.to(device))
                neg = language_model(language_neg_examples.to(device))
                marker = ["aba"]
            loss = triplet_loss_cosine_abext_marker(target, pos, neg, marker, margin=0.4)
            # loss = triplet_loss(target, pos, neg)

            target = target.cpu().detach().numpy()
            pos = pos.cpu().detach().numpy()
            neg = neg.cpu().detach().numpy()
            pos_dist = scipy.spatial.distance.cosine(target, pos)
            neg_dist = scipy.spatial.distance.cosine(target, neg)
            train_fout.write(f'{pos_dist},{neg_dist},{loss.item()}\n')

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
        print(f'Starting evaluation: {datetime.datetime.now().time()}')

        # reporting results for this epoch
        print('*********** epoch is finished ***********')
        print(f'epoch: {epoch + 1}, loss: {avg_epoch_loss[epoch]}')
        if epoch > swa_start:
            swa_language_model.update_parameters(language_model)
            swa_language_scheduler.step()
            swa_vision_model.update_parameters(vision_model)
            swa_vision_scheduler.step()
        else:        # Update learning rate schedulers.
            language_scheduler.step()
            vision_scheduler.step()

        torch.optim.swa_utils.update_bn(language_train_data, swa_language_model)
        torch.optim.swa_utils.update_bn(vision_train_data, swa_vision_model)

        language2language_fout = open(os.path.join(results_dir, 'language2language_epoch'+str(epoch)+'.txt'), 'w')
        language2language_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
        for language_index, language in enumerate(language_test_data):
            positive_indices = [i for i, name in enumerate(instance_names_test) if language[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_test) if language[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            language_data = language[0].to(device)
            embedded_language = swa_language_model(language_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_language = language_test_data[i]
                pos_language_data = pos_language[0].to(device)
                embedded_pos_language = swa_language_model(pos_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_language, embedded_pos_language)
                language2language_fout.write(f'{language[1]},{pos_language[1]},p,{dist}\n')

            for i in negative_indices:
                neg_language = language_test_data[i]
                neg_language_data = neg_language[0].to(device)
                embedded_neg_language = swa_language_model(neg_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_language, embedded_neg_language)
                language2language_fout.write(f'{language[1]},{neg_language[1]},n,{dist}\n')
        language2language_fout.close()
        print(f'Wrote language2language: {datetime.datetime.now().time()}')

        vision2vision_fout = open(os.path.join(results_dir, 'vision2vision_epoch'+str(epoch)+'.txt'), 'w')
        vision2vision_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
        for vision_index, vision in enumerate(vision_test_data):
            positive_indices = [i for i, name in enumerate(instance_names_test) if vision[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_test) if vision[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            vision_data = vision[0].to(device)
            embedded_vision = swa_vision_model(vision_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_vision = vision_test_data[i]
                pos_vision_data = pos_vision[0].to(device)
                embedded_pos_vision = swa_vision_model(pos_vision_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_vision)
                vision2vision_fout.write(f'{vision[1]},{pos_vision[1]},p,{dist}\n')

            for i in negative_indices:
                neg_vision = vision_test_data[i]
                neg_vision_data = neg_vision[0].to(device)
                embedded_neg_vision = swa_vision_model(neg_vision_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_vision)
                vision2vision_fout.write(f'{vision[1]},{neg_vision[1]},n,{dist}\n')
        vision2vision_fout.close()
        print(f'Wrote vision2vision: {datetime.datetime.now().time()}')

        vision2language_fout = open(os.path.join(results_dir, 'vision2language_epoch'+str(epoch)+'.txt'), 'w')
        vision2language_fout.write('vision_instance,language_instance,p/n,embedded_distance\n')
        for vision_index, vision in enumerate(vision_test_data):
            positive_indices = [i for i, name in enumerate(instance_names_test) if vision[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_test) if vision[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            vision_data = vision[0].to(device)
            embedded_vision = swa_vision_model(vision_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_language = language_test_data[i]
                pos_language_data = pos_language[0].to(device)
                embedded_pos_language = swa_language_model(pos_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_language)
                vision2language_fout.write(f'{vision[1]},{pos_language[1]},p,{dist}\n')

            for i in negative_indices:
                neg_language = language_test_data[i]
                neg_language_data = neg_language[0].to(device)
                embedded_neg_language = swa_language_model(neg_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_language)
                vision2language_fout.write(f'{vision[1]},{neg_language[1]},n,{dist}\n')
        vision2language_fout.close()
    print('Training Done!')
    train_fout.close()

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
        embedded_dim=ARGS.embedded_dim,
        lr=ARGS.lr
    )

if __name__ == '__main__':
    main()
