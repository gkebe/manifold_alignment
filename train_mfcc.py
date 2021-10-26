import argparse
import os
import pickle
import random
import numpy as np
import scipy
import scipy.spatial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
#import torchaudio
from tqdm import tqdm

from datasets import GLData
from lstm import LSTM
from rnn import RNN
from rownet import RowNet
from losses import triplet_loss_cosine_abext_marker
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='path to experiment output')
    parser.add_argument('--epochs', default=1, type=int,
        help='number of epochs to train')
    parser.add_argument('--train_data', help='path to train data')
    parser.add_argument('--dev_data', default='/home/iral/data_processing/gld_data_complete.pkl',
        help='Path to dev data pkl file')
    parser.add_argument('--test_data', default='/home/iral/data_processing/gld_data_complete.pkl',
        help='Path to testing data pkl file')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--pos_neg_examples_file',
        help='path to pos/neg examples pkl')
    parser.add_argument('--seed', type=int, default=75,
        help='random seed for reproducability')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='dimension of embedded manifold')
    parser.add_argument('--batch_size', type=int, default=1,
        help='training batch size')
    parser.add_argument('--num_layers', type=int, default=1,
        help='number of hidden layers')
    parser.add_argument('--awe', type=int, default=32,
        help='number of hidden units to keep')
    parser.add_argument('--h', type=int, default=None,
        help='Value for TBPTT')

    return parser.parse_known_args()

def lr_lambda(e):
    if e < 100:
        return 0.001
    elif e < 200:
        return 0.0001
    else:
        return 0.00001

#def lr_lambda(epoch):
#    return .95 ** epoch

def get_examples_batch(pos_neg_examples, indices, train_data, instance_names):
    examples = [pos_neg_examples[i] for i in indices]

    return (
        torch.stack([train_data[i[0]] for i in examples]),
        torch.stack([train_data[i[1]] for i in examples]),
        [instance_names[i[0]] for i in examples][0],
        [instance_names[i[1]] for i in examples][0],
    )

def train(experiment_name, epochs, train_data_path, dev_data_path, test_data_path, pos_neg_examples_file, batch_size, embedded_dim, gpu_num, seed, num_layers, h, awe, margin=0.4):

    results_dir = f'./output/{experiment_name}'
    os.makedirs(results_dir, exist_ok=True)

    train_results_dir = os.path.join(results_dir, 'train_results/')
    os.makedirs(train_results_dir, exist_ok=True)

    train_fout = open(os.path.join(train_results_dir, 'train_out.txt'), 'w')

    print(f'cuda:{gpu_num}; cuda is available? {torch.cuda.is_available()}')
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    with open(train_data_path, 'rb') as fin:
        train_data = pickle.load(fin)

    speech_train_data = [s for s, _, _, _, _ in train_data]
    vision_train_data = [v for _, v, _, _, _ in train_data]
    instance_names = [i for _, _, _, i, _ in train_data]
    test_path = test_data_path
    with open(test_path, 'rb') as fin:
        test_data = pickle.load(fin)

    language_test_data = [(l, i) for l, _, _, i, _ in test_data]
    vision_test_data = [(v, i) for _, v, _, i, _ in test_data]
    instance_names_test = [i for _, _, _, i, _ in test_data]

    dev_path = dev_data_path
    with open(dev_path, 'rb') as fin:
        dev_data = pickle.load(fin)

    language_dev_data = [(l, i) for l, _, _, i, _ in dev_data]
    vision_dev_data = [(v, i) for _, v, _, i, _ in dev_data]
    instance_names_dev = [i for _, _, _, i, _ in dev_data]

    with open(pos_neg_examples_file, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)

    # TODO: grab speech dimension from speech data tensor
    # TODO: set some of these from ARGS
    speech_dim = list(speech_train_data[0].size())[1]
    speech_model = LSTM(
        input_size=speech_dim,
        output_size=embedded_dim,
        hidden_dim=64,
        awe=32,
        num_layers=num_layers,
        dropout=0.0,
        device=device
    )

    # Sets number of time steps for truncated back propogation through time
    speech_model.set_TBPTT(h)

    vision_dim = list(vision_train_data[0].size())[0]
    vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)

    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    if os.path.exists(os.path.join(train_results_dir, 'speech_model.pt')) and os.path.exists(os.path.join(train_results_dir, 'vision_model.pt')):
        print('Starting from pretrained networks.')
        print(f'Loaded speech_model.pt and vision_model.pt from {train_results_dir}')
        speech_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'speech_model.pt')))
        vision_model.load_state_dict(torch.load(os.path.join(train_results_dir, 'vision_model.pt')))
    else:
        print('Couldn\'t find models. Training from scratch...')

    speech_model.to(device)
    vision_model.to(device)

    # TODO: does this need to change for the RNN?
    speech_optimizer = torch.optim.Adam(speech_model.parameters(), lr=0.001)
    vision_optimizer = torch.optim.Adam(vision_model.parameters(), lr=0.001)

    speech_scheduler = torch.optim.lr_scheduler.LambdaLR(speech_optimizer, lr_lambda)
    vision_scheduler = torch.optim.lr_scheduler.LambdaLR(vision_optimizer, lr_lambda)
    sample_size = 0
    speech_model.train()
    vision_model.train()

    batch_loss = []
    avg_epoch_loss = []

    train_fout.write('epoch,step,target,pos,neg,case,pos_dist,neg_dist,loss\n')
    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            train_fout.write(f'{epoch},{step},')
            speech, vision, object_name, instance_name, user_id = batch
            train_fout.write(f'{instance_name[0]},')

            indices = list(range(step * batch_size, min((step + 1) * batch_size, len(train_data))))
            speech_pos, speech_neg, speech_pos_instance, speech_neg_instance = get_examples_batch(pos_neg_examples, indices, speech_train_data, instance_names)
            vision_pos, vision_neg, vision_pos_instance, vision_neg_instance = get_examples_batch(pos_neg_examples, indices, vision_train_data, instance_names)

            speech_optimizer.zero_grad()
            vision_optimizer.zero_grad()

            # TODO: still using triplet loss?
            triplet_loss = torch.nn.TripletMarginLoss(margin=0.4, p=2)

            # TODO: JANKY STUFF FOR TENSOR SIZING AND ORDER ERRORS
            # this has to do with the batch_first param of the RNN

            # TODO: If batching, need to pad step dim with 0s to
            #   match the max step size

            #print('SPEECH SIZES')
            #print(speech.size())
            #print(speech_pos.size())
            #print(speech_neg.size())

            #print('VISION SIZES')
            #print(vision.size())
            #print(vision_pos.size())
            #print(vision_neg.size())

            # Randomly choose triplet case
            case = random.randint(1, 8)
            if case == 1:
                train_fout.write(f'{vision_pos_instance},{vision_neg_instance},vvv,')
                target = vision_model(vision.to(device))
                pos = vision_model(vision_pos.to(device))
                neg = vision_model(vision_neg.to(device))
                marker = ["bbb"]
            elif case == 2:
                train_fout.write(f'{speech_pos_instance},{speech_neg_instance},sss,')
                target = speech_model(speech.to(device))
                pos = speech_model(speech_pos.to(device))
                neg = speech_model(speech_neg.to(device))
                marker = ["aaa"]
            elif case == 3:
                train_fout.write(f'{speech_pos_instance},{speech_neg_instance},vss,')
                target = vision_model(vision.to(device))
                pos = speech_model(speech_pos.to(device))
                neg = speech_model(speech_neg.to(device))
                marker = ["baa"]
            elif case == 4:
                train_fout.write(f'{vision_pos_instance},{vision_neg_instance},svv,')
                target = speech_model(speech.to(device))
                pos = vision_model(vision_pos.to(device))
                neg = vision_model(vision_neg.to(device))
                marker = ["abb"]
            elif case == 5:
                train_fout.write(f'{vision_pos_instance},{speech_neg_instance},vvs,')
                target = vision_model(vision.to(device))
                pos = vision_model(vision_pos.to(device))
                neg = speech_model(speech_neg.to(device))
                marker = ["bba"]
            elif case == 6:
                train_fout.write(f'{speech_pos_instance},{vision_neg_instance},ssv,')
                target = speech_model(speech.to(device))
                pos = speech_model(speech_pos.to(device))
                neg = vision_model(vision_neg.to(device))
                marker = ["aab"]
            elif case == 7:
                train_fout.write(f'{speech_pos_instance},{vision_neg_instance},vsv,')
                target = vision_model(vision.to(device))
                pos = speech_model(speech_pos.to(device))
                neg = vision_model(vision_neg.to(device))
                marker = ["bab"]
            elif case == 8:
                train_fout.write(f'{vision_pos_instance},{speech_neg_instance},svs,')
                target = speech_model(speech.to(device))
                pos = vision_model(vision_pos.to(device))
                neg = speech_model(speech_neg.to(device))
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
            speech_optimizer.step()
            vision_optimizer.step()

            batch_loss.append(loss.item())
            epoch_loss += loss.item()

            if not step % (len(train_data) // 32):
                print(f'epoch: {epoch + 1}, batch: {step + 1}, loss: {loss.item()}')

        # Save networks after each epoch
        torch.save(speech_model.state_dict(), os.path.join(train_results_dir, 'speech_model.pt'))
        torch.save(vision_model.state_dict(), os.path.join(train_results_dir, 'vision_model.pt'))

        # Save loss data
        with open(os.path.join(train_results_dir, 'batch_loss.pkl'), 'wb') as fout:
            pickle.dump(batch_loss, fout)

        avg_epoch_loss.append(epoch_loss / len(train_data))
        with open(os.path.join(train_results_dir, 'avg_epoch_loss.pkl'), 'wb') as fout:
            pickle.dump(avg_epoch_loss, fout)
        print(f'Starting evaluation: {datetime.datetime.now().time()}')

        language2language_fout = open(os.path.join(results_dir, 'language2language_test_epoch_'+str(epoch)+'.txt'), 'w')
        language2language_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
        for language_index, language in enumerate(language_test_data):
            positive_indices = [i for i, name in enumerate(instance_names_test) if language[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_test) if language[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            language_data = language[0].to(device)
            embedded_language = speech_model(language_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_language = language_test_data[i]
                pos_language_data = pos_language[0].to(device)
                embedded_pos_language = speech_model(pos_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_language, embedded_pos_language)
                language2language_fout.write(f'{language[1]},{pos_language[1]},p,{dist}\n')

            for i in negative_indices:
                neg_language = language_test_data[i]
                neg_language_data = neg_language[0].to(device)
                embedded_neg_language = speech_model(neg_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_language, embedded_neg_language)
                language2language_fout.write(f'{language[1]},{neg_language[1]},n,{dist}\n')
        language2language_fout.close()
        print(f'Wrote language2language: {datetime.datetime.now().time()}')

        vision2vision_fout = open(os.path.join(results_dir, 'vision2vision_test_epoch_'+str(epoch)+'.txt'), 'w')
        vision2vision_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
        for vision_index, vision in enumerate(vision_test_data):
            positive_indices = [i for i, name in enumerate(instance_names_test) if vision[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_test) if vision[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            vision_data = vision[0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_vision = vision_test_data[i]
                pos_vision_data = pos_vision[0].to(device)
                embedded_pos_vision = vision_model(pos_vision_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_vision)
                vision2vision_fout.write(f'{vision[1]},{pos_vision[1]},p,{dist}\n')

            for i in negative_indices:
                neg_vision = vision_test_data[i]
                neg_vision_data = neg_vision[0].to(device)
                embedded_neg_vision = vision_model(neg_vision_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_vision)
                vision2vision_fout.write(f'{vision[1]},{neg_vision[1]},n,{dist}\n')
        vision2vision_fout.close()
        print(f'Wrote vision2vision: {datetime.datetime.now().time()}')

        vision2language_fout = open(os.path.join(results_dir, 'vision2language_test_epoch_'+str(epoch)+'.txt'), 'w')
        vision2language_fout.write('vision_instance,language_instance,p/n,embedded_distance\n')
        for vision_index, vision in enumerate(vision_test_data):
            positive_indices = [i for i, name in enumerate(instance_names_test) if vision[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_test) if vision[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            vision_data = vision[0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_language = language_test_data[i]
                pos_language_data = pos_language[0].to(device)
                embedded_pos_language = speech_model(pos_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_language)
                vision2language_fout.write(f'{vision[1]},{pos_language[1]},p,{dist}\n')

            for i in negative_indices:
                neg_language = language_test_data[i]
                neg_language_data = neg_language[0].to(device)
                embedded_neg_language = speech_model(neg_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_language)
                vision2language_fout.write(f'{vision[1]},{neg_language[1]},n,{dist}\n')
        vision2language_fout.close()

        language2language_fout = open(os.path.join(results_dir, 'language2language_dev_epoch_'+str(epoch)+'.txt'), 'w')
        language2language_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
        for language_index, language in enumerate(language_dev_data):
            positive_indices = [i for i, name in enumerate(instance_names_dev) if language[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_dev) if language[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            language_data = language[0].to(device)
            embedded_language = speech_model(language_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_language = language_dev_data[i]
                pos_language_data = pos_language[0].to(device)
                embedded_pos_language = speech_model(pos_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_language, embedded_pos_language)
                language2language_fout.write(f'{language[1]},{pos_language[1]},p,{dist}\n')

            for i in negative_indices:
                neg_language = language_dev_data[i]
                neg_language_data = neg_language[0].to(device)
                embedded_neg_language = speech_model(neg_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_language, embedded_neg_language)
                language2language_fout.write(f'{language[1]},{neg_language[1]},n,{dist}\n')
        language2language_fout.close()
        print(f'Wrote language2language: {datetime.datetime.now().time()}')

        vision2vision_fout = open(os.path.join(results_dir, 'vision2vision_dev_epoch_'+str(epoch)+'.txt'), 'w')
        vision2vision_fout.write('instance_name_1,instance_name_2,p/n,embedded_distance\n')
        for vision_index, vision in enumerate(vision_dev_data):
            positive_indices = [i for i, name in enumerate(instance_names_dev) if vision[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_dev) if vision[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            vision_data = vision[0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_vision = vision_dev_data[i]
                pos_vision_data = pos_vision[0].to(device)
                embedded_pos_vision = vision_model(pos_vision_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_vision)
                vision2vision_fout.write(f'{vision[1]},{pos_vision[1]},p,{dist}\n')

            for i in negative_indices:
                neg_vision = vision_dev_data[i]
                neg_vision_data = neg_vision[0].to(device)
                embedded_neg_vision = vision_model(neg_vision_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_vision)
                vision2vision_fout.write(f'{vision[1]},{neg_vision[1]},n,{dist}\n')
        vision2vision_fout.close()
        print(f'Wrote vision2vision: {datetime.datetime.now().time()}')

        vision2language_fout = open(os.path.join(results_dir, 'vision2language_dev_epoch_'+str(epoch)+'.txt'), 'w')
        vision2language_fout.write('vision_instance,language_instance,p/n,embedded_distance\n')
        for vision_index, vision in enumerate(vision_dev_data):
            positive_indices = [i for i, name in enumerate(instance_names_dev) if vision[1] == name]
            negative_indices = random.sample([i for i, name in enumerate(instance_names_dev) if vision[1] != name],
                                             len(positive_indices))
            if sample_size:
                positive_indices = random.sample(positive_indices, min(len(positive_indices), sample_size))
                negative_indices = random.sample(negative_indices, min(len(negative_indices), sample_size))

            vision_data = vision[0].to(device)
            embedded_vision = vision_model(vision_data).cpu().detach().numpy()

            for i in positive_indices:
                pos_language = language_dev_data[i]
                pos_language_data = pos_language[0].to(device)
                embedded_pos_language = speech_model(pos_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_pos_language)
                vision2language_fout.write(f'{vision[1]},{pos_language[1]},p,{dist}\n')

            for i in negative_indices:
                neg_language = language_dev_data[i]
                neg_language_data = neg_language[0].to(device)
                embedded_neg_language = speech_model(neg_language_data).cpu().detach().numpy()
                dist = scipy.spatial.distance.cosine(embedded_vision, embedded_neg_language)
                vision2language_fout.write(f'{vision[1]},{neg_language[1]},n,{dist}\n')
        vision2language_fout.close()

        print('***** epoch is finished *****')
        print(f'epoch: {epoch + 1}, loss: {avg_epoch_loss[epoch]}')

        speech_scheduler.step()
        vision_scheduler.step()

    print('Training done!')
    train_fout.close()

def main():
    ARGS, unused = parse_args()
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    train(
        experiment_name=ARGS.experiment,
        epochs=ARGS.epochs,
        train_data_path=ARGS.train_data,
        dev_data_path=ARGS.dev_data,
        test_data_path=ARGS.test_data,
        pos_neg_examples_file=ARGS.pos_neg_examples_file,
        batch_size=ARGS.batch_size,
        embedded_dim=ARGS.embedded_dim,
        gpu_num=ARGS.gpu_num,
        seed=ARGS.seed,
        num_layers=ARGS.num_layers,
        h=ARGS.h,
        awe=ARGS.awe,
    )

if __name__ == '__main__':
    main()
