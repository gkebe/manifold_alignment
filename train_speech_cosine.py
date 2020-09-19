import argparse
import os
import pickle
import random

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
from attention import Combiner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='path to experiment output')
    parser.add_argument('--epochs', default=1, type=int,
        help='number of epochs to train')
    parser.add_argument('--train_data', help='path to train data')
    parser.add_argument('--gpu_num', default='0', help='gpu id number')
    parser.add_argument('--pos_neg_examples_file',
        help='path to pos/neg examples pkl')
    parser.add_argument('--seed', type=int, default=75,
        help='random seed for reproducability')
    parser.add_argument('--embedded_dim', default=1024, type=int,
        help='dimension of embedded manifold')
    parser.add_argument('--batch_size', type=int, default=1,
        help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
        help='number of hidden units to keep')
    parser.add_argument("--lstm",
                        action='store_true',
                        help="Whether to use a lstm.")
    parser.add_argument('--num_layers', type=int, default=1,
        help='number of lstm hidden layers')
    parser.add_argument("--mean_pooling",
                        action='store_true',
                        help="Whether to use mean pooling on the lstm's output.")

    return parser.parse_known_args()

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

def train(experiment_name, epochs, train_data_path, pos_neg_examples_file, batch_size, embedded_dim, gpu_num, seed, margin=0.4, lr=0.001, lstm=False, num_layers=1, mean_pooling=False):
    def lr_lambda(e):
        if e < 20:
            return lr
        elif e < 40:
            return lr * 0.1
        else:
            return lr * 0.01
    results_dir = f'./output/{experiment_name}'
    os.makedirs(results_dir, exist_ok=True)

    train_results_dir = os.path.join(results_dir, 'train_results/')
    os.makedirs(train_results_dir, exist_ok=True)

    train_fout = open(os.path.join(train_results_dir, 'train_out.txt'), 'w')

    print(f'cuda:{gpu_num}; cuda is available? {torch.cuda.is_available()}')
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    with open(train_data_path, 'rb') as fin:
        train_data = pickle.load(fin)

    speech_train_data = [s for s, _, _, _ in train_data]
    vision_train_data = [v for _, v, _, _ in train_data]
    instance_names = [i for _, _, _, i in train_data]

    with open(pos_neg_examples_file, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)

    # TODO: grab speech dimension from speech data tensor
    # TODO: set some of these from ARGS
    speech_dim = 40
    if lstm:
        speech_model = LSTM(
            input_size=list(speech_train_data[0].size())[1],
            output_size=embedded_dim,
            hidden_dim=list(speech_train_data[0].size())[1],
            num_layers=num_layers,
            mean_pooling=mean_pooling,
            device=device,
        )
    else:
        speech_model = Combiner(list(speech_train_data[0].size())[1], embedded_dim)

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
    speech_optimizer = torch.optim.Adam(speech_model.parameters(), lr=lr)
    vision_optimizer = torch.optim.Adam(vision_model.parameters(), lr=lr)

    speech_scheduler = torch.optim.lr_scheduler.LambdaLR(speech_optimizer, lr_lambda)
    vision_scheduler = torch.optim.lr_scheduler.LambdaLR(vision_optimizer, lr_lambda)

    speech_model.train()
    vision_model.train()
    
    batch_loss = []
    avg_epoch_loss = []

    train_fout.write('epoch,step,target,pos,neg,case,pos_dist,neg_dist,loss\n')
    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            train_fout.write(f'{epoch},{step},')
            speech, vision, object_name, instance_name = batch
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
            speech = speech
            speech_pos = speech_pos
            speech_neg = speech_neg

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
        
        print('***** epoch is finished *****')
        print(f'epoch: {epoch + 1}, loss: {avg_epoch_loss[epoch]}')

        speech_scheduler.step()
        vision_scheduler.step()

    print('Training done!')
    train_fout.close()

def main():
    ARGS, unused = parse_args()
    train(
        experiment_name=ARGS.experiment,
        epochs=ARGS.epochs,
        train_data_path=ARGS.train_data,
        pos_neg_examples_file=ARGS.pos_neg_examples_file,
        batch_size=ARGS.batch_size,
        embedded_dim=ARGS.embedded_dim,
        gpu_num=ARGS.gpu_num,
        seed=ARGS.seed,
        lr=ARGS.lr,
        lstm=ARGS.lstm,
        num_layers=ARGS.num_layers,
        mean_pooling=ARGS.mean_pooling,
    )

if __name__ == '__main__':
    main()
