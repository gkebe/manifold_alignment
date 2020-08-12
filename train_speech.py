import argparse
import os
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
#import torchaudio

from datasets import GLData

class RowNet(torch.nn.Module):
    def __init__(self, input_size, embed_dim=1024):
        super(RowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, input_size)
        self.fc3 = torch.nn.Linear(input_size, embed_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=.2)
        x = self.fc3(x)

        return x

# TODO: might want to try an LSTM
class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,
            drop_out, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Model layers
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers,
            batch_first=True, dropout=drop_out)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(self.n_layers, batch_size,
            self.hidden_dim).to(self.device)
        _, hidden = self.rnn(x, hidden)

        # This does some reshaping, might be an old idiom
        # see: https://discuss.pytorch.org/t/when-and-why-do-we-use-contiguous/47588
        hidden = hidden.contiguous().view(-1, self.hidden_dim)
        hidden = self.fc(hidden)
    
        return hidden

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

    return parser.parse_known_args()

def lr_lambda(e):
    if e < 20:
        return 0.001
    elif e < 40:
        return 0.0001
    else:
        return 0.00001

def get_examples_batch(pos_neg_examples, indices, train_data):
    examples = [pos_neg_examples[i] for i in indices]
    
    return (
        torch.stack([train_data[i[0]] for i in examples]),
        torch.stack([train_data[i[1]] for i in examples])
    )

def train(experiment_name, epochs, train_data_path, pos_neg_examples_file, batch_size, embedded_dim, gpu_num, seed, margin=0.4):

    results_dir = f'./output/{experiment_name}'
    os.makedirs(results_dir, exist_ok=True)

    train_results_dir = os.path.join(results_dir, 'train_results/')
    os.makedirs(train_results_dir, exist_ok=True)

    device = torch.device('cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    with open(train_data_path, 'rb') as fin:
        train_data = pickle.load(fin)

    speech_train_data = [s for s, _, _, _ in train_data]
    vision_train_data = [v for _, v, _, _ in train_data]

    with open(pos_neg_examples_file, 'rb') as fin:
        pos_neg_examples = pickle.load(fin)

    # TODO: grab speech dimension from speech data tensor
    # TODO: set some of these from ARGS
    speech_dim = 40
    speech_model = RNN(
        input_size=40,
        output_size=embedded_dim,
        hidden_dim=64,
        n_layers=1,
        drop_out=0.0,
        device=device
    )
    vision_dim = list(vision_train_data[0].size())[0]
    vision_model = RowNet(vision_dim, embed_dim=embedded_dim)

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

    speech_model.train()
    vision_model.train()
    
    batch_loss = []
    avg_epoch_loss = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            speech, vision, object_name, instance_name = batch

            indices = list(range(step * batch_size, min((step + 1) * batch_size, len(train_data))))
            speech_pos, speech_neg = get_examples_batch(pos_neg_examples, indices, speech_train_data)
            vision_pos, vision_neg = get_examples_batch(pos_neg_examples, indices, vision_train_data)

            speech_optimizer.zero_grad()
            vision_optimizer.zero_grad()

            # TODO: still using triplet loss?
            triplet_loss = torch.nn.TripletMarginLoss(margin=0.4, p=2)

            # TODO: JANKY STUFF FOR TENSOR SIZING AND ORDER ERRORS
            speech = speech[0].permute(0, 2, 1)
            speech_pos = speech_pos[0].permute(0, 2, 1)
            speech_neg = speech_neg[0].permute(0, 2, 1)

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
                loss = triplet_loss(
                    vision_model(vision.to(device)),
                    vision_model(vision_pos.to(device)),
                    vision_model(vision_neg.to(device))
                )
            elif case == 2:
                loss = triplet_loss(
                    speech_model(speech.to(device)),
                    speech_model(speech_pos.to(device)),
                    speech_model(speech_neg.to(device))
                )
            elif case == 3:
                loss = triplet_loss(
                    vision_model(vision.to(device)),
                    speech_model(speech_pos.to(device)),
                    speech_model(speech_neg.to(device))
                )
            elif case == 4:
                loss = triplet_loss(
                    speech_model(speech.to(device)),
                    vision_model(vision_pos.to(device)),
                    vision_model(vision_neg.to(device))
                )
            elif case == 5:
                loss = triplet_loss(
                    vision_model(vision.to(device)),
                    vision_model(vision_pos.to(device)),
                    speech_model(speech_neg.to(device))
                )
            elif case == 6:
                loss = triplet_loss(
                    speech_model(speech.to(device)),
                    speech_model(speech_pos.to(device)),
                    vision_model(vision_neg.to(device))
                )
            elif case == 7:
                loss = triplet_loss(
                    vision_model(vision.to(device)),
                    speech_model(speech_pos.to(device)),
                    vision_model(vision_neg.to(device))
                )
            elif case == 8:
                loss = triplet_loss(
                    speech_model(speech.to(device)),
                    vision_model(vision_pos.to(device)),
                    speech_model(speech_neg.to(device))
                )

            loss.backward()
            speech_optimizer.step()
            vision_optimizer.step()

            batch_loss.append(loss.item())
            epoch_loss += loss.item()
            avg_epoch_loss.append(epoch_loss / len(train_data))

            if not step % (len(train_data) // 32):
                print(f'epoch: {epoch + 1}, batch: {step + 1}, loss: {loss.item()}')
    
        # Save networks after each epoch
        torch.save(speech_model.state_dict(), os.path.join(train_results_dir, 'speech_model.pt'))
        torch.save(vision_model.state_dict(), os.path.join(train_results_dir, 'vision_model.pt'))

        # Save loss data
        with open(os.path.join(train_results_dir, 'batch_loss.pkl'), 'wb') as fout:
            pickle.dump(batch_loss, fout)

        with open(os.path.join(train_results_dir, 'avg_epoch_loss.pkl'), 'wb') as fout:
            pickle.dump(avg_epoch_loss, fout)

        print('***** epoch is finished *****')
        print(f'epoch: {epoch + 1}, loss: {avg_epoch_loss[epoch]}')

        speech_scheduler.step()
        vision_scheduler.step()

    print('Training done!')

def main():
    ARGS, unused = parse_args()

    train(
        ARGS.experiment,
        ARGS.epochs,
        ARGS.train_data,
        ARGS.pos_neg_examples_file,
        ARGS.batch_size,
        ARGS.embedded_dim,
        ARGS.gpu_num,
        ARGS.seed,
    )

if __name__ == '__main__':
    main()
