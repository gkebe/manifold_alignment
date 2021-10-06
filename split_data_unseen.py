import argparse
import pickle
import random
from ordered_set import OrderedSet

from torch.utils.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data pkl file')
    parser.add_argument('--train', default='train_data.pkl', help='train data out path')
    parser.add_argument('--dev', default='dev_data.pkl', help='dev data out path')
    parser.add_argument('--test', default='test_data.pkl', help='test data out path')
    parser.add_argument('--train_percentage', type=float, default=0.8, help='percentage of data dedicated to train')
    parser.add_argument('--seed', type=int, default=75, help='Random seed for split reproducability')

    return parser.parse_known_args()

def gl_dataset(data_location, train_percentage=0.6, seed=None):
    with open(data_location, 'rb') as fin:
        data = pickle.load(fin)

    train, dev_test = gl_train_test_split(data, train_percentage=train_percentage, seed=seed)
    dev ,test = gl_train_test_split(dev_test, train_percentage=0.5, seed=seed)

    train_data = GLData(train)
    dev_data = GLData(dev)
    test_data = GLData(test)

    # kwargs = {
    #     'num_workers': num_workers,
    #     'pin_memory': pin_memory,
    #     'batch_size': batch_size,
    #     'batch_sampler': batch_sampler,
    #     'shuffle': shuffle
    # }

    return train_data, dev_data, test_data

def gl_train_test_split(data, train_percentage=0.8, seed=None):
    """
    Splits a grounded language dictionary into training and testing sets.

    data needs the following keys:
    language_data
    vision_data
    object_names
    instance_names
    """
    random.seed(seed)

    train = {}
    test = {}

    # ensure test and train have some of every object
    train_indices = []
    unique_object_names = list(OrderedSet(data['object_names']))

    train_objects = random.sample(unique_object_names, int(train_percentage * len(unique_object_names)))
    
    #for object_name in unique_object_names:
    #    train_indices += random.sample(
    #        [i for i, name in enumerate(data['object_names']) if name == object_name],
    #        int(train_percentage * data['object_names'].count(object_name))
    #    )
    train_indices = [i for i in range(len(data['object_names'])) if data['object_names'][i] in train_objets]
    test_indices = [i for i in range(len(data['object_names'])) if i not in train_indices]

    train['language_data'] = [data['language_data'][i] for i in train_indices]
    train['vision_data'] = [data['vision_data'][i] for i in train_indices]
    train['object_names'] = [data['object_names'][i] for i in train_indices]
    train['instance_names'] = [data['instance_names'][i] for i in train_indices]
    train['image_names'] = [data['image_names'][i] for i in train_indices]
    train['user_ids'] = [data['user_ids'][i] for i in train_indices]   
    
    test['language_data'] = [data['language_data'][i] for i in test_indices]
    test['vision_data'] = [data['vision_data'][i] for i in test_indices]
    test['object_names'] = [data['object_names'][i] for i in test_indices]
    test['instance_names'] = [data['instance_names'][i] for i in test_indices]
    test['image_names'] = [data['image_names'][i] for i in test_indices]
    test['user_ids'] = [data['user_ids'][i] for i in test_indices]   


    return train, test

class GLData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['object_names'])

    def __getitem__(self, i):
        item = (
            self.data['language_data'][i],
            self.data['vision_data'][i],
            self.data['object_names'][i],
            self.data['instance_names'][i],
            self.data['image_names'][i],
            self.data['user_ids']
        )

        return item

def main():
    ARGS, unused = parse_args()

    with open(ARGS.data, 'rb') as fin:
        data = pickle.load(fin)

    train_data, dev_data, test_data = gl_dataset(ARGS.data, ARGS.train_percentage, seed=ARGS.seed)

    with open(ARGS.train, 'wb') as fout:
        pickle.dump(train_data, fout)

    with open(ARGS.dev, 'wb') as fout:
        pickle.dump(dev_data, fout)

    with open(ARGS.test, 'wb') as fout:
        pickle.dump(test_data, fout)

    print(f'Wrote three files\n\t{ARGS.train}\n\t{ARGS.dev}\n\t{ARGS.test}\n\t')
    print(f'Training examples: {len(train_data)}, Dev examples: {len(dev_data)}, Testing examples: {len(test_data)}')

if __name__ == '__main__':
    main()
