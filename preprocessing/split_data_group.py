import argparse
import pickle
import random
from collections import Counter
from ordered_set import OrderedSet

from torch.utils.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data pkl file')
    parser.add_argument('--train', default='train_data.pkl', help='train data out path')
    parser.add_argument('--test', default='test_data.pkl', help='test data out path')
    parser.add_argument('--train_percentage', type=float, default=0.8, help='percentage of data dedicated to train')
    parser.add_argument('--seed', type=int, default=75, help='Random seed for split reproducability')
    parser.add_argument('--users', default=None, help='which users to use to train/test')
    parser.add_argument('--train_limit', default=None, help='limit data for balance')
    parser.add_argument('--test_limit', default=None, help='limit data for balance')

    return parser.parse_known_args()

def gl_dataset(data_location, train_percentage=0.8, seed=None, user_ids=None, limit=None):
    with open(data_location, 'rb') as fin:
        data = pickle.load(fin)
    random.seed(seed)
    # make per user splits
    if user_ids is not None:
        print(len(user_ids))
        data_indicies = [i for i in range(len(data["user_ids"])) if data["user_ids"][i] in user_ids]
        print(len(data_indicies))
        data['language_data'] = [data['language_data'][i] for i in data_indicies]
        data['vision_data'] = [data['vision_data'][i] for i in data_indicies]
        data['object_names'] = [data['object_names'][i] for i in data_indicies]
        data['instance_names'] = [data['instance_names'][i] for i in data_indicies]
        data['image_names'] = [data['image_names'][i] for i in data_indicies]

    train, test = gl_train_test_split(data, train_percentage=train_percentage, seed=seed, limit=limit)

    train_data = GLData(train)
    test_data = GLData(test)

    # kwargs = {
    #     'num_workers': num_workers,
    #     'pin_memory': pin_memory,
    #     'batch_size': batch_size,
    #     'batch_sampler': batch_sampler,
    #     'shuffle': shuffle
    # }

    return train_data, test_data

def gl_train_test_split(data, train_percentage=0.8, seed=None, limit=None):
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
    train_images = []
    unique_object_names = list(OrderedSet(data['object_names']))
    unique_image_names = list(OrderedSet([(data['image_names'][i], data['object_names'][i]) for i in range(len(data['image_names']))]))
    user_count = Counter(data['object_names'])

    for object_name in unique_object_names:
        if user_count[object_name] > 1:
            train_images += random.sample(
                [i[0] for i in unique_image_names if object_name== i[1]],
                int(train_percentage * [i[1] for i in unique_image_names].count(object_name))
            )


    #for object_name in unique_object_names:
    #    train_indices += random.sample(
    #        [i for i, name in enumerate(data['object_names']) if name == object_name],
    #        int(train_percentage * data['object_names'].count(object_name))
    #    )
    train_indices = [i for i in range(len(data['object_names'])) if data['image_names'][i] in train_images and user_count[data['object_names'][i]] > 1]
    test_indices = [i for i in range(len(data['object_names'])) if i not in train_indices and user_count[data['object_names'][i]] > 1]
    print(len(train_indices))
    print(len(test_indices))

    if limit != None:
        train_indices = random.sample(train_indices, int(limit[0]))
        training_objects_set = list(OrderedSet([object_ for i, object_ in enumerate(data['object_names']) if i in train_indices]))
        for i in test_indices:
            if data['object_names'][i] not in training_objects_set:
                test_indices.remove(i)
        test_indices = random.sample(test_indices, int(limit[1]))

    train['language_data'] = [data['language_data'][i] for i in train_indices]
    train['vision_data'] = [data['vision_data'][i] for i in train_indices]
    train['object_names'] = [data['object_names'][i] for i in train_indices]
    train['instance_names'] = [data['instance_names'][i] for i in train_indices]
    train['image_names'] = [data['image_names'][i] for i in train_indices]

    test['language_data'] = [data['language_data'][i] for i in test_indices]
    test['vision_data'] = [data['vision_data'][i] for i in test_indices]
    test['object_names'] = [data['object_names'][i] for i in test_indices]
    test['instance_names'] = [data['instance_names'][i] for i in test_indices]
    test['image_names'] = [data['image_names'][i] for i in test_indices]

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
            self.data['user_ids'][i]
        )

        return item

def main():
    ARGS, unused = parse_args()

    with open(ARGS.data, 'rb') as fin:
        data = pickle.load(fin)


    if ARGS.users is None:
        users = None
    else:
        users = ARGS.users
        users = users.strip('[]').split(",")

    if ARGS.train_limit is None:
        limit = None
    else:
        limit = (ARGS.train_limit, ARGS.test_limit)
    train_data, test_data = gl_dataset(ARGS.data, ARGS.train_percentage, seed=ARGS.seed, user_ids=users, limit=limit)

    with open(ARGS.train, 'wb') as fout:
        pickle.dump(train_data, fout)

    with open(ARGS.test, 'wb') as fout:
        pickle.dump(test_data, fout)

    print(f'Wrote two files\n\t{ARGS.train}\n\t{ARGS.test}')
    print(f'Training examples: {len(train_data)}, Testing examples: {len(test_data)}')

if __name__ == '__main__':
    main()