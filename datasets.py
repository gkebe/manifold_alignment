from collections import Counter
import pickle
import random

import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN
import pandas as pd

class DataDomain(Dataset):
    # Adapted from: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py

    def __init__(self, dataset, datasetname):
        self.dataset = dataset
        self.datasetname = datasetname
        if datasetname == 'svhn':
            self.labels = np.array(self.dataset.labels)
        else:
            self.labels = np.array(self.dataset.targets)
        self.data = self.dataset.data
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

    def __getitem__(self, index):

        # Determines what portion of dataset to sample from.
        mid = len(self.dataset) // 2
        if index < mid:
            marker = 'a'  # Domain A is the anchor.
        else:
            marker = 'b'  # Domain B is the anchor.

        img1, label1 = self.data[index], self.labels[index]
        positive_index = index
        if marker == 'a':
            while (positive_index == index):
                positive_index = np.random.choice([e for e in self.label_to_indices[label1] if e >= mid])
        elif marker == 'b':
            while (positive_index == index):
                positive_index = np.random.choice([e for e in self.label_to_indices[label1] if e < mid])

        img2 = self.data[positive_index]

        return (img1, img2, label1, marker)

    def __len__(self):
        return len(self.dataset)

def gl_loaders(data_location, negatives=None, num_workers=8, pin_memory=True, batch_size=1, batch_sampler=None, shuffle=False, seed=None):
    with open(data_location, 'rb') as fin:
        data = pickle.load(fin)

    train, test = gl_train_test_split(data, train_percentage=0.8, seed=seed)

    train_data = GLData(train)
    test_data = GLData(test)

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'batch_size': batch_size,
        'batch_sampler': batch_sampler,
        'shuffle': shuffle
    }

    return DataLoader(train_data, **kwargs), DataLoader(test_data, **kwargs)

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
    unique_object_names = list(set(data['object_names']))
    for object_name in unique_object_names:
        train_indices += random.sample(
            [i for i, name in enumerate(data['object_names']) if name == object_name],
            int(train_percentage * data['object_names'].count(object_name))
        )
    test_indices = [i for i in range(len(data['object_names'])) if i not in train_indices]

    train['language_data'] = [data['language_data'][i] for i in train_indices]
    train['vision_data'] = [data['vision_data'][i] for i in train_indices]
    train['object_names'] = [data['object_names'][i] for i in train_indices]
    train['instance_names'] = [data['instance_names'][i] for i in train_indices]

    test['language_data'] = [data['language_data'][i] for i in test_indices]
    test['vision_data'] = [data['vision_data'][i] for i in test_indices]
    test['object_names'] = [data['object_names'][i] for i in test_indices]
    test['instance_names'] = [data['instance_names'][i] for i in test_indices]

    return train, test

class GLData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['object_names'])

    def __getitem__(self, i):
        return self.data['language_data'][i], self.data['vision_data'][i], self.data['object_names'][i], self.data['instance_names'][i], self.data['user_ids'][i]

class GLDataBias(Dataset):
    def __init__(self, dataset, speakers_tsv):
        self.data = dataset.data
        speakers_df = pd.read_csv(speakers_tsv, sep="\t")
        speakers_dict = speakers_df.set_index("worker_id").T.to_dict()
        gender_dict = {"man": 0, "woman": 1, "undet": 2}
        others_dict = {"no": 0, "yes": 1}
        speaker_ids = self.data["user_ids"]

        speaker_data = [[speakers_dict[s]["accent"], speakers_dict[s]["gender"], speakers_dict[s]["hoarsenes"],
                               speakers_dict[s]["creak"]] for s in speaker_ids]
        speaker_data = [[others_dict[a], gender_dict[g], others_dict[h], others_dict[c]] for a, g, h, c in
                              speaker_data]
        self.data["speaker_data"] = speaker_data
    def __len__(self):
        return len(self.data['object_names'])

    def __getitem__(self, i):

        return self.data['language_data'][i], self.data['vision_data'][i], self.data['object_names'][i], self.data['instance_names'][i], self.data['user_ids'][i], self.data["speaker_data"][i]
