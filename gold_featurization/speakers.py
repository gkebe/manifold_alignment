# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:22:01 2020

@author: T530
"""
from tqdm import tqdm
import numpy as np
import skimage
import pickle
import argparse
import pandas as pd
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='wav2vec_features.pkl', help='speech features file')

    return parser.parse_known_args()

ARGS, unused = parse_args()
"""

with open("gold/speakers.tsv",'rb') as csv_file:
     speakers = pd.read_csv(csv_file, delimiter="\t", keep_default_na=False, na_values=['_'])
for trait in speakers.keys():
    print(f"{trait}:")
    if trait in ["worker_id", "num_examples"]:
        continue
    for val in list(set(speakers[trait])):
        print(f"{val}: {len(speakers[speakers[trait] == val])}")
        print(list(speakers[speakers[trait] == val]["worker_id"]))
        print()
    print()

"""
\with open("gld_vision_features.pkl",'rb') as f:
    vision_features = pickle.load(f, encoding='bytes')

language_data = []
vision_data = []
depth_data = []
object_names = []
instance_names = []
user_ids = []
image_names = []
for i in tqdm(range(0, len(dataset)), desc="Instance"):
    language_data.append(speech_features[dataset[i][0]].detach().cpu())
    vision_data.append(vision_features[dataset[i][1]])
    object_names.append(dataset[i][3])
    instance_names.append(dataset[i][4])
    user_ids.append(dataset[i][6])
    image_names.append(dataset[i][7])

data = {'language_data':language_data,
        'vision_data':vision_data,
        'object_names':object_names,
        'instance_names':instance_names,
        'user_ids':user_ids,
        'image_names':image_names
       }

import pickle
with open('../data/gld_'+ARGS.features.replace("_features.pkl", "")+'_vision_tensors.pkl', 'wb') as f:
    pickle.dump(data, f)
"""