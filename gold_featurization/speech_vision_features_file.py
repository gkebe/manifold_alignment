# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:22:01 2020

@author: T530
"""
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import torch
import flair
import skimage
from dataset import GLD_Instances
import pickle
import argparse

#astype = lambda x: np.array(x).astype(np.uint8)
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = torchvision.transforms.Resize((224,224))
transform_rgbd = torchvision.transforms.Compose([
                                                torchvision.transforms.ToPILImage(),
                                                resize,
                                                torchvision.transforms.ToTensor(),
                                                normalize])

depth2rgb = lambda x: skimage.img_as_ubyte(skimage.color.gray2rgb(x/np.max(x)))
transform_depth = torchvision.transforms.Compose([
                                                depth2rgb,
                                                torchvision.transforms.ToPILImage(),
                                                resize,
                                                torchvision.transforms.ToTensor(),
                                                normalize])

dataset = GLD_Instances("gold/speech.tsv", "images",speech=True,transform_rgb=transform_rgbd,transform_depth=transform_depth)
#dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='wav2vec_features.pkl', help='speech features file')

    return parser.parse_known_args()

ARGS, unused = parse_args()
with open(ARGS.features,'rb') as f:
    speech_features = pickle.load(f, encoding='bytes')

with open("gld_vision_features.pkl",'rb') as f:
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
