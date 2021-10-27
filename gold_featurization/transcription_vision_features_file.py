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
import pickle
from dataset import GLD_Instances
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

dataset = GLD_Instances("gold/speech.tsv", "images",transcriptions=True,transform_rgb=transform_rgbd,transform_depth=transform_depth)

#dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)
print(len(dataset))

document_embeddings = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])

def proc_sentence(t):
    sentence = flair.data.Sentence(t, use_tokenizer=True)
    document_embeddings.embed(sentence)
    return sentence.get_embedding()

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
    language_data.append(proc_sentence(dataset[i][0]).detach().cpu())
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

with open('../data/gld_transcriptions_vision_tensors.pkl', 'wb') as f:
    pickle.dump(data, f)
