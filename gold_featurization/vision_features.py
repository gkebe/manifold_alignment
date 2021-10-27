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
import sys
sys.path.append('../')
from datasets import GLD_Dataset
import pickle
import argparse

def setup_device(gpu_num=0):
    """Setup device."""
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'  # Is there a GPU? 
    device = torch.device(device_name)
    return device
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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_depth', dest='no_dpeth', action='store_true',
                        help="extract RGB features only.")
    return parser.parse_known_args()

ARGS, unused = parse_args()

dataset = GLD_Dataset("gold/speech.tsv", "images",speech=True,transform_rgb=transform_rgbd,transform_depth=transform_depth)
#dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)
print(len(dataset))

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

device = setup_device(gpu_num=0)
vision_model = torchvision.models.resnet152(pretrained=True)
vision_model.fc = Identity()
vision_model.to(device)
vision_model.eval()

vision_data = []
depth_data = []
object_names = []
instance_names = []
file_names = []
for i in tqdm(range(0, len(dataset)), desc="Instance"):
    if dataset[i][5] not in file_names:
        vision_data.append(vision_model(dataset[i][1].unsqueeze_(0).to(device)).detach().to('cpu'))
        depth_data.append(vision_model(dataset[i][2].unsqueeze_(0).to(device)).detach().to('cpu'))
        object_names.append(dataset[i][3])
        instance_names.append(dataset[i][4])
        file_names.append(dataset[i][5])
if ARGS.no_depth:
    vision_data_ = vision_data
else:
    vision_data_ = np.concatenate((np.squeeze([i.numpy() for i in depth_data]),np.squeeze([i.numpy() for i in vision_data])),axis=1)
vision_data__ = [torch.tensor(i) for i in vision_data_]
data = dict()
for i in range(len(instance_names)):
    data[file_names[i]] = vision_data__[i]
print(len(data))
import pickle
with open("gld_vision_features.pkl", 'wb') as f:
    pickle.dump(data, f)
