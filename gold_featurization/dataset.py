'''
Booz Allen Hamilton proprietary information. Do not upload this code to a public Github repo or to any other public location on the internet for legal reasons. If any questions regarding this, please contact Edward Raff and Andre Nguyen. Thanks!
'''


import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import re
#import cv2
from skimage import io

class GLD_Dataset(Dataset):

    def __init__(self, csv_file, root_dir, transcriptions=False, speech= False, transform_depth=None, transform_rgb=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset = pd.read_csv(csv_file, delimiter="\t", keep_default_na=False, na_values=['_'])
        self.images = dataset["item_id"]
        if speech:
            self.descriptions = dataset["wav"]
        else:
            if transcriptions:
                self.descriptions = dataset["transcription"]
            else:
                self.descriptions = dataset["text"]
        self.root_dir = root_dir
        self.transform_depth = transform_depth
        self.transform_rgb = transform_rgb
        self.speech = speech

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        description = self.descriptions [index]
        filename = self.images[index]+".png"
        object_name = self.getObjectName(filename)
        instance_name = self.getInstanceName(filename)

        depth_image_loc = self.root_dir + "/depth/" + object_name + "/" + instance_name + "/" + filename
        rgb_image_loc = self.root_dir + "/color/" + object_name + "/" + instance_name + "/" + filename

        #print(rgb_image_loc)
        depth_image = io.imread(depth_image_loc, as_gray=True) #cv2.imread(depth_image_loc, cv2.IMREAD_GRAYSCALE)
        rgb_image =  io.imread(rgb_image_loc, as_gray=False)#cv2.imread(rgb_image_loc)

        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)


        return description, rgb_image, depth_image, object_name, instance_name, filename

    def __len__(self):
        return len(self.images)

    def getVisionData(self, no_depth):
        filenames = list(set([i+".png" for i in self.images]))
        object_names = list(set([self.getObjectName(i) for i in filenames]))
        instance_names = list(set([self.getInstanceName(i) for i in filenames]))

        rgb_images = [self.transform_rgb(
            io.imread(self.root_dir + "/color/" + object_names[i] + "/" + instance_names[i] + "/" + filenames[i]
                      , as_gray=False)) for i in range(len(filenames))]  # cv2.imread(rgb_image_loc)
        if no_depth:
           return rgb_images
        depth_images = [self.transform_depth(io.imread(self.root_dir + "/depth/" + object_names[i] + "/" + instance_names[i] + "/" + filenames[i]
                                  , as_gray=True)) for i in range(len(filenames))]#cv2.imread(depth_image_loc, cv2.IMREAD_GRAYSCALE)
        return rgb_images, depth_images
    def getObjectName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(1)

    def getInstanceName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(0)

class GLD_Instances(Dataset):

    def __init__(self, csv_file, root_dir, transcriptions=False, speech= False, transform_depth=None, transform_rgb=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset = pd.read_csv(csv_file, delimiter="\t", keep_default_na=False, na_values=['_'])
        self.images = dataset["item_id"]
        if speech or transcriptions:
            self.audio_files = dataset["wav"]
        self.speech = speech
        self.transcriptions = transcriptions
        self.user_ids = dataset["worker_id"]
        if speech or transcriptions:
            self.descriptions = dataset["wav"]
        else:
            if transcriptions:
                self.descriptions = dataset["transcription"]
            else:
                self.descriptions = dataset["text"]
        self.root_dir = root_dir

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        description = self.descriptions [index]
        filename = self.images[index]+".png"
        object_name = self.getObjectName(filename)
        instance_name = self.getInstanceName(filename)
        if self.speech or self.transcriptions:
            audio_file = self.audio_files[index]
        user_id = self.user_ids[index]
        if self.speech or self.transcriptions:
            return description, filename, filename, object_name, instance_name, audio_file, user_id, self.images[index]
        else:
            return description, filename, filename, object_name, instance_name, None, user_id, self.images[index]
    def __len__(self):
        return len(self.images)

    def getObjectName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(1)

    def getInstanceName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(0)


def main():
    dataset = GLD_Dataset("text.tsv", "gold/images", False, False)
    print(dataset[1027])
    print(len(dataset))

if __name__ == '__main__':
    main()
