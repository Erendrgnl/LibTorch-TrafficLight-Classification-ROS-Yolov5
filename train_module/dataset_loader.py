import os
import numpy as np
import glob
from random import shuffle
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

class CustomImageDataset(Dataset):
    def __init__(self, dir_path,transform=None):
        self.transform = transform
        self.class_names = os.listdir(dir_path)
        self.num_class = len(self.class_names)
        
        print("Found {} class : {}".format(self.num_class,self.class_names))

        self.images_path = []
        self.target = np.array([])

        for i in range(self.num_class):
            files = self.jpg_loader(os.path.join(dir_path,self.class_names[i]))
            print("{} images found belonging to the class {}".format(len(files),self.class_names[i]))
            self.images_path += files
            self.target = np.append(self.target , np.ones(len(files)) * i) 

        self.target = torch.tensor(self.target,dtype=torch.long)
        self.total_images = len(self.images_path)

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        image = read_image(self.images_path[idx])
        image = image/255
        if self.transform:
            image = self.transform(image)
        label = self.target[idx]
        return image.float(), label

    @staticmethod
    def jpg_loader(path): 
        return glob.glob(os.path.join(path,'*.jpg'))


if __name__ == "__main__":
    dt = CustomImageDataset("traffic_lights/")