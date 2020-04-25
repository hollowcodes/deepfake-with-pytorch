
import numpy as np
from torchvision import transforms
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import os


class DeepFakeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str=""):
        self.dataset_path = dataset_path
        self.dataset = os.listdir(dataset_path)

        """self.transform = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.ToTensor()]),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])"""
    
    def _flip(self, image):
        pass
        
    def __getitem__(self, idx):
        # label = 0 for target image and 1 for source image
        label = self.dataset[idx].split("_")[0]

        image = np.array(Image.open(self.dataset_path + self.dataset[idx]).convert("RGB"))
        image = (image / 255)

        #image = self.transform(image).view(3, image.shape[0], image.shape[1])
        image = torch.Tensor(image).view(3, image.shape[0], image.shape[1])

        return image, label

    def __len__(self):
        return len(self.dataset)
