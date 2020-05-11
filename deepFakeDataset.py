
import numpy as np
from torchvision import transforms
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import os
from imgaug import augmenters as iaa
import random


class DeepFakeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str=""):
        self.dataset_path = dataset_path
        self.dataset = os.listdir(dataset_path)

    def _augmentate(self, image, chance: float=0.5):
        if chance > random.random():
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                iaa.LinearContrast((0.70, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.4),
                iaa.Multiply((0.8, 1.2), per_channel=0.2)
            ], random_order=True)

            image = seq(images=[image])[0]
        return image
        
    def __getitem__(self, idx):
        label = self.dataset[idx].split("_")[0]

        image = Image.open(self.dataset_path + self.dataset[idx]).convert("RGB")
        image = np.array(image)
        image = self._augmentate(image, chance=0.2)
        image = (image / 255)

        image = torch.Tensor(image).view(3, image.shape[0], image.shape[1])

        return image, label

    def __len__(self):
        return len(self.dataset)

