
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        """ encoder """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4))
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4))
        self.batchnorm4 = nn.BatchNorm2d(256)

        self.dense1 = nn.Linear(256*5*5, 128)
        self.dense2 = nn.Linear(128, 256*5*5)

        self.maxpool3x3 = nn.MaxPool2d(3)
        self.maxpool2x2 = nn.MaxPool2d(2)

        """ target-decoder """

        self.targetDeconv2 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4))
        self.targetBatchnorm2 = nn.BatchNorm2d(128)

        self.targetDeconv3 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4))
        self.targetBatchnorm3 = nn.BatchNorm2d(64)

        self.targetDeconv4 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 5))
        self.targetBatchnorm4 = nn.BatchNorm2d(32)

        self.targetDeconv5 = nn.ConvTranspose2d(32, 3, kernel_size=(5, 5))

        self.upsample3x3 = nn.Upsample(scale_factor=3)
        self.upsample2x2 = nn.Upsample(scale_factor=2)

        """ source-decoder """

        self.sourceDeconv2 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4))
        self.sourceBatchnorm2 = nn.BatchNorm2d(128)

        self.sourceDeconv3 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4))
        self.sourceBatchnorm3 = nn.BatchNorm2d(64)

        self.sourceDeconv4 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 5))
        self.sourceBatchnorm4 = nn.BatchNorm2d(32)

        self.sourceDeconv5 = nn.ConvTranspose2d(32, 3, kernel_size=(5, 5))

        self.upsample3x3 = nn.Upsample(scale_factor=3)
        self.upsample2x2 = nn.Upsample(scale_factor=2)

    def _visualize_features(self, feature_maps, dim: tuple=(), title: str=""):
        try:
            x, y = dim
            fig, axs = plt.subplots(x, y)
            c = 0
            for i in range(x):
                for j in range(y):
                    axs[i][j].matshow(feature_maps.detach().cpu().numpy()[0][c])
                    c += 1

            fig.suptitle(title)
            plt.show()

        except:
            pass

    def _gaussian_noise_layer(self, x, fixed_gaussian_noise_rate=False):
        gaussian_noise_rate = np.random.choice([0.03, 0.05, 0.0, 0.0, 0.00, 0.01])
        if fixed_gaussian_noise_rate is not False:
            gaussian_noise_rate = fixed_gaussian_noise_rate

        x += gaussian_noise_rate * torch.randn(1, 128, 128).cuda()
        return x        

    def forward(self, x, label: str="0", visualize: bool=False, print_: bool=False, mode: str="train"):
        if mode == "train":
            x = self._gaussian_noise_layer(x, fixed_gaussian_noise_rate=False)
        elif mode == "test":
            x = self._gaussian_noise_layer(x, fixed_gaussian_noise_rate=0.0)

        if print_: print(x.shape)
        batch_size = x.size()[0]

        """ encoder """
        x = self.conv1(x)
        x = F.relu(x)
        x_1 = self.maxpool2x2(x)

        if print_: print(x_1.shape)
        if visualize: self._visualize_features(x_1, dim=(4, 8))

        x = self.conv2(x_1)
        x = F.relu(x)
        x_2 = self.maxpool2x2(x)

        if print_: print(x_2.shape)
        if visualize: self._visualize_features(x_2, dim=(8, 8))

        x = self.conv3(x_2)
        x = F.relu(x)
        x_3 = self.maxpool2x2(x)

        if print_: print(x_3.shape)
        if visualize: self._visualize_features(x_3, dim=(8, 8))

        x = self.conv4(x_3)
        x = F.relu(x)
        x_4 = self.maxpool2x2(x)

        if print_: print(x_4.shape)
        if visualize: self._visualize_features(x_4, dim=(8, 8))

        x = x_4.view(-1, 256*5*5)
        x = F.relu(self.dense1(x))
        if print_: print(x.shape)

        x = F.relu(self.dense2(x))
        x = x.reshape(batch_size, 256, 5, 5)
        if print_: print(x.shape)

        """ target-decoder """
        if label == "0":
            x = self.upsample2x2(x)
            x = self.targetDeconv2(x)
            x = self.targetBatchnorm2(x)
            x = F.relu(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(8, 8))

            x = self.upsample2x2(x)
            x = self.targetDeconv3(x)
            x = self.targetBatchnorm3(x)
            x = F.relu(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(8, 8))

            x = self.upsample2x2(x)
            x = self.targetDeconv4(x)
            x = self.targetBatchnorm4(x)
            x = F.relu(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(4, 8))

            x = self.upsample2x2(x)
            x = self.targetDeconv5(x)
            x = torch.sigmoid(x)

            if print_: print(x.shape)

            return x

        """ source-decoder """
        if label == "1":
            x = self.upsample2x2(x)
            x = self.sourceDeconv2(x)
            x = self.sourceBatchnorm2(x)
            x = F.relu(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(4, 4))

            x = self.upsample2x2(x)
            x = self.sourceDeconv3(x)
            x = self.sourceBatchnorm3(x)
            x = F.relu(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(4, 4))

            x = self.upsample2x2(x)
            x = self.sourceDeconv4(x)
            x = self.sourceBatchnorm4(x)
            x = F.relu(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(4, 4))

            x = self.upsample2x2(x)
            x = self.sourceDeconv5(x)
            x = torch.sigmoid(x)

            if print_: print(x.shape)
            if visualize: self._visualize_features(x, dim=(3, 1))

            return x



"""x = torch.Tensor(torch.rand((1, 3, 128, 128))).cuda()

model = Model().cuda()
x = model.forward(x, label="0", print_=True)

"""