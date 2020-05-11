
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        """ encoder """
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(32, 64, kernel_size=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """ target-decoder """
        self.targetDecoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose2d(32, 3, kernel_size=(4, 4)),
            nn.Sigmoid(),
        )

        """ source-decoder """
        self.sourceDecoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose2d(32, 3, kernel_size=(4, 4)),
            nn.Sigmoid(),
        )

    def _visualize_features(self, feature_maps, dim: tuple=(), title: str=""):
        print(feature_maps.shape)
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

    def forward(self, x, label: str="0", visualize: bool=False):
        if not visualize:
            encoding = self.encoder(x)

            # if the input image is a target-image: use the target-decoder
            if label == "0": output = self.targetDecoder(encoding)

            # if the input image is a source-image: use the source-decoder
            elif label == "1": output = self.sourceDecoder(encoding)

            return output

        # visualize feature maps if wanted
        else:
            i = 0
            for layer in self.encoder:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d):
                    i += 1
                    self._visualize_features(x, dim=(4, 4), title=(str(i) + " conv2d"))
                    
            # if the input image is a target-image: use the target-decoder
            if label == "0":
                i = 0
                for layer in self.targetDecoder:
                    x = layer(x)
                    if isinstance(layer, nn.ConvTranspose2d):
                        i += 1
                        self._visualize_features(x, dim=(2, 2), title=(str(i) + " convTrans2d"))

                return x

            # if the input image is a source-image: use the source-decoder
            elif label == "1":
                i = 0
                for layer in self.sourceDecoder:
                    x = layer(x)
                    if isinstance(layer, nn.ReLU):
                        i += 1
                        self._visualize_features(x, dim=(1, 1), title=(str(i) + " convTrans2d"))
                
                return x
            
        

