

from residual_model import Model
from deepFakeDataset import DeepFakeDataset

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from termcolor import colored

import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms



class RunModel:
    def __init__(self, dataset_path: str="", batch_size=64, epochs=100, lr=1e2):
        self.dataset_path = dataset_path

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.train_set = self._create_dataloader()

    """ creates dataloader """
    def _create_dataloader(self):
        dataset = DeepFakeDataset(dataset_path=self.dataset_path)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
        )

        return dataloader

    """ trains model """
    def train(self):
        model = Model().cuda()
        model.load_state_dict(torch.load("models/five/five_model.pt"))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adamax(model.parameters(), lr=self.lr)

        total_target_loss, total_source_loss, total_both_loss = [], [], []
        for epoch in range(1, self.epochs + 1):

            epoch_target_loss, epoch_source_loss, both_loss = [], [], []
            for image, label in tqdm(self.train_set, ncols=90, desc=("epoch " + str(epoch))):
                optimizer.zero_grad()

                image = image.float().cuda()
                prediction = model.train()(image, label=label[0], visualize=False)

                loss = criterion(prediction, image)
                loss.backward()
                optimizer.step()

                if label[0] == "0": epoch_target_loss.append(loss.item())
                elif label[0] == "1": epoch_source_loss.append(loss.item())
                both_loss.append(loss.item())

            if epoch % 1 == 0:
                torch.save(model.state_dict(), "models/five/five_model.pt")

                print(colored("target_loss", "cyan", attrs=["bold"]), "        ", \
                        colored("source_loss", "cyan", attrs=["bold"]), "       ", \
                        colored("both_loss", "cyan", attrs=["bold"]))

                print(np.mean(epoch_target_loss), "-", np.mean(epoch_source_loss), "-", np.mean(both_loss), "\n")
                total_target_loss.append(np.mean(epoch_target_loss))
                total_source_loss.append(np.mean(epoch_source_loss))
                total_both_loss.append(np.mean(both_loss))
                
        print("\n finished training")

        plt.plot(range(len(total_target_loss)), total_target_loss, c="blue")
        plt.plot(range(len(total_source_loss)), total_source_loss, c="red")
        plt.plot(range(len(total_both_loss)), total_both_loss, c="yellow")
        plt.show()


if __name__ == "__main__":
    runModel = RunModel(
               dataset_path="datasets/images/",
               batch_size=1,
               epochs=50,
               lr=0.0001)

    #runModel.train()

    """ test """

    # load trained model
    model = Model().cuda()
    model.load_state_dict(torch.load("models/five/five_model.pt"))

    # reproduce target image
    image_0 = Image.open("datasets/images/0_1.jpg")
    image_0 = np.array(image_0)

    image0 = torch.Tensor((image_0 / 255)).reshape(1, 3, 128, 128).cuda()
    prediction0 = model.eval()(image0, label="0", visualize=False)

    produced0 = prediction0[0].cpu().detach().numpy().reshape(128, 128, 3)
    produced0 *= 255
    produced0 = np.array(produced0, dtype="int")

    # reproduce source image
    image_1 = Image.open("datasets/images/1_1.jpg")
    image_1 = np.array(image_1)

    image1 = torch.Tensor((image_1 / 255)).reshape(1, 3, 128, 128).cuda()
    prediction1 = model.eval()(image1, label="1", visualize=False)

    produced1 = prediction1[0].cpu().detach().numpy().reshape(128, 128, 3)
    produced1 *= 255
    produced1 = np.array(produced1, dtype="int")

    # prduce fake image
    faked = model.eval()(image1, label="0", visualize=False)[0].cpu().detach().numpy().reshape(128, 128, 3)
    faked *= 255
    faked = np.array(faked, dtype="int")

    # plot
    fig, axs = plt.subplots(5)
    axs[0].imshow(image_0)
    axs[1].imshow(produced0)
    axs[2].imshow(image_1)
    axs[3].imshow(produced1)
    axs[4].imshow(faked)
    plt.show()