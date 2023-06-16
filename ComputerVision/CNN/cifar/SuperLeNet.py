import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.notebook import tqdm
from tqdm.auto import trange
from statistics import mean
from cifar.pipeline import *


class LeNetMix(nn.Module):

    def __init__(self):
        super(LeNetMix, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),  # (N, 3, 32, 32) -> (N,  6, 28, 28)
            nn.LeakyReLU(),
            nn.BatchNorm2d(6),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 28, 28) -> (N,  6, 14, 14)
            nn.Conv2d(6, 16, 5),  # (N, 6, 14, 14) -> (N, 16, 10, 10)
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),
            nn.AvgPool2d(2, stride=2)  # (N,16, 10, 10) -> (N, 16, 5, 5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),  # (N, 400) -> (N, 120)
            nn.LeakyReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(120, 84),  # (N, 120) -> (N, 84)
            nn.LeakyReLU(),
            nn.BatchNorm1d(84),
            nn.Linear(84, 10)  # (N, 84)  -> (N, 10)
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight = torch.nn.parameter.Parameter(
                nn.init.kaiming_normal_(module.weight,
                                        mode="fan_in",
                                        nonlinearity='relu')*0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        if isinstance(module, nn.Linear):
            module.weight = torch.nn.parameter.Parameter(
                nn.init.kaiming_normal_(module.weight,
                                        mode="fan_in",
                                        nonlinearity='relu')*0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)


def TrainModel(epochs, show_plots=True):
    loss_arr = []
    loss_epoch_arr = []
    train_acc_arr = []
    valid_acc_arr = []
    max_epochs = epochs
    bestvalAcc = 0
    bestepoch = 0
    for epoch in trange(max_epochs):

        if epoch == 0:
            print(
                'Epoch: %d/%d Test acc: %0.2f, Valid acc: %0.2f, Train acc: %0.2f'
                % (epoch, max_epochs, evaluation(MixLeNet,
                                                 testloader), evaluation(MixLeNet, validloader),
                   evaluation(MixLeNet, trainloader)))

        start = len(loss_arr)
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = MixLeNet(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            loss_arr.append(loss.item())

            if i == 0 and epoch == 0:
                print(f'Initial Loss is {loss.item()}')

        end = len(loss_arr)
        loss_epoch_arr.append(mean(loss_arr[start:end + 1]))

        with torch.no_grad():
            trainacc = evaluation(MixLeNet, trainloader)
            validacc, validloss = evaluation(MixLeNet, validloader, valid=True)
            if validacc > bestvalAcc:
                bestvalAcc = validacc
                bestepoch = epoch
                torch.save(MixLeNet.state_dict(),
                           "../checkpoints/SuperLeNetBest.pt")
            lr_scheduler.step(validloss)
            train_acc_arr.append(trainacc)
            valid_acc_arr.append(validacc)
            print('Epoch: %d/%d, Train acc: %0.2f, Valid acc: %0.2f' %
                  (epoch, max_epochs, trainacc, validacc))

        if epoch - bestepoch >= 15:
            print("Early stopping")
            break

    if show_plots:
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title('Loss Vs Epochs')
        plt.plot(loss_epoch_arr)
        plt.savefig('../images/SuperLeNet_Loss.jpg', bbox_inches="tight")
        plt.clf()

        plt.ylabel("loss")
        plt.title('Loss across all batches')
        plt.plot(loss_arr)
        plt.savefig('../images/SuperLeNet_TLoss.jpg', bbox_inches="tight")
        plt.clf()

        plt.plot(train_acc_arr, label="Training Accuracy")
        plt.plot(valid_acc_arr, label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.title('Accuracy Vs Epochs')
        plt.savefig('../images/SuperLeNet_Accuracy.jpg', bbox_inches="tight")
        plt.clf()

    torch.save(MixLeNet.state_dict(), "../checkpoints/SuperLeNetLast.pt")
    MixLeNet.load_state_dict(torch.load("../checkpoints/SuperLeNetBest.pt"))
    print(
        'Final Accuracies are, Test acc: %0.2f, Valid acc: %0.2f, Train acc: %0.2f'
        % (evaluation(MixLeNet, testloader), evaluation(
            MixLeNet, validloader), evaluation(MixLeNet, trainloader)))
    return


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on Device {device}")
start = time.time()
loss_fn = nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')
trainloader, validloader, testloader = LoadDataloaders()
MixLeNet = LeNetMix().to(device)
PrintModelSummary(MixLeNet)
opt = optim.Adam(MixLeNet.parameters(), lr=5*1e-4, weight_decay=0.1)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=4)
TrainModel(200, show_plots=True)
end = time.time()
print(f'Total Time taken to train LeNet is {end-start} seconds')
