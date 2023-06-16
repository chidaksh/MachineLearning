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
from pipeline import *


class LeNetDrop(nn.Module):
    def __init__(self):
        super(LeNetDrop, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(3, 6, 5),         # (N, 3, 32, 32) -> (N,  6, 28, 28)
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 28, 28) -> (N,  6, 14, 14)
            nn.Conv2d(6, 16, 5),        # (N, 6, 14, 14) -> (N, 16, 10, 10)
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.AvgPool2d(2, stride=2)   # (N,16, 10, 10) -> (N, 16, 5, 5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),         # (N, 400) -> (N, 120)
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),          # (N, 120) -> (N, 84)
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(84, 10)            # (N, 84)  -> (N, 10)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


def TrainModel(epochs, show_plots=True):
    loss_arr = []
    loss_epoch_arr = []
    train_acc_arr = []
    valid_acc_arr = []
    max_epochs = epochs
    bestvalAcc = 0
    bestepoch = 0
    for epoch in trange(max_epochs):

        start = len(loss_arr)
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = DropLeNet(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            loss_arr.append(loss.item())

        end = len(loss_arr)
        loss_epoch_arr.append(mean(loss_arr[start:end+1]))

        with torch.no_grad():
            trainacc = evaluation(DropLeNet, trainloader)
            validacc = evaluation(DropLeNet, validloader)
            if validacc > bestvalAcc:
                bestvalAcc = validacc
                bestepoch = epoch
                torch.save(DropLeNet.state_dict(), "../checkpoints/LeNetDpBest.pt")
            train_acc_arr.append(trainacc)
            valid_acc_arr.append(validacc)
            print('Epoch: %d/%d, Train acc: %0.2f, Valid acc: %0.2f' %
                  (epoch, max_epochs, trainacc, validacc))

        if epoch % 25 == 24:
            torch.save(DropLeNet.state_dict(), "../checkpoints/LeNetDpLast.pt")

    if show_plots:
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title('Loss Vs Epochs')
        plt.plot(loss_epoch_arr)
        plt.savefig('../images/LeNetDp_Loss.jpg', bbox_inches="tight")
        plt.clf()

        plt.plot(train_acc_arr, label="Training Accuracy")
        plt.plot(valid_acc_arr, label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.title('Accuracy Vs Epochs')
        plt.savefig('../images/LeNetDp_Accuracy.jpg', bbox_inches="tight")
        plt.clf()

    torch.save(DropLeNet.state_dict(), "../checkpoints/LeNetDpLast.pt")
    DropLeNet.load_state_dict(torch.load("../checkpoints/LeNetDpBest.pt"))
    print('Test acc: %0.2f, Valid acc: %0.2f, Train acc: %0.2f' % (evaluation(DropLeNet,
          testloader), evaluation(DropLeNet, validloader), evaluation(DropLeNet, trainloader)))
    return


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on Device {device}")
start = time.time()
loss_fn = nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
trainloader, validloader, testloader = LoadDataloaders()
DropLeNet = LeNetDrop().to(device)
PrintModelSummary(DropLeNet)
opt = optim.Adam(DropLeNet.parameters(), lr=5*1e-5)
TrainModel(200, show_plots=True)
end = time.time()
print(f'Total Time taken to train DropLeNet is {end-start} seconds')
