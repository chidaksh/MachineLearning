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


class LeNetMix(nn.Module):

    def __init__(self):
        super(LeNetMix, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),  # (N, 3, 32, 32) -> (N,  6, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 28, 28) -> (N,  6, 14, 14)
            nn.Conv2d(6, 16, 5),  # (N, 6, 14, 14) -> (N, 16, 10, 10)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.AvgPool2d(2, stride=2)  # (N,16, 10, 10) -> (N, 16, 5, 5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),  # (N, 400) -> (N, 120)
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(120, 84),  # (N, 120) -> (N, 84)
            nn.ReLU(),
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
                                        nonlinearity='relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        if isinstance(module, nn.Linear):
            module.weight = torch.nn.parameter.Parameter(
                nn.init.kaiming_normal_(module.weight,
                                        mode="fan_in",
                                        nonlinearity='relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)


class SPSAModel(LeNetMix):

    def __init__(self):
        super().__init__()
        self.total_param = 0

    def stop_backprop(self):
        for param in self.parameters():
            param.requires_grad = False
        return
    
    def flatten(self,lst):
        flattened = []
        for item in lst:
            if isinstance(item, list):
                flattened += self.flatten(item)
            else:
                flattened.append(item)
        return flattened

    def spsa_update(self, loss_fn, inputs, lables, a, A, c, alpha, gamma,
                    num_samples):

        # old_params = []
        initloss = loss_fn(self.forward(inputs), lables)
        # updated_params = []
        # for param in self.parameters():
        #     # self.total_param += param.numel()
        #     updated_params.append(torch.zeros_like(param))
        #     print(param)
        # print("-------------------------------------------------------------------------------------------------------")
        #     old_params.append(param)
        # print(updated_params)
        # updated_params = torch.Tensor(updated_params)
        # print(updated_params.shape, self.total_param)
        updated_params = []
        for param in self.parameters():
            updated_params.append(torch.zeros_like(param))
                
        for k in range(1,num_samples+1):
            
            # entireUpdate = []
            # updated_params = []
            # for param in self.parameters():
            #     # self.total_param += param.numel()
            #     updated_params.append(torch.zeros_like(param))
            lr = a / ((k + A)**alpha)
            curr_c = c / (k**gamma)
            
            for param, update in zip(self.parameters(), updated_params):
                update.data.zero_()
                delta = torch.sign(torch.rand_like(update) - 0.5)
                update.data.add_(curr_c * delta)
                # print(update)
                # print("-------------------------------------------------------------------------------------------------------")
                param.data.add_(update)
                # entireUpdate.extend(self.flatten(update.tolist()))
            # print("-------------------------------------------------------------------------------------------------------")
            # for param in self.parameters():
            #     print(param)
            # print("-------------------------------------------------------------------------------------------------------")
            # entireUpdate = torch.Tensor(entireUpdate).to(device)

            loss_incrpert = loss_fn(self.forward(inputs), lables)

            for param, update in zip(self.parameters(), updated_params):
                # print(-1 * update)
                # print("-------------------------------------------------------------------------------------------------------")
                param.data.add_(-2 * update)
                
            # print("-------------------------------------------------------------------------------------------------------")
            # for param in self.parameters():
            #     print(param)
            # print("-------------------------------------------------------------------------------------------------------")
            loss_decrpert = loss_fn(self.forward(inputs), lables)
            temp = (loss_incrpert - loss_decrpert)
            
            for param, update in zip(self.parameters(), updated_params):
                # print(-1 * update)
                # print("-------------------------------------------------------------------------------------------------------")
                param.data.add_(update)
                
            for param,update in zip(self.parameters(), updated_params):
                grad = (0.5 * temp) / update
                param.data.add_(-1 * lr * grad)
                
            # print("-------------------------------------------------------------------------------------------------------")
            for param in self.parameters():
                print(param)
            print("-------------------------------------------------------------------------------------------------------")
            
        return initloss


def TrainModel(epochs, show_plots=True):
    MixLeNet.stop_backprop()
    loss_arr = []
    loss_epoch_arr = []
    train_acc_arr = []
    valid_acc_arr = []
    max_epochs = epochs
    bestvalAcc = 0
    bestepoch = 0
    for epoch in trange(max_epochs):

        MixLeNet.train()
        if epoch == 0:
            print(
                'Epoch: %d/%d Test acc: %0.2f, Valid acc: %0.2f, Train acc: %0.2f'
                % (epoch, max_epochs, evaluation(
                    MixLeNet, testloader), evaluation(MixLeNet, validloader),
                   evaluation(MixLeNet, trainloader)))

        start = len(loss_arr)
        for i, data in enumerate(trainloader, 0):
                
            inputs, lables = data
            inputs, lables = inputs.to(device), lables.to(device)

            # opt.zero_grad()

            # outputs = MixLeNet(inputs)
            # loss = loss_fn(outputs, lables)
            # loss.backward()
            # opt.step()
            loss = MixLeNet.spsa_update(loss_fn,
                                        inputs,
                                        lables,
                                        a=0.05,
                                        A=25,
                                        c=0.1,
                                        alpha=0.9,
                                        gamma=0.3,
                                        num_samples=3)
            
            # if i == 0 and epoch == 0:
            #     print(f'Initial Loss is {loss.item()}')
            
            # loss = MixLeNet.spsa_update(loss_fn,
            #                             inputs,
            #                             lables,
            #                             a=0.05,
            #                             A=25,
            #                             c=0.1,
            #                             alpha=0.9,
            #                             gamma=0.3,
            #                             num_samples=1)
            if torch.isnan(loss):
                break
            print(f'Initial Loss is {loss.item()}')
            
            loss_arr.append(loss.item())
        
        end = len(loss_arr)
        # loss_epoch_arr.append(mean(loss_arr[start:end + 1]))

        with torch.no_grad():
            MixLeNet.eval()
            trainacc = evaluation(MixLeNet, trainloader)
            validacc, validloss = evaluation(MixLeNet, validloader, valid=True)
            if validacc > bestvalAcc:
                bestvalAcc = validacc
                bestepoch = epoch
                torch.save(MixLeNet.state_dict(),
                           "./checkpoints/SuperLeNetSpsaBest.pt")
            # lr_scheduler.step(validloss)
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
        plt.savefig('./images/SuperLeNetSpsa_Loss.jpg', bbox_inches="tight")
        plt.clf()

        plt.ylabel("loss")
        plt.title('Loss across all batches')
        plt.plot(loss_arr)
        plt.savefig('./images/SuperLeNetSpsa_TLoss.jpg', bbox_inches="tight")
        plt.clf()

        plt.plot(train_acc_arr, label="Training Accuracy")
        plt.plot(valid_acc_arr, label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.title('Accuracy Vs Epochs')
        plt.savefig('./images/SuperLeNetSpsa_Accuracy.jpg',
                    bbox_inches="tight")
        plt.clf()

    torch.save(MixLeNet.state_dict(), "./checkpoints/SuperLeNetSpsaLast.pt")
    MixLeNet.load_state_dict(torch.load("./checkpoints/SuperLeNetSpsaBest.pt"))
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
MixLeNet = SPSAModel().to(device)
# PrintModelSummary(MixLeNet)
# opt = optim.Adam(MixLeNet.parameters(), lr=5*1e-5, weight_decay=1e-3)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     opt, mode='min', factor=0.5, patience=4)
TrainModel(1, show_plots=True)
end = time.time()
print(f'Total Time taken to train LeNet is {end-start} seconds')
