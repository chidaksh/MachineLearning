import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange
from cifar.pipeline import *
from PIL import Image
from statistics import mean

# Original images come in shapes of [3,64,64]
DATA_DIR = './data/tiny-imagenet-200'
TRAIN_DIR = DATA_DIR + '/train'
VALID_DIR = DATA_DIR + '/val'
TEST_DIR = DATA_DIR + '/test'


class ImageNet(Dataset):
    def __init__(self, root_dir, size, transform=None):
        super(ImageNet, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.testsize = size  # image transform
        self.imageNames = []
        for i in range(self.testsize):
            string = self.root_dir + 'test_' + str(i) + '.JPEG'
            self.imageNames.append(string)
        self.images = []
        for i in range(self.testsize):
            # print(self.imageNames[i])
            temp_image = Image.open(self.imageNames[i])
            temp_image = temp_image.convert('RGB')
            self.images.append(self.transform(temp_image))

    def __len__(self):
        return len(self.imageNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images[idx]


def dataloader(data, name, transform, batch_sz=128):
    if data is None:
        print("Data argument is missing")
        return
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_sz,
                            shuffle=(name == "train"),num_workers=4)

    return dataloader


def TestDataset():
    return ImageNet(TEST_DIR + '/images/', 1000, transform=validtransform)


def preprocess_validation():
    valid_images_dir = VALID_DIR + '/images'
    if os.path.exists(valid_images_dir):
        print('entered')
        fp = open(VALID_DIR + '/val_annotations.txt', 'r')
        data = fp.readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

        for img, folder in val_img_dict.items():
            newpath = VALID_DIR + '/' + folder
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(valid_images_dir + '/', img)):
                os.rename(os.path.join(valid_images_dir + '/', img),
                          os.path.join(newpath + '/', img))

    return


def generate_dataloaders():
    trainloader = dataloader(TRAIN_DIR, "train", traintransform)
    validloader = dataloader(VALID_DIR, 'val', validtransform)
    testloader = DataLoader(TestDataset(), 64)
    print(len(trainloader), len(validloader), len(testloader))
    return trainloader, validloader, testloader


traintransform = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),  # Resize images to 256 x 256
    transforms.CenterCrop(224),  # Center crop image
    transforms.ColorJitter(brightness=0.1, contrast=0.1,
                           saturation=0.1, hue=0.1),
    transforms.ToTensor(),  # Converting cropped images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
validtransform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256 x 256
    transforms.CenterCrop(224),  # Center crop image
    transforms.ToTensor(),  # Converting cropped images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Residual(nn.Module):
    def __init__(self, num_blocks, input_channels, output_channels, stride=1, p=0.5):
        super(Residual, self).__init__()
        self.ResidualFirstBlock = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(p),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(p)
        )
        self.ResidualOtherBlock = nn.Sequential(
            nn.Conv2d(output_channels, output_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(p),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(p)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(p)
        )
        self.isdownsample = False
        if input_channels != output_channels:
            self.isdownsample = True
        self.relu = nn.ReLU(inplace=True)
        self.numBlocks = num_blocks

    def forward(self, x):
        residual = x.clone()
        count = 0
        if count == 0:
            x = self.ResidualFirstBlock(x)
            if self.isdownsample:
                residual = self.downsample(residual)
                x += residual
            else:
                x += residual
            x = self.relu(x)
            count += 1
        else:
            while (count != self.numBlocks):
                x = self.ResidualOtherBlock(x)
                x += residual
                x = self.relu(x)
                count += 1
        return x


class NonResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, p=0.2):
        super(NonResidual, self).__init__()
        self.NonResidualBlock = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=stride,
                      padding=3, bias=False),  # (N,3,224,224) ---> (N,64,112,112)
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(p),
            nn.ReLU(inplace=True),
            # (N,64,112,112) ----> (N,64,56,56)
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        )

    def forward(self, x):
        x = self.NonResidualBlock(x)
        return x


class ResNet(nn.Module):
    def __init__(self, Residual, NonResidual):
        super(ResNet, self).__init__()
        self.nonResidualLayer = self.createLayers(
            NonResidual, 1, 3, 64, 2, False)  # Output ---> (N,64,56,56)
        self.residualLayer1 = self.createLayers(Residual, 3, 64, 64, 1)
        self.residualLayer2 = self.createLayers(Residual, 4, 64, 128, 2)
        self.residualLayer3 = self.createLayers(Residual, 6, 128, 256, 2)
        self.residualLayer4 = self.createLayers(
            Residual, 3, 256, 512, 2)  # Output ---> 7 x 7 x 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 200)
        )

    def createLayers(self, name, num_layers, in_sz, out_sz, stride, isResidual=True):
        layers = []
        if isResidual == False:
            layers.append(name(in_sz, out_sz, stride))
        else:
            layers.append(name(num_layers, in_sz, out_sz, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.nonResidualLayer(x)
        x = self.residualLayer1(x)
        x = self.residualLayer2(x)
        x = self.residualLayer3(x)
        x = self.residualLayer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def name(self):
        return "ResNetDp"


def TrainResNet(model, epochs, trainloader, validloader, testloader, show_plots=False):
    loss_arr = []
    loss_epoch_arr = []
    train_acc_arr = []
    valid_acc_arr = []
    bestvalAcc = 0
    bestepoch = 0
    best = "./checkpoints/" + model.name() + "Best.pt"
    last = "./checkpoints/" + model.name() + "Last.pt"
    # print(f"total batches are {len(trainloader)}")
    for epoch in trange(epochs):
        start = len(loss_arr)
        # print(f"started epoch {epoch}")
        for i, datam in enumerate(trainloader, 0):
            # print(f'Started batch {i}')
            inputs, labels = datam
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            loss_arr.append(loss.item())
            # print(f'Batch {i} is done')
        end = len(loss_arr)
        loss_epoch_arr.append(mean(loss_arr[start:end+1]))

        with torch.no_grad():
            trainacc = evaluation(model, trainloader)
            validacc,validloss = evaluation(model, validloader,valid=True)
            if validacc > bestvalAcc:
                bestvalAcc = validacc
                bestepoch = epoch
                torch.save(model.state_dict(), best)
                
            lr_scheduler.step(validloss)
            train_acc_arr.append(trainacc)
            valid_acc_arr.append(validacc)
            print('Epoch: %d/%d, Train acc: %0.2f, Valid acc: %0.2f' %
                  (epoch, epochs, trainacc, validacc))  

        if epoch % 25 == 24:
            torch.save(model.state_dict(), last)

        if epoch - bestepoch >= 10:
            print("Early stopping")
            break

    if show_plots:
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title('Loss Vs Epochs')
        plt.plot(loss_epoch_arr)
        plt.savefig(f'./images/{model.name()}_Loss.jpg', bbox_inches="tight")
        plt.clf()

        plt.plot(train_acc_arr, label="Training Accuracy")
        plt.plot(valid_acc_arr, label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.title('Accuracy Vs Epochs')
        plt.savefig(
            f'./images/{model.name()}_Accuracy.jpg', bbox_inches="tight")
        plt.clf()

    torch.save(model.state_dict(), last)
    print('Valid acc: %0.2f, Train acc: %0.2f' %
          (evaluation(model, validloader), evaluation(model, trainloader)))
    return


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on Device {device}")
start = time.time()
loss_fn = nn.CrossEntropyLoss()
preprocess_validation()
trainloader, validloader, testloader = generate_dataloaders()
# trainloader, validloader, testloader = LoadDataloaders()
resnet = ResNet(Residual, NonResidual).to(device)
PrintModelSummary(resnet, (3, 224, 224))
opt = optim.Adam(resnet.parameters(), lr=1e-5, weight_decay=1e-2)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=4)
# TrainModel(200, show_plots=True)
print("Training Started")
TrainResNet(resnet, 100, trainloader, validloader, testloader, show_plots=True)
end = time.time()
print(f'Total Time taken to train LeNet is {end-start} seconds')
