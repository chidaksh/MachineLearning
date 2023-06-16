import torch
import numpy as np
import torchvision.transforms as transforms
import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchsummary import summary
import torch.nn as nn

# Original images come in shapes of [3,64,64]
DATA_DIR = '../data/tiny-imagenet-200'
TRAIN_DIR = DATA_DIR + '/train'
VALID_DIR = DATA_DIR + '/val'
TEST_DIR = DATA_DIR + '/test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


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
                            shuffle=(name == "train"), num_workers=4)

    return dataloader


def TestDataset(name=""):
    if name == "resnet":
        validtransform = transforms.Compose([
            transforms.Resize(256),  # Resize images to 256 x 256
            transforms.CenterCrop(224),  # Center crop image
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
    else:
        validtransform = transforms.Compose([
            transforms.Resize(32),  # Resize images to 32 x 32
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
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


def generate_dataloaders(name=""):
    if (name == "resnet"):
        traintransform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),  # Resize images to 256 x 256
            transforms.CenterCrop(224),  # Center crop image
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.1),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

        validtransform = transforms.Compose([
            transforms.Resize(256),  # Resize images to 256 x 256
            transforms.CenterCrop(224),  # Center crop image
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
    else:
        traintransform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),  # Resize images to 32 x 32
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.1),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

        validtransform = transforms.Compose([
            transforms.Resize(32),  # Resize images to 32 x 32
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
    trainloader = dataloader(TRAIN_DIR, "train", traintransform)
    validloader = dataloader(VALID_DIR, 'val', validtransform)
    testloader = DataLoader(TestDataset(name=name), 128, num_workers=4)
    print(len(trainloader), len(validloader), len(testloader))
    return trainloader, validloader, testloader


def PrintModelSummary(model, input_size=(3, 224, 224)):
    print(summary(model, input_size=input_size))  # (C,H,W)
    return


def evaluation(model, dataloader, valid=False):
    total, correct = 0, 0
    runningLoss = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        if valid:
            ls = criterion(outputs, labels)
            runningLoss += ls.item()
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    if valid:
        return 100 * correct/total, runningLoss/len(dataloader)
    else:
        return 100 * correct / total
