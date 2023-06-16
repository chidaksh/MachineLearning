import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.notebook import tqdm
from torchsummary import summary


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.clf()
    return


def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment=False,
                           random_seed=4,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1, name=""):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    valid_transform = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256 x 256
        transforms.CenterCrop(224),  # Center crop image
        transforms.ToTensor(),
        normalize  # Converting cropped images to tensors
    ])
    if name == "resnet":
        if augment:
            # print("train transform")
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256),  # Resize images to 256 x 256
                transforms.CenterCrop(224),  # Center crop image
                transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                       saturation=0.1, hue=0.1),
                transforms.ToTensor(),  # Converting cropped images to tensors
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),  # Resize images to 256 x 256
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    # load the dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    # train_dataset = datasets.Image

    valid_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if name == "resnet":
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=4,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=4,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers,
        )
    # visualize some images
    if show_sample:
        sample_bz = 6
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=sample_bz,
            num_workers=num_workers, shuffle=False,
        )
        show_batch(sample_loader)
        # print(' '.join(classes[labels[j]] for j in range(sample_bz)))

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=False,
                    num_workers=1, name=""):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256 x 256
        transforms.CenterCrop(224),  # Center crop image
        transforms.ToTensor(),
        normalize,
    ])

    if name == "resnet":

        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=4,
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
        )

    return data_loader


def LoadDataloaders(trainbz=64, testbz=128, name=""):
    train_bz = trainbz
    if name == "resnet":
        # print("load ", name)
        (trainloader, validloader) = get_train_valid_loader(
            '../data', train_bz, augment=True, show_sample=False, name=name)
    else:
        (trainloader, validloader) = get_train_valid_loader(
            '../data', train_bz, show_sample=False, name=name)
    test_bz = testbz
    testloader = get_test_loader('../data', test_bz, name=name)
    # print(len(trainloader), len(validloader), len(testloader))
    return trainloader, validloader, testloader


def SaveModelCheckpoint(model, PATH):
    torch.save(model.state_dict(), PATH)
    return


def LoadModelCheckpoint(model, PATH):
    model = torch.load(PATH)
    return model


def PrintModelSummary(model, input_size=(3, 32, 32)):
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
