from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import trange
from io import open
import glob
import os
import unicodedata
import string
import time

class Dataset():
    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters) + 1 # Plus EOS marker
        self.all_lines = []

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self,filename):
        with open(filename, encoding='utf-8') as some_file:
            return [self.unicodeToAscii(line.strip()) for line in some_file]
        
    def readFromNames(self,filename):
        lines = self.readLines(filename)
        self.all_lines.extend(lines)
    
    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self,line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor.to(device)

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self,line):
        letter_indexes = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(self.n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes).to(device)

    def TrainingExample(self,idx):
        line = self.all_lines[idx]
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)
        input_line_tensor = input_line_tensor.to(device)
        target_line_tensor = target_line_tensor.to(device)
        return input_line_tensor, target_line_tensor

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def train(self, input_line_tensor, target_line_tensor, opt):
        target_line_tensor.unsqueeze_(-1)
        hidden = self.initHidden().to(device)
        opt.zero_grad()
        loss = 0

        for i in range(input_line_tensor.size(0)):
            output, hidden = self(input_line_tensor[i], hidden)
            l = self.criterion(output, target_line_tensor[i])
            loss += l

        loss.backward(retain_graph=True)
        opt.step()
        return output, loss.item() / input_line_tensor.size(0)

    def train_setup(self,data,epochs,lr):
        start = time.time()
        self.criterion = nn.NLLLoss()
        opt = optim.Adam(self.parameters(), lr = lr, weight_decay = 0.001)
        num_epochs = epochs
        self.all_losses = []
        total_loss = 0
        for i in trange(num_epochs):
            for j in trange(len(data.all_lines)):
                output,loss = self.train(*data.TrainingExample(j),opt)
                total_loss += loss
            print(f"Loss after epoch {i} is {total_loss/len(data.all_lines)}")
            self.all_losses.append(total_loss/len(data.all_lines))
            total_loss = 0

    def sample(self,data,start_letter="S",max_length = 20):
        with torch.no_grad():  # no need to track history in sampling
            input = data.inputTensor(start_letter)
            hidden = self.initHidden().to(device)

            output_name = start_letter

            for i in range(max_length):
                output, hidden = self(input[0], hidden)
                
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == data.n_letters - 1:
                    break
                else:
                    letter = data.all_letters[topi]
                    output_name += letter
                input = data.inputTensor(letter)
            return output_name
        
    def generate_samples(self,data,start_letters="ABC"):
        for letter in start_letters:
            print(self.sample(data,letter))
    
    def showplots(self,):
        plt.figure()
        plt.plot(self.all_losses)
        plt.xlabel("epochs")
        plt.ylabel("Loss Value")
        plt.savefig("LossPlot.jpg")
        plt.clf()
        
n_hidden = 128
dataset = Dataset()
dataset.readFromNames("Names.txt")
net = RNNModel(dataset.n_letters, n_hidden, dataset.n_letters)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(net)
net.to(device)
print(f"Running on Device {device}")
net.train_setup(dataset,epochs = 30,lr = 0.001)
net.showplots()
net.generate_samples(dataset,start_letters='SAC')
