import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
from preprocess import *


class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.actv = nn.Tanh()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        # shape: B x S x Feature   since batch = True
        embeds = self.embedding(x)
        # print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.actv(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

    

def TrainingLoop(show_plots=False):
    clip = 5
    epochs = 5
    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []
    count = 0
    for epoch in trange(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        # initialize hidden state
        h = model.init_hidden(BatchSize)
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = evaluation(output, labels)
            train_acc += accuracy
            if epoch == 0 and i == 0:
                print(f'Epoch {epoch}')

                print(
                    f'Intial train_loss : {loss.item()}')
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        val_h = model.init_hidden(BatchSize)
        val_losses = []
        val_acc = 0.0
        model.eval()
        for i, data in enumerate(valid_loader):

            inputs, labels = data
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

            accuracy = evaluation(output, labels)
            val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc/len(train_loader.dataset)
        epoch_val_acc = val_acc/len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(
            f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), './state_dict.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, epoch_val_loss))
            valid_loss_min = epoch_val_loss
        else:
            count += 1
        print(25*'==')

        if count >= 3:
            break

    if show_plots:
        plt.plot(epoch_tr_acc, label='Train Acc')
        plt.plot(epoch_vl_acc, label='Validation Acc')
        plt.title("Accuracy")
        plt.legend()
        plt.savefig('WordLstmAcc.jpg')
        plt.clf()

        plt.plot(epoch_tr_loss, label='Train loss')
        plt.plot(epoch_vl_loss, label='Validation loss')
        plt.title("Loss")
        plt.legend()
        plt.savefig('WordLstmLoss.jpg')


device = setup_device()
no_layers = 2
vocab_size = len(vocab) + 1  # extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256
model = SentimentRNN(no_layers, vocab_size, hidden_dim,
                     embedding_dim, drop_prob=0.5)
# moving to gpu
model.to(device)
print(model)
# loss and optimization functions
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
TrainingLoop(show_plots=True)
# function to predict accuracy
