import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def setup_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device


def load_dataset():
    base_csv = './imdbDataset.csv'
    df = pd.read_csv(base_csv)
    print(df.head())

    X, y = df['review'].values, df['sentiment'].values
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2)
    print(f'shape of train data is {x_train.shape}')
    print(f'shape of test data is {x_test.shape}')

    dd = pd.Series(y_train).value_counts()
    sns.barplot(x=np.array(['negative', 'positive']), y=dd.values)
    plt.ylabel("Number of data points")
    plt.savefig("IMDBdataset.jpg")
    plt.clf()
    
    return (x_train, x_test, y_train, y_test)


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s


def tockenize(x_train, y_train, x_val, y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i+1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]

    return np.array(final_list_train, dtype=object), np.array(encoded_train), np.array(final_list_test, dtype=object), np.array(encoded_test), onehot_dict


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def add_padding():
    x_train_pad = padding_(x_train, 500)
    x_test_pad = padding_(x_test, 500)
    return x_train_pad, x_test_pad


def dataloaders(x_train_pad, x_test_pad, bz=50):
    global train_loader, valid_loader
    train_data = TensorDataset(torch.from_numpy(
        x_train_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(
        x_test_pad), torch.from_numpy(y_test))
    batch_size = bz

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


def predict_text(model, device, text):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
                         if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h)
    return (output.item())


def accuracy(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


def evaluation(model,device,h, dataloader):
    model.eval()
    acc = 0.0
    Loss = []
    for i, data in enumerate(dataloader):

        inputs, labels = data
        h = tuple([each.data for each in h])

        inputs, labels = inputs.to(device), labels.to(device)

        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        acc += accuracy(output, labels)
        Loss.append(loss.item())
        
    acc = acc/len(dataloader.dataset)
    return (Loss,acc)


BatchSize = 50
(x_train, x_test, y_train, y_test) = load_dataset()
x_train, y_train, x_test, y_test, vocab = tockenize(
    x_train, y_train, x_test, y_test)
print(f'Length of vocabulary is {len(vocab)}')
x_train_pad, x_test_pad = add_padding()
dataloaders(x_train_pad, x_test_pad, BatchSize)
criterion = nn.BCELoss()
