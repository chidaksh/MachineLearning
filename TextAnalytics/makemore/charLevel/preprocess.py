import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
random.seed(42)


def readFromFile(file):
    words = open(file, 'r').read().splitlines()
    # print(len(words))
    maxlen = max(len(w) for w in words)
    # print(words[:8])
    random.shuffle(words)
    return words, maxlen
# print(words[:8])


def vocabulary(words):
    global stoi,itos
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    # stoi['E'] = 1
    # stoi['P'] = 2
    # print(stoi)
    itos = {i: s for s, i in stoi.items()}
    num_letters = len(itos)
    # print(itos)
    # print(vocab_size)
    return stoi, itos, num_letters


def makeDataset(words, maxlen, block_size=3):
    global stoi
    X, Y = [], []
    # wordlength = {}
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(''.join(itos[ix] for ix in context))
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    return X, Y


def name2onehot(name):
    global n_letters,stoi
    # print("name is: ",name)
    repr = torch.zeros(len(name), 1, n_letters)
    for index, letter in enumerate(name):
        pos = stoi[letter]
        repr[index][0][pos] = 1
    return repr

def label2torch(label):
    return torch.tensor([label])

def datapoint(npoints, X_, y_):
    to_ret = []
    for i in range(npoints):
        index_ = torch.randint(len(X_), (1, 1))
        name, lang = X_ [index_], y_[index_]
        to_ret.append((name, lang, name2onehot(name), label2torch(lang)))
    return to_ret

def infer(net, name):
    net.eval()
    input = name2onehot(name)
    hidden = None
    for i in range(input.size()[0]):
        output, hidden = net(input[i], hidden)

    return output


def evaluation(net, n_points, k, X_, y_):

    data_ = datapoint(n_points, X_, y_)
    correct = 0

    for name, language, name_ohe, lang_rep in data_:

        output = infer(net, name)
        val, indices = output.topk(k)

        if lang_rep in indices:
            correct += 1

    accuracy = correct/n_points
    return accuracy

def  sample(model):
    for _ in range(20):
        out = []
        context = [0] * block_sz
        # initialize with all ...
        while True:
            # forward pass the neural net
            input = ''.join(itos[ix] for ix in context)
            logits = model(name2onehot(input),None)
            probs = nn.Softmax(logits,dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            # shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # if we sample the special '.' token, break
            if ix == 0:
                break
            
        print(''.join(itos[i] for i in out)) # decode and print the generated word

words, maxlen = readFromFile('names.txt')
# print(maxlen)
stoi, itos, n_letters = vocabulary(words)
block_sz = 8
X, Y = makeDataset(words, maxlen,block_sz)
# print(X[3],Y[3])
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X, Y, test_size=0.2, random_state=0, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_dev, Y_dev, test_size=0.5, random_state=0, shuffle=False)
print(len(X_train), len(X_val), len(X_test))
# for x, y in zip(X_train[:20], Y_train[:20]):
#     print(''.join(ix for ix in x), '-->', itos[y])
#     break
