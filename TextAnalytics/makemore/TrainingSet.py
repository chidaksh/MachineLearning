import torch
import torch.nn.functional as F
import math

words = open('names.txt','r').read().splitlines()
distinct_chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(distinct_chars)}
stoi['.'] = 0  
xs, ys = [], []
def createTrainingSet():
    global xs, ys
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for (ch1, ch2) in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return

def initializeweights():
    global W,xenc
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27,27), generator=g)/math.sqrt(27)
    W.requires_grad = True
    xenc = F.one_hot(xs,num_classes = 27).float() # default dtype would be int64 but we need floats for neural nets
    
def forwardpass():
    
    logits = xenc @ W # log counts
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdim=True)
    # print(prob.shape)
    # print(prob)
    # print(prob[0].sum())
    return prob

def loss(n,prob):
    temp = prob[torch.arange(n),ys]
    # print(f'Model Predicted Probabiliity of factual classes of first {n} entries is {temp}')
    nll = -temp.log().mean()
    print(f'Average NLL Loss is {nll}')
    return nll

def backpass(n,prob,lr):
    # print(W.shape)
    W.grad = None
    nllloss = loss(n,prob)
    nllloss.backward()
    # print(W.grad)
    W.data += -lr * W.grad
    return

def setpipeline(n,epochs,lr = 50):
    initializeweights()
    for i in range(epochs):
        prob = forwardpass()
        backpass(n,prob,lr)

createTrainingSet()
num = xs.nelement()
print(num)
setpipeline(num,100)
# print(xs)
# print(ys)
# prob = forwardpass()
# loss(5,prob)