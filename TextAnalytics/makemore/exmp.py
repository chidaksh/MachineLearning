import torch
import matplotlib.pyplot as plt

words = open('names.txt','r').read().splitlines()
# print(words[:5])
# print(len(words))
# bc = {}
# for w in words:
#     chs = ['<S>'] + list(w) + ['<E>']
#     for (ch1, ch2) in zip(chs, chs[1:]):
#         bigram = (ch1, ch2)
#         bc[bigram] = bc.get(bigram, 0) + 1
        # print(ch1, ch2)
# print(sorted(bc.items(),key = lambda x:x[1], reverse=True))
arr = torch.zeros((27,27), dtype = torch.int32)
distinct_chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(distinct_chars)}
stoi['.'] = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for (ch1, ch2) in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        arr[ix1,ix2] += 1

itos = {i:s for s,i in stoi.items()}

def plot():
    plt.figure(figsize=(16,16))
    plt.imshow(arr, cmap = "Blues")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            chstr = itos[i] + itos[j]
            plt.text(j,i,chstr,ha="center",va="bottom",color="gray")
            plt.text(j,i,arr[i,j].item(),ha="center",va="top",color="gray")
    plt.axis('off')
    plt.show()
    return


# p_ = torch.rand(3,generator=genarator)
# p_ /= p_.sum()
# print(p_)
genarator = torch.Generator().manual_seed(2147483647)
P = (arr+1).float()
P /= P.sum(dim = 1,keepdim = True)
# print(P[0].sum())
def averagenll():
    count = 0
    nll = 0.0
    for w in ["chidakshzq"]:
        chs = ['.'] + list(w) + ['.']
        for (ch1, ch2) in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1,ix2]
            count += 1
            nll += torch.log(prob)
            print(f'{ch1}{ch2}:{prob:0.4f} {nll:0.4f}')
    print(f"Log Likelihood value is {nll}")
    nll *= -1
    print(f'Average neegative log likelihood is {nll/count}')
    return

def sampleWord(numiter):
    for i in range(numiter):
        
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p,num_samples=1,replacement=True,generator=genarator).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))
    
# sampleWord(50)
averagenll()

