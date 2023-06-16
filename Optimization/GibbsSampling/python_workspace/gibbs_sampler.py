import numpy as np
import torch
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import return_arrays,plot_graph,plot_graph_multiple,take_mean_dicts
import pickle as pkl
import time

def normal(mean,var):
    return mean +  math.sqrt(var) * float(torch.randn(1))


def gibbs_sampler(num_iter,start,lr = 1):
    iter = num_iter
    x = torch.zeros((iter*dim,dim)).to(device)
    for j in range(iter):
        for i in range(dim):
            temp = torch.cat((start[:i],start[i+1:]))
            var = 1/stiffness[i][i]
            intermediate = (torch.cat((stiffness[i][:i],stiffness[i][i+1:]))/(-stiffness[i][i]))
            mean = intermediate @ temp.T
            start[i] = lr * normal(mean,var)
            x[j*dim+i] = start
    return x

def sigma_convergence(x,sigma,samples,num_iter):
    acc_sigma_convergence = {}
    sum = torch.zeros((dim,dim)).to(device)
    burn_in = num_iter//10
    count = 0
    visited = 0

    for j in tqdm(range(num_iter),desc = "Progress bar for sigma"):
        for i in range(dim):
            ind = j*dim + i
            sum = (1 - 1/(ind + 1))*sum  + torch.outer(x[j*dim+i].T, x[j*dim+i])/(ind + 1)
            count += 1
            if count == samples:
                visited = 1
                break
            dist = (torch.linalg.norm(sum - sigma).to('cpu'),torch.norm((sum-sigma).to('cpu'),p = float('inf')))
        acc_sigma_convergence[j] = dist

        if visited == 1:
            break
    return dist,acc_sigma_convergence


def jump_sigma_convergence(x,sigma,samples,num_iter):
    acc_jump_sigma_convergence = {}
    jump_sum = torch.zeros((dim,dim)).to(device)
    # burn_in = num_iter//10
    burn_in = 0
    count = 0
    for j in tqdm(range(burn_in,num_iter),desc = "Progress bar for jump sigma"):
        # if (j == 0):
        #         print(jump_sum)
        #         print(x[0])
        ind = j
        jump_sum = (1 - 1/(ind + 1)) * jump_sum + torch.outer(x[j*dim].T, x[j*dim])/(ind + 1)
        count += 1
        # if (j == 0):
        #         print('entered jump')
        #         print(jump_sum)
        dist = (torch.linalg.norm(jump_sum - sigma).to('cpu'),torch.norm((jump_sum - sigma).to('cpu'),p = float('inf')))
        acc_jump_sigma_convergence[j] = dist
        if count == samples:
            break
    return dist,acc_jump_sigma_convergence 


def conj_sigma_convergence(x,A,sigma,samples,num_iter):
    acc_conj_sigma_convergence = {}
    val = torch.zeros((dim,dim)).to(device)
    # burn_in = num_iter//10
    burn_in = 0
    count = 0
    for j in tqdm(range(burn_in,num_iter),desc = "Progress bar for conj"):
            # if (j == 0):
            #     print(val)
            #     print(x[0])
            val = val  + torch.outer(x[j*dim].T, x[j*dim])/(x[j*dim] @ A @ x[j*dim])
            count += 1
            # if (j == 0):
            #     print(val)
            dist = (torch.linalg.norm((val/count * dim) - sigma).to('cpu'),torch.norm(((val/count * dim) - sigma),p = float('inf')).to('cpu'))
            acc_conj_sigma_convergence[j] = dist
            if count == samples:
                break
    # val = (val/samples)*N
    # dist = (torch.linalg.norm(val- sigma),torch.norm(val - sigma ,p = float('inf')))
    return dist,acc_conj_sigma_convergence

def sigma_conj_sigma_convergence(x,A,sigma,samples,num_iter):
    acc_sigma_conj_sigma_convergence = {}
    for_sigma = torch.zeros((dim,dim)).to(device)
    for_conj = torch.zeros((dim,dim)).to(device)
    # burn_in = num_iter//10
    burn_in = 0
    count = 0
    for j in tqdm(range(burn_in,num_iter),desc = "Progress bar for sigma conj"):
            # if (j == 0):
            #     print(val)
            #     print(x[0])
            alpha = (1/2)
            count += 1
            for_sigma += torch.outer(x[j*dim].T, x[j*dim])/samples
            for_conj += torch.outer(x[j*dim].T, x[j*dim])/(x[j*dim] @ A @ x[j*dim])
            val = alpha * for_sigma + (1-alpha) * (for_conj/count * dim)
            # (alpha * torch.outer(x[j*dim].T, x[j*dim])/samples) +
            # if (j == 0):
            #     print(val)
            dist = (torch.linalg.norm(val - sigma).to('cpu'),torch.norm((val - sigma),p = float('inf')).to('cpu'))
            acc_sigma_conj_sigma_convergence[j] = dist
            if count == samples:
                break
    # val = (val/samples)*N
    # dist = (torch.linalg.norm(val- sigma),torch.norm(val - sigma ,p = float('inf')))
    return dist,acc_sigma_conj_sigma_convergence


if __name__ == "__main__":
    mat1 = loadmat('ForceVec.mat')
    mat2 = loadmat('Stiffness.mat')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device activated is: ",device)
    global ForceVec 
    ForceVec = torch.from_numpy(mat1['B']).float()
    global stiffness 
    stiffness = torch.from_numpy(mat2['K']).float()
    global dim 
    dim = int(mat1['B'].shape[0])
    ForceVec = ForceVec.to(device)
    stiffness = stiffness.to(device)
    global sigma 
    sigma = torch.linalg.inv(stiffness)
    initial_norm = torch.linalg.norm(sigma)
    sigma = sigma.to(device)
    print("Initial norm of sigma is: ",initial_norm)
    # dim = int(mat1['B'].shape[0])
    # exec(open("./loading_matrices.py").read())
    runs = 5
    dict_acc_sigma_convergence = {}
    dict_acc_jump_sigma_convergence = {}
    dict_acc_conj_sigma_convergence = {}
    dict_acc_sigma_conj_sigma_convergence = {}
    num_iter = 10000
    start_time = time.time()
    count = 0
    for i in tqdm(range(runs),desc = "Progress bar"):
        start = torch.zeros(dim).to(device)
        x = gibbs_sampler(num_iter,start)
        count += 1
        norm,acc_sigma_convergence = sigma_convergence(x, sigma, num_iter * dim, num_iter)
        dict_acc_sigma_convergence[i] = acc_sigma_convergence
        # print(i,'1')
        if count%2 == 1:
                pkl.dump(dict_acc_sigma_convergence, open("sigma.p","wb"))
        
        jump_norm,acc_jump_sigma_convergence = jump_sigma_convergence(x, sigma, num_iter, num_iter)
        dict_acc_jump_sigma_convergence[i] = acc_jump_sigma_convergence
        # print(i,'2')
        if count%2 == 1:
            pkl.dump(dict_acc_jump_sigma_convergence, open("jump_sigma.p","wb"))
        
        conj_norm,acc_conj_sigma_convergence = conj_sigma_convergence(x, stiffness, sigma, num_iter, num_iter)
        dict_acc_conj_sigma_convergence[i] = acc_conj_sigma_convergence
        if count%2 == 1:
            pkl.dump(dict_acc_conj_sigma_convergence, open("conj_sigma.p","wb"))

        sigma_conj_norm,acc_sigma_conj_sigma_convergence = sigma_conj_sigma_convergence(x, stiffness, sigma, num_iter, num_iter)
        dict_acc_sigma_conj_sigma_convergence[i] = acc_sigma_conj_sigma_convergence
        # print(i,'3')
        if count%2 == 1:
            pkl.dump(dict_acc_sigma_conj_sigma_convergence, open("sigma_conj_sigma.p","wb"))

    plot_graph_multiple(dict_acc_sigma_convergence,dict_acc_jump_sigma_convergence,dict_acc_conj_sigma_convergence,
                        dict_acc_sigma_conj_sigma_convergence,"normal_averaged_convergence_plots_" + str(dim),average = True)
    end_time = time.time()
    print(f'Total time taken to execute the above program for matrices of dimension {dim} is: {end_time - start_time}')