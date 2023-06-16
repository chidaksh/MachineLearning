import numpy as np
import torch
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt
from plotting import return_arrays,plot_graph,plot_graph_multiple,take_mean_dicts
from tqdm import tqdm
import time
import sys

def normal(mean,var):
    return mean +  torch.sqrt(var) * torch.randn(mean.size()).to(device)

def new_gibbs_sampler(num_iter,start):
    iter = num_iter
    x = torch.zeros((iter,dim)).to(device)
    diagonal = torch.diag(stiffness)
    var = 1/diagonal
    for j in range(iter):
        # print(stiffness.shape,start.shape)
        intermediate = stiffness @ start.T
        temp = torch.multiply(start,diagonal)
        mean = torch.div((intermediate - temp),(-1 * diagonal))
        x[j] = start
        start = normal(mean,var)
    return x

def new_sigma_convergence(x,sigma,samples,num_iter):
    acc_sigma_convergence = {}
    sum = torch.zeros((dim,dim)).to(device)
    burn_in = num_iter//10
    count = 0

    for j in range(num_iter):
        ind = j
        sum = (1 - 1/(ind + 1))*sum  + torch.outer(x[j].T, x[j])/(ind + 1)
        count += 1
        dist = (torch.linalg.norm(sum - sigma).to('cpu'),torch.norm((sum-sigma).to('cpu'),p = float('inf')))
        acc_sigma_convergence[j] = dist
        if count == samples:
            break
    
    return dist,acc_sigma_convergence

def new_conj_convergence(x,A,sigma,samples,num_iter):
    acc_conj_sigma_convergence = {}
    val = torch.zeros((dim,dim)).to(device)
    # burn_in = num_iter//10
    burn_in = 0
    count = 0
    for j in range(burn_in,num_iter):
            # if (j == 0):
            #     print(val)
            #     print(x[0])
            if count != 0:
                val = val  + torch.outer(x[j].T, x[j])/(x[j] @ A @ x[j])
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

def new_sigma_conj_convergence(x,A,sigma,samples,num_iter):
    acc_sigma_conj_sigma_convergence = {}
    for_sigma = torch.zeros((dim,dim)).to(device)
    for_conj = torch.zeros((dim,dim)).to(device)
    # burn_in = num_iter//10
    burn_in = 0
    count = 0
    for j in range(burn_in,num_iter):
            # if (j == 0):
            #     print(val)
            #     print(x[0])
            alpha = (1/2)

            for_sigma = (1 - 1/(j + 1))*for_sigma  + torch.outer(x[j].T, x[j])/(j + 1)
            if count != 0:
                for_conj += torch.outer(x[j].T, x[j])/(x[j] @ A @ x[j])
            count += 1
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
    start_time = time.time()
    mat1 = loadmat('ForceVec_1225.mat')
    mat2 = loadmat('Stiffness_1225.mat')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device activated is: ",device)
    global ForceVec 
    ForceVec = torch.from_numpy(mat1['B']).float()
    global stiffness 
    stiffness = torch.from_numpy(mat2['K']).float()
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
    runs = int(sys.argv[2])
    avg_acc_sigma_convergence = {}
    # avg_acc_jump_sigma_convergence = {}
    avg_acc_conj_convergence = {}
    avg_acc_sigma_conj_convergence = {}
    num_iter = int(sys.argv[1])
    avg_loss_sigma_convergence = {}
    avg_loss_jump_sigma_convergence = {}
    avg_loss_conj_sigma_convergence = {}
    avg_loss_sigma_conj_sigma_convergence = {}

    for i in tqdm(range(runs),desc = "Progress bar"):
        start = torch.zeros(dim).to(device)
        x = new_gibbs_sampler(num_iter,start)
        norm,acc_sigma_convergence = new_sigma_convergence(x, sigma, num_iter, num_iter)
        avg_acc_sigma_convergence[i] = acc_sigma_convergence
        
        # loss_norm,loss_sigma_convergence = approx_gibbs_solution(x,ForceVec,solution_matrix,num_iter * dim,num_iter)
        # avg_loss_sigma_convergence[i] = loss_sigma_convergence
        # print(i,'1')

        jump_norm,acc_conj_convergence = new_conj_convergence(x, stiffness,sigma, num_iter, num_iter)
        avg_acc_conj_convergence[i] = acc_conj_convergence
        # loss_jump_norm,loss_jump_sigma_convergence = approx_jump_gibbs_solution(x,ForceVec,solution_matrix,num_iter,num_iter)
        # avg_loss_jump_sigma_convergence[i] = loss_jump_sigma_convergence
        # print(i,'2')

        # conj_norm,acc_conj_sigma_convergence = conj_sigma_convergence(x, stiffness, sigma, num_iter, num_iter)
        # dict_acc_conj_sigma_convergence[i] = acc_conj_sigma_convergence
        # loss_conj_norm,loss_conj_sigma_convergence = approx_conj_gibbs_solution(x,ForceVec,stiffness,solution_matrix,num_iter,num_iter)
        # avg_loss_conj_sigma_convergence[i] = loss_conj_sigma_convergence

        sigma_conj_norm,acc_sigma_conj_convergence = new_sigma_conj_convergence(x, stiffness, sigma, num_iter, num_iter)
        avg_acc_sigma_conj_convergence[i] = acc_sigma_conj_convergence
        # loss_sigma_conj_norm,loss_sigma_conj_sigma_convergence = approx_sigma_conj_gibbs_solution(x,ForceVec,stiffness,solution_matrix,num_iter,num_iter)
        # avg_loss_sigma_conj_sigma_convergence[i] = loss_sigma_conj_sigma_convergence
        # print(i,'3')
    plot_graph(avg_acc_sigma_convergence,avg_acc_conj_convergence,avg_acc_sigma_conj_convergence,
                "averaged_convergence_plots_" + str(dim),average = True)
    end_time = time.time()
    print(f'Total time taken to execute the above program for matrices of dimension {dim} is: {end_time - start_time}')
