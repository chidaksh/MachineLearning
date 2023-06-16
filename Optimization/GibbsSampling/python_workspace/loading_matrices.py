import numpy as np
import torch
from scipy.io import loadmat

if __name__ == "__main__":
    mat1 = loadmat('../ForceVec.mat')
    mat2 = loadmat('../Stiffness.mat')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device activated is: ",device)
    ForceVec = torch.from_numpy(mat1['B']).float()
    stiffness = torch.from_numpy(mat2['K']).float()
    ForceVec = ForceVec.to(device)
    stiffness = stiffness.to(device)
    sigma = torch.linalg.inv(stiffness)
    initial_norm = torch.linalg.norm(sigma)
    sigma = sigma.to(device)
    print("Initial norm of sigma is: ",initial_norm)
    dim = int(mat1['B'].shape[0])
    