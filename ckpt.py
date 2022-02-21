import torch
M= torch.arange(0,12).view(3,4)
print(M)
print(torch.matmul(M,M.transpose(-1,-2)))