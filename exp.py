import os
import matplotlib.pyplot as plt
output_path = './experiments'
import torch
import numpy as np

def read(dir):
    topk=[]
    file_path = os.path.join(output_path,dir)
    with open(file_path,'r') as f:
        for line in f.readlines():
            topk.append(float(line.split('\n')[0]))
    return topk

if __name__ == '__main__':
    # max = 'result_max.txt'
    # mean = 'result_mean.txt'
    # topk = 'result_topkmean.txt'
    #
    # max_data = read(max)
    # mean_data = read(mean)
    # topk_data = read(topk)
    # topk_data = topk_data[:-1]
    #
    # i = range(800)
    # plt.figure()
    # plt.plot(i,max_data,'r',i,mean_data,'y',i,topk_data,'b')
    # plt.show()
    # a = torch.arange(0,12).view(3,4)
    # print(a)
    #
    # index = torch.nonzero(a)
    #
    # index = index.transpose(-1,-2)
    # c = torch.Tensor([0.3,0.5,0.8])
    # print((c>0.3).type_as(a))
    # d = torch.Tensor([0.1,0.8,0.1])
    # print(torch.where(c>0.3,1.0,0.2))
    # print(torch.mean(c))
    d = torch.rand(2,3,3)
    d_norm = torch.norm(d,p=2, dim =-1,keepdim=True)
    print()
    print(d_norm)
    d_n = torch.div(d,d_norm)
    print(torch.matmul(d_n,d_n.transpose(-1,-2)))

