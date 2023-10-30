import sys
sys.path.append('../utils')

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *
import pdb
import operator
from functools import reduce
from functools import partial
import sys
from timeit import default_timer
from torchsummary import summary
from Adam import Adam
from loguru import logger
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import argparse

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        t1 = time.time()
        x_ft = torch.fft.rfft2(x)

        t2 = time.time()
        #print(f'fft: {t2 - t1: .4f}s')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] =             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] =             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        t3 = time.time()
        #print(f'multiple: {t3 - t2: .4f}s')

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        t4 = time.time()
        #print(f'irfft: {t4 - t3: .4f}s')
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        t1 = time.time()
        x1 = self.conv0(x)
        t2 = time.time()

        t1 = time.time()
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        t2 = time.time()

        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def gen4ddata(data,TT):
    data_4d = np.zeros((data.shape[0]-TT, data.shape[1], data.shape[2], TT))

    for i in range(data_4d.shape[0]):
        for j in range(TT):
            data_4d[i,:,:,j] = data[i+j,:,:]

    return data_4d

case_name = 'out-entrainment2dm_d_0.512_g_2048_init1'
case_name = 'out-entrainment2dm_g_1024'
learning_rate = 0.002
scheduler_step = 100
scheduler_gamma = 0.5
epochs = 50
modes = 30#60#12
width = 32
step = 1
device = torch.device('cuda:1')
batch_size = 5
res = 1024

logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
log_path = f'../../logs/FNO_log_{case_name}_{var}'
if os.path.exists(log_path):
    os.remove(log_path)
ii = logger.add(log_path)

dataset_train_path = f'../../data/pr-dns/data/out-entrainment2dm_d_0.512_g_{res}_init2/pr_dns_{var}.npy'
dataset_3d = np.load(dataset_train_path)
dataset_4d = gen4ddata(dataset_3d, 20)
del dataset_3d
train_a = dataset_4d[:ntrain,:,:,:T_in].astype(np.float32)
train_u = dataset_4d[:ntrain,:,:,T_in:T+T_in].astype(np.float32)
test_a = dataset_4d[-ntest:,:,:,:T_in].astype(np.float32)
test_u = dataset_4d[-ntest:,:,:,T_in:T+T_in].astype(np.float32)
del dataset_4d

train_a = torch.from_numpy(train_a).to(device)
train_u = torch.from_numpy(train_u).to(device)
test_a = torch.from_numpy(test_a).to(device)
test_u = torch.from_numpy(test_u).to(device)

    
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
 
del train_a
del train_u
del test_a

    
if process == 'Train':
    train_l2 = []
    model = FNO2d(modes, modes, width).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)
     
    for ep in range(epochs):
        model.train()
        train_l2_step = 0
        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
    
            im = model(xx)

            loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(im.shape[0], -1))
            train_l2_step += loss.item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
