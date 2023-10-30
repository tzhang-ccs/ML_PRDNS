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
from Adam import Adam
from loguru import logger
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--var", required=True)
    parser.add_argument("-p", "--process", required=True)
    parser.add_argument("-r", "--resolution", required=True)
    parser.add_argument("-n", "--normalization",default="False")
    #parser.add_argument("process",help="Train or Test")
    args = parser.parse_args()
    
    print(args.var)
    print(args.process)
    
    var = args.var
    process = args.process
    res = args.resolution
    norm_flag = args.normalization
    
    torch.manual_seed(0)
    np.random.seed(0)
    # In[2]:
    
    
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d-%H_%M_%S")
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
    logger.add(sys.stdout, format=fmt)


# In[3]:

casename = 'sameinit'
ntrain = 4000#2000#4000
ntest = 2000#1000#2000
T_in = 10
T = 1
learning_rate = 0.002
scheduler_step = 100
scheduler_gamma = 0.5
epochs = 500
batch_size = 50#100
modes = 30#60#12
width = 32
step = 1
device = torch.device('cuda:0')

#n1 = 32
#n2 = 64
#n3 = 128
# # FNO model

# In[4]:


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

# # load data


if __name__ == '__main__':

    if process == 'ITest':
        norm_init2 = {}
        norm_init2['xvel'] = [0.56544, -0.33976]
        norm_init2['temp'] = [0.64661, 270.75000]

        dataset_train_path = f'../../data/pr-dns/data/out-entrainment2dm_d_0.512_g_{res}_init1/pr_dns_{var}.npy'
        dataset_3d = np.load(dataset_train_path)
        dataset_4d = gen4ddata(dataset_3d, 20)
        del dataset_3d
        train_a = dataset_4d[:ntrain,:,:,:T_in].astype(np.float32)
        train_u = dataset_4d[:ntrain,:,:,T_in:T+T_in].astype(np.float32)
        test_a = dataset_4d[-ntest:,:,:,:T_in].astype(np.float32)
        test_u = dataset_4d[-ntest:,:,:,T_in:T+T_in].astype(np.float32)
        del dataset_4d

    #if process == 'ITest':
    else:
        norm_init2 = {}
        norm_init2['xvel'] = []
        norm_init2['temp'] = []
        norm_init2['vel'] = []

        if process == 'STest':
            norm_init2['xvel'] = [0.28377, -0.13178]
            norm_init2['temp'] = [0.83325, 270.56335]

        dataset_train_path = f'/work/tzhang/FNO_DNS/data/ml_pde_prdns/out-entrainment2dm_d_0.512_g_{res}_init2/pr_dns_{var}.npy'
        dataset_3d = np.load(dataset_train_path)
        dataset_4d = gen4ddata(dataset_3d, 20)
        del dataset_3d
        train_a = dataset_4d[:ntrain,:,:,:T_in].astype(np.float32)
        train_u = dataset_4d[:ntrain,:,:,T_in:T+T_in].astype(np.float32)
        test_a = dataset_4d[-ntest:,:,:,:T_in].astype(np.float32)
        test_u = dataset_4d[-ntest:,:,:,T_in:T+T_in].astype(np.float32)
        del dataset_4d


    #train_a = np.expand_dims(dataset_4d[:ntrain,:,:,0],axis=1)
    #train_u = np.expand_dims(dataset_4d[:ntrain,:,:,T_in],axis=1)
    #test_a = np.expand_dims(dataset_4d[-ntest:,:,:,0],axis=1)
    #test_u = np.expand_dims(dataset_4d[-ntest:,:,:,T_in],axis=1)
    if norm_flag == 'True':
        print("Do normalization========")
        scaler  = MinMaxNormalizer(train_a,norm_init2[var])
        train_a = scaler.encode(train_a)
        train_u = scaler.encode(train_u)
        test_a  = scaler.encode(test_a)
        test_u  = scaler.encode(test_u)
    
    train_a = torch.from_numpy(train_a).to(device)
    train_u = torch.from_numpy(train_u).to(device)
    test_a = torch.from_numpy(test_a).to(device)
    test_u = torch.from_numpy(test_u).to(device)

    print(train_a.shape)
    print(train_u.shape)
    print(test_a.shape)
    print(test_u.shape)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
 
    del train_a
    del train_u
    del test_a
#    del test_u
#    del dataset_3d
    
    # # train
    
    # In[ ]:

    train_l2 = []
    test_l2 = []
    
    if process == 'Train':
        model = FNO2d(modes, modes, width).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        myloss = LpLoss(size_average=False)
        #y_normalizer.cuda()
        
        log_path = f'../../logs/FNO_log_{var}_r_{res}_nf'
        print(log_path)
        if os.path.exists(log_path):
            os.remove(log_path)   
        ii = logger.add(log_path)
     
     
        for ep in range(epochs):
            model.train()
            train_l2_step = 0
            forward_time = 0
            backward_time = 0
            infer_time = 0
            t5 = time.time()
            for xx, yy in train_loader:
                xx = xx.to(device)
                yy = yy.to(device)
        
                t1 = time.time()
                im = model(xx)
                t2 = time.time()

                t3 = time.time()
                loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(im.shape[0], -1))
                train_l2_step += loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t4 = time.time()
                forward_time = forward_time + t2 - t1
                backward_time = backward_time + t4 - t3
            t6 = time.time()
                
#            test_l2_step = 0
#            model.eval()
#            with torch.no_grad():
#                for xx, yy in test_loader:
#                    xx = xx.to(device)
#                    yy = yy.to(device)
#                    t7 = time.time()
#                    im = model(xx)
#                    t8 = time.time()
#
#                    infer_time = infer_time + t8 - t7
#
#                    loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(im.shape[0], -1))
#        
#                    test_l2_step += loss.item()
        
#            print(f"{infer_time=}")
            scheduler.step()
            l2_train_step = train_l2_step/ntrain
            logger.info(f'{ep=} {l2_train_step=:.5f}')
        
        torch.save(model, f'../../models/FNO_model_{var}_{res}') 
        #torch.save(model, f'../../models/archive/FNO_model_{var}_r_{res}_nf') 
        #torch.save(model, f'../../models/archive/FNO_model_{var}_r_{res}_Dinit') 
        logger.remove(ii)


    elif process == 'Test':
        test_l2_step = 0
        myloss = LpLoss(size_average=False)
        #model = torch.load(f'../../models/archive/FNO_model_{var}_r_{res}_init1', map_location=device)
        model = torch.load(f'../../models/FNO_model_{var}_{res}', map_location=device)
        #model = torch.load(f'../../models/archive/FNO_model_{var}_r_{res}_nf', map_location=device)
        model.eval()

        preds = torch.zeros(test_u.shape).to(device,dtype=torch.float)

        print(f'{preds.shape=}')
        index = 0
        with torch.no_grad():
            s_time = time.time()
            infer_time = 0
            for xx, yy in test_loader1:
                xx = xx.to(device,dtype=torch.float)
                yy = yy.to(device,dtype=torch.float)
                
                t1 = time.time()
                out = model(xx)
                t2 = time.time()
                infer_time = infer_time + t2 - t1

                preds[index] = out
                index = index + 1
                loss = myloss(out.reshape(out.shape[0], -1), yy.reshape(out.shape[0], -1))
                test_l2_step += loss.item()


            e_time = time.time()

            test_score = test_l2_step/ntest
            print(f'{test_score=:.5f} time={e_time-s_time:.2f}')
            tests = torch.stack([preds,test_u]).view(2,-1,1,int(res),int(res)).cpu().numpy()
            if norm_flag == 'True':
                tests = scaler.decode(tests)

        np.save(f"../../tests/FNO_out_{var}_{res}",tests)

    elif process == 'STest': #super-resolution test
        test_l2_step = 0
        myloss = LpLoss(size_average=False)
        model = torch.load(f'../../models/archive/FNO_model_{var}_r_64', map_location=device)
        model.eval()

        preds = torch.zeros(test_u.shape).to(device,dtype=torch.float)
        index = 0

        print(f'{preds.shape=}')

        with torch.no_grad():
            s_time = time.time()
            for xx, yy in test_loader1:
                xx = xx.to(device,dtype=torch.float)
                yy = yy.to(device,dtype=torch.float)

                out = model(xx)
                preds[index] = out
                loss = myloss(out.reshape(out.shape[0], -1), yy.reshape(out.shape[0], -1))
                test_l2_step += loss.item()

                index = index + 1


            e_time = time.time()

            test_score = test_l2_step/ntest
            print(f'{test_score=:.5f} time={e_time-s_time:.2f}')
            tests = torch.stack([preds,test_u]).view(2,-1,1,int(res),int(res)).cpu().numpy()
            if norm_flag == 'True':
                tests[0,:,:,:,:] = scaler.decode(tests[0,:,:,:,:])
                tests[1,:,:,:,:] = scaler.decode(tests[1,:,:,:,:])

        np.save(f"../../stests/FNO_out_{var}_r64_to_r_{res}",tests)


    elif process == 'ITest': #different init test
        test_l2_step = 0
        myloss = LpLoss(size_average=False)
        model = torch.load(f'../../models/archive/FNO_model_{var}_r_128', map_location=device)
        model.eval()

        preds = torch.zeros(test_u.shape).to(device,dtype=torch.float)
        index = 0

        print(f'{preds.shape=}')

        with torch.no_grad():
            s_time = time.time()
            for xx, yy in test_loader1:
                xx = xx.to(device,dtype=torch.float)
                yy = yy.to(device,dtype=torch.float)

                out = model(xx)
                preds[index] = out
                loss = myloss(out.reshape(out.shape[0], -1), yy.reshape(out.shape[0], -1))
                test_l2_step += loss.item()

                index = index + 1


            e_time = time.time()

            test_score = test_l2_step/ntest
            print(f'{test_score=:.5f} time={e_time-s_time:.2f}')
            tests = torch.stack([preds,test_u]).view(2,-1,1,int(res),int(res)).cpu().numpy()
            if norm_flag == 'True':
                tests[0,:,:,:,:] = scaler.decode(tests[0,:,:,:,:])
                tests[1,:,:,:,:] = scaler.decode(tests[1,:,:,:,:])

        np.save(f"../../itests/FNO_out_{var}_init2_to_init1xxxx",tests)
