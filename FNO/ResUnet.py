#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from loguru import logger
from datetime import datetime
import sys
from Adam import Adam
from utilities3 import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.nn as nn
from torchsummary import summary

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
import time

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


# # config

# In[3]:


casename = 'sameinit'
ntrain = 2000#4000
ntest = 1000#2000
T_in = 1
learning_rate = 0.002
scheduler_step = 100
scheduler_gamma = 0.5
epochs = 500
batch_size = 50
device = torch.device('cuda:0')

n1 = 32
n2 = 64
n3 = 128

# # model

# ## block

# In[4]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


# In[5]:


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False) 

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
    


# In[6]:


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


# ## UNet

# In[8]:


class Encoder(nn.Module):
    #def __init__(self, chs=(1,64,128,256,512,1024)):
    def __init__(self, chs=(1,32,64,128)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ResNetBasicBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    


# In[9]:


class Decoder(nn.Module):
    #def __init__(self, chs=(1024, 512, 256, 128, 64)):
    def __init__(self, chs=(128, 64, 32)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) 
                                         for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([ResNetBasicBlock(chs[i], chs[i+1]) 
                                         for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


# In[10]:

class UNet(nn.Module):
    #def __init__(self, enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
    def __init__(self, enc_chs=(1,n1,n2,n3), dec_chs=(n3, n2,n1), 
                 num_class=1, retain_dim=False, out_sz=(128,128)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


# # load data

# In[12]:

def gen4ddata(data,TT):
    data_4d = np.zeros((data.shape[0]-TT, data.shape[1], data.shape[2], TT))

    for i in range(data_4d.shape[0]):
        for j in range(TT):
            data_4d[i,:,:,j] = data[i+j,:,:]

    return data_4d

if __name__ == '__main__':
 
    if process != 'ITest':
        dataset_train_path = f'../../data/pr-dns/data/out-entrainment2dm_d_0.512_g_{res}_init1/pr_dns_{var}.npy'
    else:
        dataset_train_path = f'../../data/pr-dns/data/out-entrainment2dm_d_0.512_g_{res}_init1/pr_dns_{var}.npy'

        dataset_ref_path = f'../../data/pr-dns/data/out-entrainment2dm_d_0.512_g_{res}_init2/pr_dns_{var}.npy'
        dataset_ref_3d = np.load(dataset_ref_path)
        dataset_ref_4d = gen4ddata(dataset_ref_3d, 20)
        train_ref_a = dataset_ref_4d[:ntrain,:,:,:T_in].astype(np.float32)
        scaler_ref = MinMaxNormalizer(train_ref_a)
        del dataset_ref_3d
        del dataset_ref_4d
        del train_ref_a

    dataset_3d = np.load(dataset_train_path)
    dataset_4d = gen4ddata(dataset_3d,20)
	
    print(dataset_4d.shape)
    #print(dataset_4d2.shape)
    
    train_a = np.expand_dims(dataset_4d[:ntrain,:,:,9],axis=1)
    train_u = np.expand_dims(dataset_4d[:ntrain,:,:,9+T_in],axis=1)
    test_a = np.expand_dims(dataset_4d[-ntest:,:,:,9],axis=1)
    test_u = np.expand_dims(dataset_4d[-ntest:,:,:,9+T_in],axis=1)
    
    if norm_flag == 'True':
        print("Do normalization========")
        scaler  = MinMaxNormalizer(train_a)
        train_a = scaler.encode(train_a)
        train_u = scaler.encode(train_u)
        test_a  = scaler.encode(test_a)
        test_u  = scaler.encode(test_u)

    train_a = torch.from_numpy(train_a).to(device)
    train_u = torch.from_numpy(train_u).to(device)
    test_a = torch.from_numpy(test_a).to(device)
    test_u = torch.from_numpy(test_u).to(device)
    
    print(dataset_4d.shape)
    print(train_a.shape)
    print(train_u.shape)
    print(test_a.shape)
    print(test_u.shape)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    del dataset_3d
    del dataset_4d
    del train_a
    del train_u
    del test_a
#    del test_u
    
    train_l2 = []
    test_l2 = []
    
    if process == 'Train':
        model = UNet(retain_dim=True,out_sz=(int(res),int(res))).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        myloss = LpLoss(size_average=False)
        
        #print(model)
        log_path = f'../../logs/ResNet_log_{var}_r_{res}'
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
                xx = xx.to(device,dtype=torch.float)
                yy = yy.to(device,dtype=torch.float)

                t1 = time.time()
                im = model(xx)
                t2 = time.time()

                t3 = time.time()
                loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(im.shape[0], -1))    
                train_l2_step = train_l2_step +  loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t4 = time.time()
                forward_time = forward_time + t2 - t1
                backward_time = backward_time + t4 - t3
            t6 = time.time()
#            print(f"{forward_time=}")
#            print(f"{backward_time=}")
#            print(f"train_time={t6-t5}")

            test_l2_step = 0
            model.eval()
            with torch.no_grad():
                for xx, yy in test_loader:
                    xx = xx.to(device,dtype=torch.float)
                    yy = yy.to(device,dtype=torch.float)
                    
                    t7 = time.time()
                    im = model(xx)
                    t8 = time.time()
                    infer_time = infer_time + t8 - t7

                    loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(im.shape[0], -1))
                    test_l2_step += loss.item()
             
#            print(f"{infer_time=}")
            scheduler.step()
            
            l2_train_step,  l2_test_step = train_l2_step/ntrain, test_l2_step/ntest
            
            train_l2.append(l2_train_step)
            test_l2.append(l2_test_step)
            logger.info(f'{ep=} {l2_train_step=:.5f} {l2_test_step=:.5f}')
                    
        torch.save(model, f'../../models/archive/ResNet_model_{var}_r_{res}_init1') 
        logger.remove(ii)
    
    elif process == 'Test':
        test_l2_step = 0
        myloss = LpLoss(size_average=False)
        model = torch.load(f'../../models/archive/ResNet_model_{var}_r_{res}_init1', map_location=device)
        model.eval()

        preds = torch.zeros(test_u.shape).to(device,dtype=torch.float)
        index = 0

        print(f'{preds.shape=}')

        with torch.no_grad():
            infer_time = 0
            for xx, yy in test_loader1:
                xx = xx.to(device,dtype=torch.float)
                yy = yy.to(device,dtype=torch.float)
                
                t1 = time.time()
                out = model(xx)
                t2 = time.time()
                infer_time = infer_time + t2 - t1

                preds[index] = out
                loss = myloss(out.reshape(out.shape[0], -1), yy.reshape(out.shape[0], -1))
                test_l2_step += loss.item()
            
                if index == 1000:
                    break
                index = index + 1
            
            #print(index)
            #print(f"{infer_time=}")
            e_time = time.time()
            test_score = test_l2_step/ntest
            tests = torch.stack([preds,test_u])
            if norm_flag == 'True':
                tests = scaler.decode(tests)

        np.save(f"../../tests/ResNet_out_{var}_r_{res}_init1",tests.cpu().numpy())

    elif process == 'ITest': #different init test
        test_l2_step = 0
        myloss = LpLoss(size_average=False)
        model = torch.load(f'../../models/archive/ResNet_model_{var}_r_128', map_location=device)
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

        np.save(f"../../itests/ResNet_out_{var}_init2_to_init1",tests)
