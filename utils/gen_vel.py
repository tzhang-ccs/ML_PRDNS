import numpy as np
import os
num = 200

process = 'test'
case_name = 'out-entrainment2dm_g_1024'
os.chdir(f"/pscratch/sd/z/zhangtao/PR_DNS_base_ray/{case_name}/python_data")

for i in range(num):
    xvel = np.load(f'xvel/{process}/data-{i:04d}.npz')
    xdata_x = xvel['data_x']
    xdata_y = xvel['data_y']
    
    yvel = np.load(f'yvel/{process}/data-{i:04d}.npz')
    ydata_x = yvel['data_x']
    ydata_y = yvel['data_y']
    
    data_x = (xdata_x**2 + ydata_x**2) ** 0.5
    data_y = (xdata_y**2 + ydata_y**2) ** 0.5
    
    os.system('mkdir -p vel')
    np.savez(f'vel/{process}/data-{i:04d}',data_x=data_x,data_y=data_y)
    print(f'data-{i:04d}')
