# Final clipping code
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group
# from utils.options import args
import matplotlib.pyplot as plt

def occ(mac_values,B_ADC,xr,LC=0):
    
    # Optimal Clipping Criterion
    mac_tensor = torch.from_numpy(mac_values).float()
    delta = (xr-LC)/(2**(B_ADC))
    temp = torch.from_numpy(mac_values[mac_values > xr]).float() # Boolean indexing to filter
    est = torch.mean(torch.pow((temp-xr),2))
    sigsq = est*((temp.size()).numel()/(mac_tensor.size()).numel())
    delsq = (delta**2)/12
    mse = delsq + sigsq
    
    return [mse,B_ADC,xr]

def clip(mac_values, xr_start, xr_end, B_start, B_end, layer):

    y_list = []
    for i in range(B_start, B_end + 1):
        y_list.append([i])
        for j in range(xr_start, xr_end + 1):
            p = occ(mac_values,i,j)
            y_list[i-B_start].append([])
            y_list[i-B_start][1].append(p[0]) # Format of y_list is [[6,[mac_6_tensor]], [7, [mac_7_tensor]], ....]

    x_list = torch.arange(xr_start, xr_end + 1)

    for k in range(B_start, B_end + 1):
        plt.plot(x_list.numpy(), y_list[k-B_start][1]) # MSE plot for k-bit ADC
        ymin = min(y_list[k-B_start][1])
        ymin_arg = (y_list[k-B_start][1]).index(ymin)
        plt.plot((x_list.numpy())[ymin_arg], y_list[k-B_start][1][ymin_arg], 'ro')
        y_rounded = round(y_list[k-B_start][1][ymin_arg].item(), 5)
        plt.text((x_list.numpy())[ymin_arg], y_list[k-B_start][1][ymin_arg], f'{k}B:({(x_list.numpy())[ymin_arg]},{y_rounded})',verticalalignment='bottom',horizontalalignment='center')

    plt.xlabel("Clip Point")
    plt.ylabel("MSE")
    B_range = range(B_start, B_end + 1)
    #leg = tuple(f'B = {i}' for i in B_range)
    #plt.legend(leg)
    plt.title(f"LAYER {layer}")
    plt.show()

def load_instance_and_tensor(filename):
        # Read the instance number and tensor from the text file
        with open(filename, "r") as f:
            lines = f.readlines()
            instance_number = int(lines[0].split(":")[1])
            tensor_flat = np.loadtxt(lines[2:])
            return [instance_number, tensor_flat]

def histplot(mac_tens):
    plt.hist(mac_tens[1],bins=1000,histtype='stepfilled')
    plt.xlabel('MAC Value')
    plt.ylabel('# of Occurences')
    plt.title(f"Instance {mac_tens[0]}")
    plt.show()

iter1_torch = torch.load(r'OCC_MACValues\conv1\MACVALUES_LEN256_LAYER1_ITER0.pt').to('cpu')
iter1 = iter1_torch.numpy()
print(iter1.shape)
print(iter1.reshape(-1).shape)
print(iter1.reshape(-1))
clip(iter1.reshape(-1), 10, 256, 3, 5, layer=1)
