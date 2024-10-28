import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

def occ(mac_values, B_ADC, xr, LC=0):
    # Optimal Clipping Criterion
    mac_tensor = torch.from_numpy(mac_values).float()
    delta = (xr - LC) / (2 ** B_ADC)
    temp = torch.from_numpy(mac_values[mac_values > xr]).float()  # Boolean indexing to filter
    est = torch.mean(torch.pow((temp - xr), 2))
    sigsq = est * (temp.size().numel() / mac_tensor.size().numel())
    delsq = (delta ** 2) / 12
    mse = delsq + sigsq
    return [mse, B_ADC, xr]

def clip(mac_values, xr_start, xr_end, B_start, B_end, layer):
    results = []
    y_list = []
    for i in range(B_start, B_end + 1):
        y_list.append([i])
        for j in range(xr_start, xr_end + 1):
            p = occ(mac_values, i, j)
            y_list[i - B_start].append([])
            y_list[i - B_start][1].append(p[0])  # Format of y_list is [[6,[mac_6_tensor]], [7, [mac_7_tensor]], ....]

    x_list = torch.arange(xr_start, xr_end + 1)

    for k in range(B_start, B_end + 1):
        plt.plot(x_list.numpy(), y_list[k - B_start][1])  # MSE plot for k-bit ADC
        ymin = min(y_list[k - B_start][1])
        ymin_arg = y_list[k - B_start][1].index(ymin)
        plt.plot((x_list.numpy())[ymin_arg], y_list[k - B_start][1][ymin_arg], 'ro')
        y_rounded = round(y_list[k - B_start][1][ymin_arg].item(), 5)
        plt.text((x_list.numpy())[ymin_arg], y_list[k - B_start][1][ymin_arg],
                 f'{k}B:({(x_list.numpy())[ymin_arg]},{y_rounded})', verticalalignment='bottom', horizontalalignment='center')
        results.append([k, (x_list.numpy())[ymin_arg], y_list[k - B_start][1][ymin_arg]])

    plt.xlabel("Clip Point")
    plt.ylabel("MSE")
    B_range = range(B_start, B_end + 1)
    plt.title(f"LAYER {layer}")
    plt.show()
    return results

def histplot(mac_tens):
    plt.hist(mac_tens[1], bins=1000, histtype='stepfilled')
    plt.xlabel('MAC Value')
    plt.ylabel('# of Occurrences')
    plt.title(f"Instance {mac_tens[0]}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, required=True, help='Path to the file containing MAC values')
    parser.add_argument('--xr_start', type=int, required=True, help='Start value for xr range')
    parser.add_argument('--xr_end', type=int, required=True, help='End value for xr range')
    parser.add_argument('--B_start', type=int, required=True, help='Start value for B range')
    parser.add_argument('--B_end', type=int, required=True, help='End value for B range')
    parser.add_argument('--layer', type=int, required=True, help='Layer number')

    args = parser.parse_args()

    iter1_torch = torch.load(args.file, weights_only=True).to('cpu')
    iter1 = iter1_torch.numpy()
    print(f"{iter1.reshape(-1).shape[0]} MAC Values")
    results_occ = clip(iter1.reshape(-1), args.xr_start, args.xr_end, args.B_start, args.B_end, layer=args.layer)

    print("B", "Clip", "MSE")
    for i in results_occ:
        print(i[0], i[1], i[2].item())
