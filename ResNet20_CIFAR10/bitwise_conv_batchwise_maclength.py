import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
count = 0 # To print only once

def quantize_tensor(tensor, num_bits, mode='signed'):
    if mode == 'unsigned':
        qmin = 0
        qmax = 2**num_bits - 1
    else:
        qmin = -2**(num_bits - 1)
        qmax = 2**(num_bits - 1) - 1

    if (tensor.abs().max() > tensor.max()):
        scale = (-qmin) / tensor.abs().max()
    else:
        scale = (qmax) / tensor.max()
    
    if mode == 'unsigned':
        tensor_q = (tensor * scale).round().clamp(min=0, max=2**num_bits - 1).to(torch.int32)
    else:
        tensor_q = (tensor * scale).round().clamp(min=-2**(num_bits - 1), max=2**(num_bits - 1) - 1).to(torch.int32)

    return tensor_q, scale

def quantize_to_binary(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
 
    # Ensure the tensor has integers only
    tensor = tensor.int()

    # Ensure all numbers are within the valid range for the given bit-width
    assert torch.all((tensor >= -(2**(num_bits-1))) & (tensor < 2**(num_bits-1))), \
        f"Values in tensor must fit within {num_bits}-bit signed integers"

    # Handle negative numbers by converting to two's complement
    max_val = 2 ** num_bits
    tensor = tensor % max_val  # Use modulus to wrap around for two's complement behavior

    # Prepare an empty tensor to store the binary representation (0s and 1s)
    shape = tensor.shape + (num_bits,)
    binary_tensor = torch.zeros(shape, dtype=torch.int32).to(device)

    # Convert each integer to binary and store the bits in binary_tensor
    for i in range(num_bits):
        # Extract the i-th bit of each number (from least significant to most significant)
        binary_tensor[..., num_bits - 1 - i] = (tensor >> i) & 1

    return binary_tensor

def im2col_5d(input, filter_h, filter_w, stride=1, padding=0):
    N, C, H, W, num_bits = input.shape  # Expecting input to be 5D (N, C, H, W, bits)
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    # Zero-pad the input tensor
    input_padded = torch.nn.functional.pad(input, (0, 0, padding, padding, padding, padding)).to(device)

    # Prepare an empty tensor for im2col
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w, num_bits), dtype=input.dtype).to(device)

    # Fill col tensor by sliding filter over the input
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride, :]

    # Reshape to 3D: (N * out_h * out_w, C * filter_h * filter_w, num_bits)
    col = col.permute(0, 4, 5, 1, 2, 3, 6).reshape(N * out_h * out_w, -1, num_bits)

    return col, (N, out_h, out_w)

def binary_to_bits_5d(arr, axis=1):
    
    if len(arr.shape) != 3:
        raise ValueError(f"Input tensor must have 3 dimensions, got {arr.shape}")

    N, M, num_bits = arr.shape  # N = N * out_h * out_w, M = C * filter_h * filter_w

    if axis == 1:
        # Expand horizontally: (N, M, bits) -> (N, M * bits)
        result = arr.reshape(N, M * num_bits)
    elif axis == 0:
        # Expand vertically: (N, M, bits) -> (N * bits, M)
        result = arr.permute(0, 2, 1).reshape(N * num_bits, M)
    else:
        raise ValueError("axis must be either 0 (expand vertically) or 1 (expand horizontally)")

    return result

def adc(array, B_ADC, MAC_length):
    return (torch.round((array / MAC_length) * (2**B_ADC - 1))) * (MAC_length / (2**B_ADC - 1))

def conv2d_5d(input, filters, B_ADC, MAC_length, layer, bias=None, stride=1, padding=0):
    global count

    N_F = filters.shape[0]

    # Use im2col_5d to transform input to column form
    col, out_dims = im2col_5d(input, filter_h=filters.shape[2], filter_w=filters.shape[3], stride=stride, padding=padding)
    
    # Flatten the filters to 3D: (num_filters, C * filter_h * filter_w, num_bits)
    filters_col = filters.reshape(filters.shape[0], -1, filters.shape[-1]) 
    
    # Perform binary matrix multiplication
    col_bits = binary_to_bits_5d(col, axis=0)
    filters_bits = binary_to_bits_5d(filters_col, axis=0)

    # Transpose filters_bits to make the second dimension match the first of col_bits
    filters_bits = filters_bits.T

    if count == 0:
        print(f"2D Mult: {col_bits.shape} x {filters_bits.shape}")
        count += 1

    iter = col_bits.shape[1] // MAC_length + 1
    in_X, in_Y = col_bits.shape
    filt_X, filt_Y = filters_bits.shape

    col_accum = torch.zeros((in_X, filt_Y)).to(device)

    for i in range(iter):  
        if i != (iter - 1):
            prod = torch.matmul(col_bits[:, i * MAC_length:(i + 1) * MAC_length].float(), filters_bits[i * MAC_length:(i + 1) * MAC_length, :].float())
            col_accum += adc(prod, B_ADC, MAC_length=MAC_length)
        else:
            prod = torch.matmul(col_bits[:, i * MAC_length:].float(), filters_bits[i * MAC_length:, :].float())
            col_accum += adc(prod, B_ADC, MAC_length=MAC_length)
        
        # if i == 0:
        #     output_subdir = f'OCC_MACValues/conv{layer}'
        #     os.makedirs(output_subdir, exist_ok=True)
        #     output_file = os.path.join(output_subdir, f'MACVALUES_LEN{MAC_length}_LAYER{layer}_ITER{i}.pt')
        #     torch.save(torch.flatten(prod), output_file)
    
    if bias is not None:
        col_accum += bias
    
    return col_accum, (out_dims[0], N_F, out_dims[1], out_dims[2])

def twos_power_mult(tensor_2d, x_bits, w_bits, out_dims):
    global count
    
    matrix = torch.zeros((1 , 1, x_bits, w_bits)).to(device)
    for i in range(x_bits):
        for j in range(w_bits):
            if (i == (0) or j == (0)) and (i != j):
                matrix[0, 0, i, j] = -(2**(x_bits + w_bits - 2 - (i+j)))
            else:
                matrix[0, 0, i, j] = (2**(x_bits + w_bits - 2 - (i+j)))

    N_o, C_o, H_o, W_o = out_dims

    tensor_4d = tensor_2d.unsqueeze(0).unsqueeze(0).float()

    if count == 0:
        print("Starting 2's power multiplication")
    output = F.conv2d(tensor_4d, matrix, stride=(x_bits, w_bits), padding=0)

    output_2d = output.squeeze(0)
    output_3d = output_2d.reshape(N_o, H_o*W_o, C_o)
    output_4d = output_3d.permute(0, 2, 1).reshape(N_o, C_o, H_o, W_o)

    return output_4d

def main():
    parser = argparse.ArgumentParser(description='Quantize tensors and perform convolution.')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to access the corresponding input and weight files.')
    parser.add_argument('--stride', type=int, required=True, help='Convolution stride')
    parser.add_argument('--padding', type=int, required=True, help='Convolution padding')
    parser.add_argument('--x-bits', type=int, required=True, help='Number of bits for input tensor quantization.')
    parser.add_argument('--w-bits', type=int, required=True, help='Number of bits for weights tensor quantization.')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size for processing input tensors.')
    parser.add_argument('--B_ADC', type=int, required=True, help='Number of bits of the ADC/TDC to be used')
    parser.add_argument('--MAC_length', type=int, required=True, help='Column MAC length')

    args = parser.parse_args()

    global x_bits, w_bits
    x_bits = args.x_bits
    w_bits = args.w_bits

    # Construct the file paths using the layer number
    input_file = f'layer_params/conv{args.layer}/inputs.pt'
    weight_file = f'layer_params/conv{args.layer}/weights.pt'

    # Load tensors
    inp_tensor = torch.load(input_file, weights_only=True).to(device)
    weight_tensor = torch.load(weight_file, weights_only=True).to(device)

    print(f"Convolution: {inp_tensor.shape} x {weight_tensor.shape}")

    inp_q, scale_x = quantize_tensor(inp_tensor, x_bits)
    weight_q, scale_w = quantize_tensor(weight_tensor, w_bits)

    inp_5d_bin = quantize_to_binary(inp_q, x_bits)
    weight_5d_bin = quantize_to_binary(weight_q, w_bits)

    # Process in batches
    N = inp_5d_bin.shape[0]
    batch_size = args.batch_size
    outputs = []

    for i in range(0, N, batch_size):
        inp_batch = inp_5d_bin[i:i + batch_size]
        output, out_dims = conv2d_5d(inp_batch, weight_5d_bin, B_ADC=args.B_ADC, MAC_length=args.MAC_length, layer=args.layer, stride=args.stride, padding=args.padding)
        outputs.append(twos_power_mult(output, x_bits, w_bits, out_dims))

    # Concatenate the outputs to get the final output tensor
    conv_out_final = torch.cat(outputs, dim=0) / (scale_x * scale_w).item()
    print(f"Convolution output dims: {conv_out_final.shape}")

    output_subdir = f'conv_out/conv{args.layer}'

    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f'CONV_OUT_LAYER{args.layer}_{args.x_bits}BX_{args.w_bits}BW_{args.B_ADC}BADC_{args.MAC_length}MAC.pt')

    torch.save(conv_out_final, output_file)

if __name__ == '__main__':
    main()
