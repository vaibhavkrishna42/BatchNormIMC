### Running bit-wise convolution without output quantization  
`python bitwise_conv_batchwise.py --layer 4 --stride 1 --padding 1 --x-bits 6 --w-bits 9 --batch-size 64`  
Output file name should be of the kind : `conv_out/conv1/CONV_OUT_LAYER1_6BX_9BW.pt`  

### Running bit-wise convolution with control over MAC Length and ADC bits (no clipping)  
`python bitwise_conv_batchwise_maclength.py --layer 1 --stride 1 --padding 1 --x-bits 6 --w-bits 9 --batch-size 64 --B_ADC 4 --MAC_length 256`  
Output file name should be of the kind : `conv_out/conv1/CONV_OUT_LAYER1_6BX_9BW_4BADC_256MAC.pt`  

### Running bit-wise convolution employing the Optimal Clipping Criterion (OCC)  
`python bitwise_conv_batchwise_maclength_clip.py --layer 1 --stride 1 --padding 1 --x-bits 6 --w-bits 9 --batch-size 64 --B_ADC 4 --MAC_length 256 --clip 20`  
Output file name should be of the kind : `conv_out/conv1/CONV_OUT_LAYER1_6BX_9BW_4BADC_256MAC_20CLIP.pt`    

### Running inference on the test dataset (CIFAR10)  
`python inference_vgg11_cifar10.py --num_images 1000 --layer 1 --file 'conv_out\conv1\CONV_OUT_LAYER1_6BX_9BW_4BADC_256MAC_20CLIP.pt'`  
Choose a layer and a CONV_OUT file to run the inference with.  

### Finding clipping points for OCC
`python occ.py --file "OCC_MACValues/conv1/MACVALUES_LEN256_LAYER1_ITER0.pt" --xr_start 10 --xr_end 256 --B_start 3 --B_end 5 --layer 1`  
Choose a MAC tensor (from the OCC_MACValues folder), range of clipping point to iterate through, and range of ADC bits to iterate through.
