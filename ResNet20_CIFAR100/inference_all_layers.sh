#!/bin/bash

# Define the common arguments

# Run the python script for each layer from 1 to 9
for LAYER in {1..8}
do
    echo "Running inference using modified layer $LAYER..."
    FILE_IN=conv_out\\conv${LAYER}\\CONV_OUT_LAYER${LAYER}_6BX_9BW_5BADC_256MAC.pt
    python inference_vgg11_cifar10.py --num_images 1000 --layer $LAYER --file "$FILE_IN"
done

echo "All layers processed!"

# First run: chmod +x inference_all_layers.sh

# Then run: ./inference_all_layers.sh
