#!/bin/bash

# Define the common arguments
X_BITS=6
W_BITS=9
STRIDE=1
PADDING=1
BATCH_SIZE=64
B_ADC=5
MAC_LENGTH=256

# Run the python script for each layer from 1 to 9
echo "Convolution output generation with MAC Length = ${MAC_LENGTH} and B_ADC = ${B_ADC}"
for LAYER in {1..8}
do
    echo "Running layer $LAYER..."
    python bitwise_conv_batchwise_maclength.py --layer $LAYER --stride $STRIDE --padding $PADDING --x-bits $X_BITS --w-bits $W_BITS --batch-size $BATCH_SIZE --B_ADC $B_ADC --MAC_length $MAC_LENGTH
done

echo "All layers processed!"

# First run: chmod +x conv_out_all_layers.sh

# Then run: ./conv_out_all_layers.sh
