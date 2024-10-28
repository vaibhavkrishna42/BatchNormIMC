#!/bin/bash

# Define the common arguments
X_BITS=6
W_BITS=9
STRIDE=1
PADDING=1
BATCH_SIZE=64
B_ADC=5
MAC_LENGTH=256

CLIP_POINTS=(19 64 54 36 49 32 31 20)

# Run the python script for each layer from 1 to 9
for LAYER in {1..8}
do
    echo "Running layer $LAYER..."
    CLIP=${CLIP_POINTS[LAYER-1]}  # Arrays are zero-indexed
    echo "Clip Point = $CLIP"
    python bitwise_conv_batchwise_maclength_clip.py --layer $LAYER --stride $STRIDE --padding $PADDING --x-bits $X_BITS --w-bits $W_BITS --batch-size $BATCH_SIZE --B_ADC $B_ADC --MAC_length $MAC_LENGTH --clip $CLIP
done

echo "All layers processed!"

# First run: chmod +x clip_all_layers.sh

# Then run: ./clip_all_layers.sh
