#!/bin/bash

# Define the common arguments
X_BITS=6
W_BITS=8
STRIDE=1
PADDING=1
BATCH_SIZE=64
B_ADC=4
MAC_LENGTH=256

CLIP_POINTS=(17 47 37 46 33 42 29 45 40 58 28 63 24 65 39 42 26 42 31)

# Run the python script for each layer from 1 to 9
for LAYER in {1..19}
do
    # echo "Running inference using modified layer $LAYER..."
    CLIP=${CLIP_POINTS[LAYER-1]}
    FILE_IN=conv_out\\conv${LAYER}\\CONV_OUT_LAYER${LAYER}_${X_BITS}BX_${W_BITS}BW_${B_ADC}BADC_${MAC_LENGTH}MAC_${CLIP}CLIP.pt
    python inference_resnet20_cifar100.py --num_images 1000 --layer $LAYER --file "$FILE_IN"
done

echo "All layers processed!"

# First run: chmod +x inference_post_clip.sh

# Then run: ./inference_post_clip.sh
