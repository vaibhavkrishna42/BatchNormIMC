#!/bin/bash

# Define the common arguments
CLIP_POINTS=(19 64 54 36 49 32 31 20)

# Run the python script for each layer from 1 to 9
for LAYER in {1..8}
do
    echo "Running inference using modified layer $LAYER..."
    CLIP=${CLIP_POINTS[LAYER-1]}
    FILE_IN=conv_out\\conv${LAYER}\\CONV_OUT_LAYER${LAYER}_6BX_9BW_5BADC_256MAC_${CLIP}CLIP.pt
    python inference_vgg11_cifar10.py --num_images 1000 --layer $LAYER --file "$FILE_IN"
done

echo "All layers processed!"

# First run: chmod +x inference_post_clip.sh

# Then run: ./inference_post_clip.sh
