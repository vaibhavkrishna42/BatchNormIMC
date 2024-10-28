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

# Convolution without Clipping
echo "Convolution output generation with MAC Length = ${MAC_LENGTH} and B_ADC = ${B_ADC}"
for LAYER in {1..8}
do
    echo "Running layer $LAYER..."
    python bitwise_conv_batchwise_maclength.py --layer $LAYER --stride $STRIDE --padding $PADDING --x-bits $X_BITS --w-bits $W_BITS --batch-size $BATCH_SIZE --B_ADC $B_ADC --MAC_length $MAC_LENGTH
done

echo "All convolution outputs without Clipping generated!"

# Convolution with Clipping
echo "Convolution output generation with MAC Length = ${MAC_LENGTH} and Clipping"
for LAYER in {1..8}
do
    echo "Running layer $LAYER..."
    CLIP=${CLIP_POINTS[LAYER-1]}  # Arrays are zero-indexed
    echo "Clip Point = $CLIP"
    python bitwise_conv_batchwise_maclength_clip.py --layer $LAYER --stride $STRIDE --padding $PADDING --x-bits $X_BITS --w-bits $W_BITS --batch-size $BATCH_SIZE --B_ADC $B_ADC --MAC_length $MAC_LENGTH --clip $CLIP
done

echo "All convolution outputs after Clipping generated!"

# Inference without Clipping
for LAYER in {1..8}
do
    echo "Running inference using modified layer $LAYER..."
    FILE_IN=conv_out\\conv${LAYER}\\CONV_OUT_LAYER${LAYER}_${X_BITS}BX_${W_BITS}BW_${B_ADC}BADC_${MAC_LENGTH}MAC.pt
    python inference_vgg11_cifar10.py --num_images 1000 --layer $LAYER --file "$FILE_IN"
done

echo "Pre-Clipping accuracies found!"

echo "Running inferences on the Clipped MACs"
for LAYER in {1..8}
do
    echo "Running inference using modified layer $LAYER..."
    CLIP=${CLIP_POINTS[LAYER-1]}
    FILE_IN=conv_out\\conv${LAYER}\\CONV_OUT_LAYER${LAYER}_6BX_9BW_5BADC_256MAC_${CLIP}CLIP.pt
    python inference_vgg11_cifar10.py --num_images 1000 --layer $LAYER --file "$FILE_IN"
done

echo "All layers processed!"


# First run: chmod +x layerwise_imc_conv.sh

# Then run: ./layerwise_imc_conv.sh
