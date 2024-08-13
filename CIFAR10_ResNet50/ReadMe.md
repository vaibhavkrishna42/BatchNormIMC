### Run the following line in terminal to run the Neural Network inference
`python inference.py --xq 4 --wq 4 --use_device 'CPU' --run_type 'BASELINE'`  
`use_device = 'CPU' or 'GPU'`  
`run_type = 'BASELINE' or 'QUANTISED' (default = 'BASELINE')` 

### Run the following line in terminal to generate the output of the Convolution layer only
`python conv.py --input conv1_input.pth --weights conv1_weights.pth --x-bits 6 --w-bits 9`

### Run the following line in terminal to generate the output of the Convolution + BatchNorm layers
`python batchnorm.py --input_tensor conv1_input.pth --weight_tensor conv1_weights.pth --x_bits 6 --w_bits 9 --Rc Rc_resnet50_cifar10.pth --Tc Tc_resnet50_cifar10.pth`

### Software baseline accuracy (all tensors on CPU) = 89.84375%
