### Run the following line in terminal to generate the output of the Convolution layer only
`python conv.py --input conv1_input.pth --weights conv1_weights.pth --x-bits 6 --w-bits 9`

### Run the following line in terminal to generate the output of the Convolution + BatchNorm layers
`python batchnorm.py --input_tensor conv1_input.pth --weight_tensor conv1_weights.pth --x_bits 6 --w_bits 9 --Rc Rc_resnet50_cifar10.pth --Tc Tc_resnet50_cifar10.pth`
