### Run the following line in terminal to generate output of Convolution + BatchNorm layer
python batchnorm.py --input_tensor conv1_input.pth --weight_tensor conv1_weights.pth --x_bits 6 --w_bits 9 --Rc Rc_resnet50_cifar10.pth --Tc Tc_resnet50_cifar10.pth
