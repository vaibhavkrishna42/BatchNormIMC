import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from unittest.mock import patch

def parse_arguments():
    parser = argparse.ArgumentParser(description="ResNet with quantization")
    parser.add_argument('--xq', type=int, help='Quantization bits for input tensors (default: 3)')
    parser.add_argument('--wq', type=int, help='Quantization bits for weights (default: 4)')
    parser.add_argument('--use_device', type=str, default='GPU', help='Choose whether to run on CPU or GPU')
    parser.add_argument('--run_type', type=str, default="BASELINE", help='Choose between quantised/unquantised run')
    return parser.parse_args()

def main(xq, wq, use_device, run_type):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with patch('builtins.print'):
        train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=2)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    ##########################################################################################################################
    # Finding Rc and Tc
    def load_bn_values(filename):
        bn_values = {
            'bn_weight': [],
            'bn_bias': [],
            'bn_running_mean': [],
            'bn_running_var': []
        }
        
        with open(filename, 'r') as file:
            lines = file.readlines()
            current_key = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('bn_'):
                    current_key = line.split(':')[0].strip()
                    values_str = line.split(':')[1].strip()[1:-1]  # Remove the square brackets
                    values = [float(value.strip()) for value in values_str.split(',')]
                    bn_values[current_key] = values

        return bn_values

    filename = 'bn_params_stats.txt'
    bn_values = load_bn_values(filename)
    # print(len(bn_values['bn_weight']))

    Rc_stats = []
    Tc_stats = []
    for i in range(64):
        Rc_stats.append(bn_values['bn_running_mean'][i] - (bn_values['bn_bias'][i] * np.sqrt(bn_values['bn_running_var'][i] + 1e-5))/(bn_values['bn_weight'][i]))
        Tc_stats.append((bn_values['bn_weight'][i])/np.sqrt(bn_values['bn_running_var'][i] + 1e-5))

    if use_device == 'GPU':
        Rc = (torch.tensor(Rc_stats)).to(torch.float32).to('cuda')
        Tc = (torch.tensor(Tc_stats)).to(torch.float32).to('cuda')

    elif use_device == 'CPU':
        Rc = (torch.tensor(Rc_stats)).to(torch.float32)
        Tc = (torch.tensor(Tc_stats)).to(torch.float32)

    else:
        print("INVALID DEVICE")

    torch.save(Rc,'Rc_resnet50_cifar10.pth')
    torch.save(Tc,'Tc_resnet50_cifar10.pth')
    const_tensor = 1
    ##########################################################################################################################

    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
            super(Bottleneck, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.batch_norm1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.batch_norm2 = nn.BatchNorm2d(out_channels)
            
            self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
            self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
            
            self.i_downsample = i_downsample
            self.stride = stride
            self.relu = nn.ReLU()
            
        def forward(self, x):
            identity = x.clone()
            x = self.relu(self.batch_norm1(self.conv1(x)))
            
            x = self.relu(self.batch_norm2(self.conv2(x)))
            
            x = self.conv3(x)
            x = self.batch_norm3(x)
            
            #downsample if needed
            if self.i_downsample is not None:
                identity = self.i_downsample(identity)
            #add identity
            x += identity
            x = self.relu(x)
            
            return x

    class Block(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
            super(Block, self).__init__()
        

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
            self.batch_norm1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
            self.batch_norm2 = nn.BatchNorm2d(out_channels)

            self.i_downsample = i_downsample
            self.stride = stride
            self.relu = nn.ReLU()

        def forward(self, x):
            identity = x.clone()

            x = self.relu(self.batch_norm2(self.conv1(x)))
            x = self.batch_norm2(self.conv2(x))

            if self.i_downsample is not None:
                identity = self.i_downsample(identity)
            print(x.shape)
            print(identity.shape)
            x += identity
            x = self.relu(x)
            return x

    class ResNet(nn.Module):

        instance_count = 0
        forward_call_count = 0

        def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
            
            ResNet.instance_count += 1
            self.instance_number = ResNet.instance_count

            super(ResNet, self).__init__()
            self.in_channels = 64
            
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.batch_norm1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
            
            self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
            self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
            self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
            self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
            
        def forward(self, x):

            ResNet.forward_call_count += 1
            
            s = self.conv1.weight
            if use_device == 'CPU':
                x = x.cpu()
                s = s.cpu()

            # Normal forward pass
            if run_type == 'BASELINE':
                x = F.conv2d(x, s, None, (2,2), (3,3), (1,1), 1)
                x = self.batch_norm1(x)
            
            # Quantized inputs and weights Convolution + BatchNorm (No change to Rc and s is quantized on its own)
            elif run_type == 'QUANTISED':
                s, scale_s = self.quantize_tensor(self.conv1.weight, wq, 'signed')
                x, scale_x = self.quantize_tensor(x, xq, 'signed')
                x = F.conv2d(x, s, None, (2,2), (3,3), (1,1), 1)
                x = x/(scale_x*scale_s).item()
                x = (x - Rc[None, :, None, None])*Tc[None, :, None, None]

            # torch.save(x, 'conv1_input.pth')
            # torch.save(self.conv1.weight, 'conv1_weights.pth')

            # Convolution + Batchnorm using a constant for subtraction
            '''
            # Modified weights convolution
            # s = (self.conv1.weight)*(const_tensor/Rc[:,None,None,None])
            # s = s.clamp(min=-250,max=250)
            # s = self.quantize_tensor(s, 9, mode='signed')
            # x = self.quantize_tensor(x, 4, mode='signed')
            # x = F.conv2d(x.to('cuda'), s.to('cuda'), None, (2,2), (3,3), (1,1), 1)

            # Modified weights Batch Norm
            # x = (x - const_tensor)*((Rc[None,:,None,None])*Tc[None,:,None,None])/const_tensor
            '''

            # x = (torch.load(r'BN_Out_Saved\bn1_out_x6_w9')).to(dtype = torch.float32)

            x = self.relu(x)
            x = self.max_pool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)
            
            # Save parameters for the first instance
            # self.save_bn_params_to_file('bn_params_stats.txt')

            return x

        def get_bn_params(self):
            return {
                'bn_weight': self.batch_norm1.weight.data.tolist(),
                'bn_bias': self.batch_norm1.bias.data.tolist(),
                'bn_running_mean': self.batch_norm1.running_mean.data.tolist(),
                'bn_running_var': self.batch_norm1.running_var.data.tolist()
            }

        def save_bn_params_to_file(self, filename):
            bn_params = self.get_bn_params()
            with open(filename, 'w') as file:
                for key, values in bn_params.items():
                    file.write(f'{key}: {values}\n')

        def _make_layer(self, ResBlock, blocks, planes, stride=1):
            ii_downsample = None
            layers = []
            
            if stride != 1 or self.in_channels != planes * ResBlock.expansion:
                ii_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes*ResBlock.expansion))
            layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
            self.in_channels = planes * ResBlock.expansion
            
            for i in range(blocks-1):
                layers.append(ResBlock(self.in_channels, planes))
            return nn.Sequential(*layers)

        def quantize_tensor(self, tensor, num_bits, mode='signed'):
            if mode == 'unsigned':
                qmin = 0
                qmax = 2**num_bits - 1
            else:
                qmin = -2**(num_bits - 1)
                qmax = 2**(num_bits - 1) - 1

            if tensor.abs().max() > tensor.max():
                scale = (-qmin)/tensor.abs().max()
            else:
                scale = (qmax)/tensor.max()
            
            if mode == 'unsigned':
                tensor_q = (tensor*scale).round().clamp(min=0, max=2**(num_bits)-1).to(torch.int32)
            else:
                tensor_q = (tensor*scale).round().clamp(min=-2**(num_bits-1), max=2**(num_bits-1)-1).to(torch.int32)

            return tensor_q, scale


    def ResNet50(num_classes, channels=3):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)

    def test_resnet(xq, wq, use_device, run_type):
        # Define your model architecture (assuming it's defined as 'Net' class)
        if use_device == 'GPU':
            net = ResNet50(10).to('cuda')
        elif use_device == 'CPU':
            net = ResNet50(10)
        else:
            print('INVALID DEVICE')

        # Load the checkpoint file
        checkpoint = torch.load('final_checkpoint.pth')

        # Load the model state from the checkpoint
        net.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        net.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data

                if use_device == 'GPU':
                    images, labels = images.to('cuda'), labels.to('cuda')
                elif use_device == 'CPU':
                    images, labels = images, labels
                else:
                    print("INVALID DEVICE")
                outputs = net(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                break

        # print('Accuracy on 10,000 test images: ', 100 * (correct / total), '%')
        acc = 100*(correct/total)
        # print(xq,wq,acc,loss_acc,"\n")
        print(acc)

    test_resnet(xq, wq, use_device, run_type)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.xq, args.wq, args.use_device, args.run_type)
