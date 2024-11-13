import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import argparse
from contextlib import redirect_stdout

class SuppressOutput:
    def write(self, msg):
        pass

class ResNet20ExplicitLayers(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet20ExplicitLayers, self).__init__()

        # Layer 1
        self.layer1_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn1 = nn.BatchNorm2d(16)

        # Layers 2-3
        self.layer2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_bn1 = nn.BatchNorm2d(16)
        self.layer3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_bn2 = nn.BatchNorm2d(16)

        # Layers 4-5
        self.layer4_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_bn1 = nn.BatchNorm2d(16)
        self.layer5_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5_bn2 = nn.BatchNorm2d(16)

        # Layers 6-7
        self.layer6_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer6_bn1 = nn.BatchNorm2d(16)
        self.layer7_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer7_bn2 = nn.BatchNorm2d(16)

        # Layers 8-9
        self.layer8_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer8_bn1 = nn.BatchNorm2d(32)
        self.shortcut1 = nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False)
        self.shortcut1_bn = nn.BatchNorm2d(32)
        self.layer9_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer9_bn2 = nn.BatchNorm2d(32)

        # Layers 10-11
        self.layer10_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer10_bn1 = nn.BatchNorm2d(32)
        self.layer11_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer11_bn2 = nn.BatchNorm2d(32)

        # Layers 12-13
        self.layer12_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer12_bn1 = nn.BatchNorm2d(32)
        self.layer13_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer13_bn2 = nn.BatchNorm2d(32)

        # Layers 14-15
        self.layer14_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer14_bn1 = nn.BatchNorm2d(64)
        self.shortcut2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)
        self.shortcut2_bn = nn.BatchNorm2d(64)
        self.layer15_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer15_bn2 = nn.BatchNorm2d(64)

        # Layers 16-17
        self.layer16_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer16_bn1 = nn.BatchNorm2d(64)
        self.layer17_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer17_bn2 = nn.BatchNorm2d(64)

        # Layers 18-19
        self.layer18_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer18_bn1 = nn.BatchNorm2d(64)
        self.layer19_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer19_bn2 = nn.BatchNorm2d(64)

        # Layer 20
        self.layer20_linear = nn.Linear(64, num_classes)

    def forward(self, x, layer=None, file=None):
        # Initial convolution
        save_conv_params_and_inputs('conv1', x, self.layer1_conv1.weight, self.layer1_conv1.bias)
        save_batchnorm_params('bn1', self.layer1_bn1)

        if layer == 1 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer1_conv1(x)

        x = self.layer1_bn1(x)
        x = F.relu(x)

        # Layer 1
        shortcut = x
        save_conv_params_and_inputs('conv2', x, self.layer2_conv1.weight, self.layer2_conv1.bias)
        save_batchnorm_params('bn2', self.layer2_bn1)

        if layer == 2 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer2_conv1(x)

        x = self.layer2_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv3', x, self.layer3_conv2.weight, self.layer3_conv2.bias)
        save_batchnorm_params('bn3', self.layer3_bn2)

        if layer == 3 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer3_conv2(x)

        x = self.layer3_bn2(x)
        x += shortcut
        x = F.relu(x)

        # Layer 2
        shortcut = x
        save_conv_params_and_inputs('conv4', x, self.layer4_conv1.weight, self.layer4_conv1.bias)
        save_batchnorm_params('bn4', self.layer4_bn1)

        if layer == 4 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer4_conv1(x)

        x = self.layer4_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv5', x, self.layer5_conv2.weight, self.layer5_conv2.bias)
        save_batchnorm_params('bn5', self.layer5_bn2)

        if layer == 5 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer5_conv2(x)

        x = self.layer5_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 3
        shortcut = x
        save_conv_params_and_inputs('conv6', x, self.layer6_conv1.weight, self.layer6_conv1.bias)
        save_batchnorm_params('bn6', self.layer6_bn1)

        if layer == 6 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer6_conv1(x)

        x = self.layer6_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv7', x, self.layer7_conv2.weight, self.layer7_conv2.bias)
        save_batchnorm_params('bn7', self.layer7_bn2)

        if layer == 7 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer7_conv2(x)

        x = self.layer7_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 4
        shortcut = self.shortcut1_bn(self.shortcut1(x))
        save_conv_params_and_inputs('conv8', x, self.layer8_conv1.weight, self.layer8_conv1.bias)
        save_batchnorm_params('bn8', self.layer8_bn1)

        if layer == 8 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer8_conv1(x)

        x = self.layer8_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv9', x, self.layer9_conv2.weight, self.layer9_conv2.bias)
        save_batchnorm_params('bn9', self.layer9_bn2)

        if layer == 9 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer9_conv2(x)

        x = self.layer9_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 5
        shortcut = x
        save_conv_params_and_inputs('conv10', x, self.layer10_conv1.weight, self.layer10_conv1.bias)
        save_batchnorm_params('bn10', self.layer10_bn1)

        if layer == 10 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer10_conv1(x)

        x = self.layer10_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv11', x, self.layer11_conv2.weight, self.layer11_conv2.bias)
        save_batchnorm_params('bn11', self.layer11_bn2)

        if layer == 11 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer11_conv2(x)

        x = self.layer11_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 6
        shortcut = x
        save_conv_params_and_inputs('conv12', x, self.layer12_conv1.weight, self.layer12_conv1.bias)
        save_batchnorm_params('bn12', self.layer12_bn1)

        if layer == 12 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer12_conv1(x)

        x = self.layer12_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv13', x, self.layer13_conv2.weight, self.layer13_conv2.bias)
        save_batchnorm_params('bn13', self.layer13_bn2)

        if layer == 13 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer13_conv2(x)

        x = self.layer13_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 7
        shortcut = self.shortcut2_bn(self.shortcut2(x))
        save_conv_params_and_inputs('conv14', x, self.layer14_conv1.weight, self.layer14_conv1.bias)
        save_batchnorm_params('bn14', self.layer14_bn1)

        if layer == 14 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer14_conv1(x)

        x = self.layer14_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv15', x, self.layer15_conv2.weight, self.layer15_conv2.bias)
        save_batchnorm_params('bn15', self.layer15_bn2)

        if layer == 15 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer15_conv2(x)

        x = self.layer15_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 8
        shortcut = x
        save_conv_params_and_inputs('conv16', x, self.layer16_conv1.weight, self.layer16_conv1.bias)
        save_batchnorm_params('bn16', self.layer16_bn1)

        if layer == 16 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer16_conv1(x)

        x = self.layer16_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv17', x, self.layer17_conv2.weight, self.layer17_conv2.bias)
        save_batchnorm_params('bn17', self.layer17_bn2)

        if layer == 17 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer17_conv2(x)

        x = self.layer17_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 9
        shortcut = x
        save_conv_params_and_inputs('conv18', x, self.layer18_conv1.weight, self.layer18_conv1.bias)
        save_batchnorm_params('bn18', self.layer18_bn1)
        
        if layer == 18 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer18_conv1(x)

        x = self.layer18_bn1(x)
        x = F.relu(x)

        save_conv_params_and_inputs('conv19', x, self.layer19_conv2.weight, self.layer19_conv2.bias)
        save_batchnorm_params('bn19', self.layer19_bn2)
        
        if layer == 19 and file:
            x = torch.load(file, weights_only=True, map_location=x.device)
        else:
            x = self.layer19_conv2(x)

        x = self.layer19_bn2(x)

        x += shortcut
        x = F.relu(x)

        # Layer 10
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.layer20_linear(x)
        return x

save = False
def save_conv_params_and_inputs(layer_name, inputs, weights, bias=None, save_dir='./layer_params', save_data=save):
    if not save_data:
        return  # Do nothing if saving is disabled
    
    # Create a directory for the layer
    layer_dir = os.path.join(save_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)
    
    # Save inputs, weights, and bias (if applicable)
    torch.save(inputs, os.path.join(layer_dir, 'inputs.pt'))
    torch.save(weights, os.path.join(layer_dir, 'weights.pt'))
    
    if bias is not None:
        torch.save(bias, os.path.join(layer_dir, 'bias.pt'))

def save_batchnorm_params(layer_name, bn_layer, save_dir='./layer_params', save_data=save):
    if not save_data:
        return  # Do nothing if saving is disabled
    
    # Create a directory for the layer
    layer_dir = os.path.join(save_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)
    
    # Save batch norm parameters
    torch.save(bn_layer.weight, os.path.join(layer_dir, 'weight.pt'))
    torch.save(bn_layer.bias, os.path.join(layer_dir, 'bias.pt'))
    torch.save(bn_layer.running_mean, os.path.join(layer_dir, 'running_mean.pt'))
    torch.save(bn_layer.running_var, os.path.join(layer_dir, 'running_var.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet20 CIFAR100 Evaluation')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to test')
    parser.add_argument('--layer', type=int, default=None, help='Layer to load the output from')
    parser.add_argument('--file', type=str, default=None, help='File to load the layer output from')
    args = parser.parse_args()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-100 normalization
    ])

    with redirect_stdout(SuppressOutput()):
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2) # BATCH SIZE CAREFUL

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize and load the model's state_dict
    model = ResNet20ExplicitLayers().to(device)
    model.load_state_dict(torch.load('ResNet20_CIFAR100.pth', weights_only=True, map_location=device))

    # Function to evaluate the model
    def test(num_images=None):
        model.eval()
        correct = 0
        total = 0
        processed = 0  # Counter to track number of processed images

        with torch.no_grad():
            for inputs, targets in testloader:
                if num_images and processed >= num_images:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, layer=args.layer, file=args.file)
                _, predicted = outputs.max(1)
                batch_size = targets.size(0)
                
                # Adjust for cases where num_images is smaller than batch size
                if num_images:
                    batch_size = min(batch_size, num_images - processed)
                    inputs, targets, predicted = inputs[:batch_size], targets[:batch_size], predicted[:batch_size]

                total += batch_size
                correct += predicted.eq(targets).sum().item()
                processed += batch_size

        print(f'{100.*correct/total:.2f}')

    # Run the test function
    test(num_images=args.num_images)