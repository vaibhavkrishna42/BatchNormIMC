import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Define VGG11 model with Batch Normalization and Dropout
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Norm

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Norm

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch Norm

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Batch Norm

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)  # Batch Norm

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)  # Batch Norm

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)  # Batch Norm

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)  # Batch Norm

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(512, 4096)
        self.dropout1 = nn.Dropout(0.5)  # Dropout
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)  # Dropout
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Conv1, BN1, ReLU, and MaxPool
        save_conv_params_and_inputs('conv1', x, self.conv1.weight, self.conv1.bias)
        save_batchnorm_params('bn1', self.bn1)

        x = self.conv1(x)
        # x = torch.load(r'conv_out\conv1\CONV_OUT_LAYER1_6BX_9BW.pt')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Conv2, BN2, ReLU, and MaxPool
        save_conv_params_and_inputs('conv2', x, self.conv2.weight, self.conv2.bias)
        save_batchnorm_params('bn2', self.bn2)

        x = self.conv2(x) 
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Conv3, BN3, ReLU
        save_conv_params_and_inputs('conv3', x, self.conv3.weight, self.conv3.bias)
        save_batchnorm_params('bn3', self.bn3)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Conv4, BN4, ReLU, and MaxPool
        save_conv_params_and_inputs('conv4', x, self.conv4.weight, self.conv4.bias)
        save_batchnorm_params('bn4', self.bn4)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Conv5, BN5, ReLU
        save_conv_params_and_inputs('conv5', x, self.conv5.weight, self.conv5.bias)
        save_batchnorm_params('bn5', self.bn5)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Conv6, BN6, ReLU, and MaxPool
        save_conv_params_and_inputs('conv6', x, self.conv6.weight, self.conv6.bias)
        save_batchnorm_params('bn6', self.bn6)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Conv7, BN7, ReLU
        save_conv_params_and_inputs('conv7', x, self.conv7.weight, self.conv7.bias)
        save_batchnorm_params('bn7', self.bn7)

        x = self.conv7(x)      
        x = self.bn7(x)
        x = self.relu(x)

        # Conv8, BN8, ReLU, and MaxPool
        save_conv_params_and_inputs('conv8', x, self.conv8.weight, self.conv8.bias)
        save_batchnorm_params('bn8', self.bn8)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with Dropout
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

save = True

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

# The main function for the test script
if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 normalization
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Initialize and load the model's state_dict
    model = VGG11().to(device)
    model.load_state_dict(torch.load('model_state_dict_VGG11_CIFAR10.pth', weights_only=True))

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
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                batch_size = targets.size(0)
                
                # Adjust for cases where num_images is smaller than batch size
                if num_images:
                    batch_size = min(batch_size, num_images - processed)
                    inputs, targets, predicted = inputs[:batch_size], targets[:batch_size], predicted[:batch_size]

                total += batch_size
                correct += predicted.eq(targets).sum().item()
                processed += batch_size

        print(f'Test Accuracy for {total} images: {100.*correct/total:.2f}%')


    # Run the test function
    test(num_images=1000)