import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ResNet20ExplicitLayers(nn.Module):
    def __init__(self, num_classes=10):
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

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.layer1_bn1(self.layer1_conv1(x)))

        # Layer 1
        shortcut = x
        x = F.relu(self.layer2_bn1(self.layer2_conv1(x)))
        x = self.layer3_bn2(self.layer3_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 2
        shortcut = x
        x = F.relu(self.layer4_bn1(self.layer4_conv1(x)))
        x = self.layer5_bn2(self.layer5_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 3
        shortcut = x
        x = F.relu(self.layer6_bn1(self.layer6_conv1(x)))
        x = self.layer7_bn2(self.layer7_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 4
        shortcut = self.shortcut1_bn(self.shortcut1(x))
        x = F.relu(self.layer8_bn1(self.layer8_conv1(x)))
        x = self.layer9_bn2(self.layer9_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 5
        shortcut = x
        x = F.relu(self.layer10_bn1(self.layer10_conv1(x)))
        x = self.layer11_bn2(self.layer11_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 6
        shortcut = x
        x = F.relu(self.layer12_bn1(self.layer12_conv1(x)))
        x = self.layer13_bn2(self.layer13_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 7
        shortcut = self.shortcut2_bn(self.shortcut2(x))
        x = F.relu(self.layer14_bn1(self.layer14_conv1(x)))
        x = self.layer15_bn2(self.layer15_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 8
        shortcut = x
        x = F.relu(self.layer16_bn1(self.layer16_conv1(x)))
        x = self.layer17_bn2(self.layer17_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 9
        shortcut = x
        x = F.relu(self.layer18_bn1(self.layer18_conv1(x)))
        x = self.layer19_bn2(self.layer19_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Layer 10
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.layer20_linear(x)
        return x

# Define transformations for data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop to add variation
    transforms.RandomHorizontalFlip(),    # Randomly flip images horizontally
    transforms.ToTensor(),                # Convert PIL images to tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 training data
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Load CIFAR-10 test data
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Instantiate and train the ResNet20ExplicitLayers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet20ExplicitLayers().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Training function
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

# Testing function
def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test Loss: {test_loss/len(testloader):.4f}, Accuracy: {100.*correct/total:.2f}%')

# Training loop
for epoch in range(200):
    train(epoch)
    test()
    scheduler.step()

torch.save(net.state_dict(), 'ResNet20_CIFAR10.pth')