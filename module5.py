import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Basic Principles of Deep Learning
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Training a Deep Neural Network (DNN)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
# Use a subset of the dataset for faster training
trainset, _ = train_test_split(trainset, test_size=0.95, random_state=42)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

net = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the neural network
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

print("Finished Training")

# Training a Convolutional Neural Network (CNN)
transform_cifar = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset_cifar = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_cifar
)
# Use a subset of the dataset for faster training
trainset_cifar, _ = train_test_split(trainset_cifar, test_size=0.95, random_state=42)
trainloader_cifar = DataLoader(trainset_cifar, batch_size=64, shuffle=True)

cnn = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# Train the CNN with reduced epochs
for epoch in range(5):
    running_loss_cnn = 0.0
    for i, data in enumerate(trainloader_cifar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_cnn.zero_grad()
        outputs = cnn(inputs)
        loss = criterion_cnn(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss_cnn / len(trainloader_cifar)}")

print("Finished Training CNN")
