"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Define your PyTorch Network (adjust input dimensions to match your CSV features if needed)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Example linear layers for tabular data - adjust input features count accordingly
        self.fc1 = nn.Linear(78, 128)  # Change 78 to match len(features)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Change output classes as needed

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs, lr, device):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    epoch_loss = 0.0
    for _ in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)

    return epoch_loss


def test(net, testloader, device):
    """Validate the network on the validation set."""
    criterion = nn.CrossEntropyLoss()
    net.eval()

    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            outputs = net(features)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    val_loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    return val_loss, accuracy
