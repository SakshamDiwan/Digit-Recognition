# Importing all the necessary libraries
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Defining the model
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(784, 392)
        self.linear_layer2 = nn.Linear(392, 196)
        self.linear_layer3 = nn.Linear(196, 98)
        self.linear_layer4 = nn.Linear(98, 49)
        self.linear_layer5 = nn.Linear(49, 10)

    def forward(self, x):  # 5 linear layers with relu activation function
        x = x.view(-1, 784)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        x = F.relu(self.linear_layer3(x))
        x = F.relu(self.linear_layer4(x))
        x = F.relu(self.linear_layer5(x))
        return x


model = FFNN()

# Training Dataset and Dataloader
train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Testing Dataset and Dataloader
test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)


# Cross Entropy Loss (Loss Function)
cross_entropy_loss = nn.CrossEntropyLoss()

# Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50


# Function to train model and back propagate for one epoch
def train_epoch(model, optimizer, dataloader):
    model.train()
    losses = []
    num_correct = 0
    for input, labels in dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy_loss(output, torch.squeeze(labels).float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        num_correct += (
            torch.argmax(output, dim=1) == torch.argmax(torch.squeeze(labels), dim=1)
        ).sum()
    print("End of epoch accuracy", float(num_correct / 60000))
    return losses


# Function to test model for one epoch
def test_epoch(model, dataloader):
    model.eval()
    losses = []
    num_correct = 0
    with torch.no_grad():
        for input, labels in dataloader:
            output = model(input)
            loss = cross_entropy_loss(output, torch.squeeze(labels).float())
            losses.append(loss.item())
            num_correct += (
                torch.argmax(output, dim=1)
                == torch.argmax(torch.squeeze(labels), dim=1)
            ).sum()
    print("End of validation epoch accuracy", float(num_correct / 10000))
    return losses


# Lists to store the losses for plotting
training_epoch_losses = []
testing_epoch_losses = []

# Training and testing for {num_epochs} epochs
for epoch in range(num_epochs):
    training_epoch_losses.append(train_epoch(model, optimizer, train_dataloader))
    testing_epoch_losses.append(test_epoch(model, test_dataloader))


# Plot of the average loss per epoch vs number of epochs
plt.plot(range(num_epochs), [np.mean(x) for x in training_epoch_losses])
plt.plot(range(num_epochs), [np.mean(x) for x in testing_epoch_losses])
plt.legend(["training", "validation"], loc="upper right")
plt.xlabel("Number of epochs")
plt.ylabel("Average loss per epoch")
plt.title("Loss per epoch")
plt.show()
