import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms


class MovingAverage():
    def __init__(self):
        self.total = 0
        self.avg = 0
        self.count = 0

    # Update avg to reflect total and count
    def calc(self):
        self.avg = self.total / self.count

    # Incorperate number into average
    def incorp(self, n):
        self.total += n
        self.count += 1
        self.calc()

    # Reset average to 0
    def reset(self):
        self.__init__()


# Our convulutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Setup layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Runs network on input
    def forward(self, x):
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten
        x = x.view(-1, 256)
        # Linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Display image, for debugging
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    # Setup data transorm, used to 'format' dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load CIFAR10 training set
    training_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    # Load CIFAR10 testing set
    testing_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        testing_set,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    # Create network, loss functino and optimizer
    net = Net()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    for epoch in range(2):
        # Create moving average to track average loss
        avg_loss = MovingAverage()

        # Iterate through all training data
        for i, data in enumerate(train_loader):

            # Split data into input and label
            x, y = data

            # Run input through network
            optimizer.zero_grad()
            out = net(x)

            # Calculate loss and backprop
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()

            # Keep track of loss
            avg_loss.incorp(loss.item())

            # Ever 1000 mini-batchs, print average loss
            if i % 1000 == 0 and i != 0:
                print(f'{epoch} | {i}\tLoss: {avg_loss.avg:.3f}')
                avg_loss.reset()

    # Test accuracy
    accuracy = MovingAverage()
    for data in test_loader:
        x, y = data
        _, predictions = torch.max(net(x), 1)
        for i in range(4):
            if predictions[i] == y[i]:
                accuracy.incorp(1)
            else:
                accuracy.incorp(0)

    # Report accuracy
    print(f'Accuracy: {accuracy.avg * 100}%')


if __name__ == '__main__':
    main()
