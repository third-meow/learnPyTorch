import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# The neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Setup layers
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load dataset
clear_edges = pickle.load(open('data/clear_edges_data.pickle', 'rb'))

# Format data
xlearn = [torch.tensor(shape_label_pair[0], dtype=torch.float)
          for shape_label_pair in clear_edges]
ylearn = [torch.tensor([shape_label_pair[1]], dtype=torch.float)
          for shape_label_pair in clear_edges]


# Don't let len(xlearn) not equal len(ylearn)
assert (len(xlearn) == len(ylearn)), "no. labels != no. data"

# Create neural net, loss function and optimizer
net = Net()
mse = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.004)

for p in range(20):
    total_loss = 0
    for i in range(50):
        # Reshape input
        inp = xlearn[i % len(xlearn)].view(4)

        # Run input through neural network
        out = net(inp)

        # Calculate loss
        target = ylearn[i % len(ylearn)]
        loss = mse(out, target)

        # Keep track of total loss
        total_loss += loss

        # Backprop network
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate and print average loss
    print(total_loss/50)
