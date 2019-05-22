import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Learn dataset
xlearn = [torch.tensor([0., 1.]), torch.tensor([1., 1.]), torch.tensor([0., 0.])]
ylearn = [torch.tensor([1.]), torch.tensor([0.]), torch.tensor([0.])]

# Don't let len(xlearn) not equal len(ylearn)
assert (len(xlearn) == len(ylearn)), "no. labels != no. data"

net = Net()
mse = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(30):
    inp = xlearn[i % len(xlearn)]
    out = net(inp)
    target = ylearn[i % len(ylearn)]

    loss = mse(out, target)

    print(loss)

    net.zero_grad()
    loss.backward()

    optimizer.step()
