import torch
import torch.nn as nn
import torch.nn.functional as F


conv = nn.Conv1d(4, 2, 2)
maxpool = nn.MaxPool1d(2, 2)

x = torch.rand(1, 4, 4)
print(x, end='\n\n')
x = conv(x)
print(x, end='\n\n')
x = F.relu(x)
print(x, end='\n\n')
x = maxpool(x)
print(x, end='\n\n')

conv = nn.Conv1d(4, 4, 2)
maxpool = nn.MaxPool1d(2, 2)


x = torch.Tensor(
    [[[0., 1., 1., 0.],
      [0., 0., 0., 0.],
      [1., 0., 0., 1.],
      [0., 1., 1., 0.]]]
)
print(x, end='\n\n')
x = conv(x)
print(x, end='\n\n')
x = F.relu(x)
print(x, end='\n\n')
x = maxpool(x)
print(x, end='\n\n')

print(
