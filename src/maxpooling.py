import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = torch.Tensor(
    [[[0., 1., 1., 0.],
      [0., 0., 0., 0.],
      [1., 0., 0., 1.],
      [0., 1., 1., 0.]]]
)

conv = nn.Conv1d(4, 4, 2, stride=2)
maxpool = nn.MaxPool1d(2, 2)

x = BASE
print(x, end='\n\n')
x = conv(x)
print(x, end='\n\n')
x = F.relu(x)
print(x, end='\n\n')
x = maxpool(x)
print(x, end='\n\n')

print()
print()
print()

conv = nn.Conv1d(4, 4, 2, stride=1)
maxpool = nn.MaxPool1d(2, 2)

x = BASE
print(x, end='\n\n')
x = conv(x)
print(x, end='\n\n')
x = F.relu(x)
print(x, end='\n\n')
x = maxpool(x)
print(x, end='\n\n')
