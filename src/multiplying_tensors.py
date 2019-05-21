import torch
import torchvision

BAR = "~"*80

def display(title, thing):
    print(title)
    print('-' * (len(title) + 4))
    print(thing)
    print(BAR)

twos = torch.ones(3) + 1
rand = torch.rand((3,3))

display('a random 3x3 tensor', rand)
display('twos, a 3x1 tensor', twos)

display('random tensor * 2', rand * 2)
display('random tensor * twos', rand * twos)

big_rand = torch.rand((2, 2, 2, 2))
display('big random tensor (2x2x2x2)', big_rand)

