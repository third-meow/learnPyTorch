import torch
import torchvision

BAR = "~"*80

def display(title, thing):
    print(title)
    print('-' * (len(title) + 4))
    print(thing)
    print(BAR)


empty = torch.empty((3, 3))
display('empty', empty)

rand = torch.rand((3, 3))
display('rand', rand)

zeros = torch.zeros((3, 3))
display('zeros', zeros)

ones = zeros.new_ones((3, 3))
display('ones, from zeros with new_ones()', ones)

added_num = ones + 8
display('ones + 8', added_num)

from_lit = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
display('from literal', from_lit)

added_tensor = ones + from_lit
display('ones + from literal', added_tensor)




