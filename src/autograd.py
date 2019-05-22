import torch
import torchvision

BAR = "~"*80

def display(title, thing):
    print(title)
    print('-' * (len(title) + 4))
    print(thing)
    print(BAR)

print()

x = torch.ones((2,2), requires_grad=True)
display('x = 2x2 ones tensor', x)

y = x * 0.2
display('y = x * 0.2', y)

f = y * 0.2
display('f = y * 0.2', f)

z = f.mean()

z.backward()
display('x.grad after f.mean().backward()', x.grad)

print(BAR)

x = torch.ones((2,2), requires_grad=True)
display('x = 2x2 ones tensor', x)

y = x * 0.4
display('y = x * 0.4', y)

z = y.mean()

z.backward()
display('x.grad after y.mean().backward()', x.grad)

print(BAR)

x = torch.ones((3,3), requires_grad=True)
display('x = 3x3 ones tensor', x)

y = x * 0.4
display('y = x * 0.4', y)

z = y.mean()

z.backward()
display('x.grad after y.mean().backward()', x.grad)
