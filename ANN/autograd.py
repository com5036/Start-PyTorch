import torch

w = torch.tensor(1.0, requires_grad=True)
a = w * 3
l = a ** 2

l.backward()

print(f'l을 w로 미분한 값은 {w.grad}')
