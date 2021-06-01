import torch

w = torch.randn(5, 3, dtype=torch.float)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

print("w size", w.size())
print("x size", x.size())
print("w:", w)
print("x:", x)

b = torch.randn(5, 2, dtype=torch.float)
print("b size", b.size())
print("b:", b)

# 행렬 곱
wx = torch.mm(w, x)
print("wx size:", wx.size())
print("wx:", wx)

result = wx + b
print("result size:", result.size())
print("result:", result)

