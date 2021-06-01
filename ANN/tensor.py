import torch


def print_description(x):
    print(x)
    print(f'Size:{x.size()}')
    print(f'Shape:{x.shape}')
    print(f'dimension:{x.ndimension()}')
    print()


# make tensor and print description
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print_description(x)

# 랭크 늘리기
x = torch.unsqueeze(x, 0)
print_description(x)

# 랭크 줄이기
x = torch.squeeze(x)
print_description(x)

# view
x = x.view(9)
print_description(x)
