import torch
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 데이터셋 생성
n_dim = 2
x_train, y_train = make_blobs(
    n_samples=80, n_features=n_dim,
    centers=[[1, 1], [1, -1], [-1, 1], [-1, -1]],
    shuffle=True,
    cluster_std=0.3
)

x_test, y_test = make_blobs(
    n_samples=20, n_features=n_dim,
    centers=[[1, 1], [1, -1], [-1, 1], [-1, -1]],
    shuffle=True,
    cluster_std=0.3
)


# label 값을 from 에서 to로 바꿈
def label_map(y_, from_, to_):
    y = np.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y


# 시각화
def vis_data(x, y=None, c='r'):
    if y is None:
        y = [None] * len(x)

    for x_, y_ in zip(x, y):
        # label 값이 없으면
        if y_ is None:
            plt.plot(x_[0], x_[1], '*', markerfacecolor='none', markeredgecolor=c)
        # 레이블 0 : o , 레이블 1 : +
        else:
            plt.plot(x_[0], x_[1], c + 'o' if y_ == 0 else c + '+')


# 레이블을 4개에서 2개로 줄임 (0, 1 -> 0) (2, 3 -> 1)
y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)

# 시각화
plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()

# torch.tensor 로 변환
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


# 신경망 모델 정의
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor):
        linear1 = self.linear_1(input_tensor)
        relu = self.relu(linear1)
        linear2 = self.linear_2(relu)
        output = self.sigmoid(linear2)
        return output


# 모델 생성
model = NeuralNet(2, 5)
learning_rate = 0.03
criterion = torch.nn.BCELoss()
epochs = 2000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 학습 전 평가
model.eval()  # 평가 모드로 바꿈
test_loss_before = criterion(model(x_test).squeeze(), y_test)
print(f'Before Training, test loss is {test_loss_before.item()}')

# 학습
for epoch in range(epochs):
    model.train()  # 학습 모드로 바꿈
    optimizer.zero_grad()  # 경사 값을 0로 만들어줌

    train_output = model(x_train)
    train_loss = criterion(train_output.squeeze(), y_train)

    if epoch % 100 == 0:
        print(f'Train loss at {epoch} is {train_loss.item()}')

    train_loss.backward()
    optimizer.step()

# 학습 후 평가
model.eval()
test_loss = criterion(torch.squeeze(model(x_test)), y_test)
print(f'After Training, test loss is {test_loss.item()}')

# 모델 저장
torch.save(model.state_dict(), './model.pt')
print(f'state_dict format of the model: {model.state_dict()}')

# 저장된 모델 불러옴
new_model = NeuralNet(2, 5)
new_model.load_state_dict(torch.load('./model.pt'))
new_model.eval()
print(f'벡터 [-1, 1]이 레이블 1을 가질 확률은 {new_model(torch.FloatTensor([-1, 1])).item()}')
