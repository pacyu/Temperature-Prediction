import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time


class Data(Dataset):
    def __init__(self):
        """
        加载数据
        """
        self.X = np.loadtxt('input_data.txt').astype(np.float32)
        self.y = np.loadtxt('output_data.txt').astype(np.float32)
        self.y_m = self.y.mean(axis=0)
        self.X = self.X / self.X.max(axis=0)
        self.y = self.y / self.y.mean(axis=0)

    def __getitem__(self, inx):
        return self.X[inx, :], self.y[inx, :]

    def __len__(self):
        return len(self.X)

    def base_mean(self):
        return self.y_m


class PolyRegress(nn.Module):
    """
    定义多元回归模型
    """
    def __init__(self, input_size, output_size):
        """
        :param input_size: 输入维度
        :param output_size: 输出维度
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred


EPOCH = 100
BATCH_SIZE = 3
LR = 0.001

ct_data = Data()
x_test = np.loadtxt('校验数据/校核-输入-影响因素.txt').astype(np.float32)
y_test = np.loadtxt('校验数据/校核-对比-实际温度.txt').astype(np.float32)
x_test = x_test / x_test.max(axis=0)
base_mean = y_test.mean(axis=0)
train_loader = DataLoader(ct_data, batch_size=BATCH_SIZE, shuffle=True)
# assert False
model = PolyRegress(5, 5)  # 回归模型
loss_func = nn.MSELoss()  # 默认损失
optimizer = torch.optim.SGD(model.parameters(), lr=LR)  # 随机梯度下降优化算法

start = time.process_time()
print('开始训练...')
losses = []
for epoch in range(EPOCH):
    for step, (X, y) in enumerate(train_loader):
        y_pred = model(X)
        loss = loss_func(y_pred, y)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:', epoch, 'Step: ', step, 'Loss:', loss.item())
        if loss.item() <= 1e-3:
            torch.save(model.state_dict(), 'poly_regress_epoch_%d.pth' % epoch)  # 保存训练模型
            break
    # if epoch % 5:
    #     torch.save(model.state_dict(), 'poly_regress_epoch_%d.pth' % epoch)  # 保存训练模型

duration = time.process_time() - start
print('训练结束!\n总共用时：%s' % duration)
w, b = model.parameters()
t_pred = np.abs(w[0].detach().numpy()) * x_test + np.abs(b[0].detach().numpy())
t_pred = base_mean * t_pred
plt.figure(0)
plt.plot(losses, label='loss')
plt.legend()
for _, (pred_t, vail_t) in enumerate(zip(t_pred.transpose(), y_test.transpose()), 1):
    plt.figure(_)
    plt.plot(pred_t, label='Predicted T %d' % _)
    plt.plot(vail_t, label='Compared T %d' % _)
    plt.legend()
plt.show()