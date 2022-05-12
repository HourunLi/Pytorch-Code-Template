import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
生成数据集 (无需填写代码)
其中C1 代表的是从高斯分布采样 (X,Y) ~ N(3,6,1,1,0)
其中C2 代表的是从高斯分布采样 (X,Y) ~ N(6,3,1,1,0)
其中C3 代表的是从高斯分布采样 (X,Y) ~ N(7,7,1,1,0)
"""

dot_num = 100
x_p = np.random.normal(3., 1, dot_num)
y_p = np.random.normal(6., 1, dot_num)
y = np.ones(dot_num)
C1 = np.array([x_p, y_p, y]).T

x_n = np.random.normal(6., 1, dot_num)
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)
C2 = np.array([x_n, y_n, y]).T

x_b = np.random.normal(7., 1, dot_num)
y_b = np.random.normal(7., 1, dot_num)
y = np.ones(dot_num)*2
C3 = np.array([x_b, y_b, y]).T

dataset = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(dataset)

# 用于可视化数据集分布
# plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
# plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
# plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')
# plt.show()


"""
建立模型 (需要填空哦！)
在这一部分，建立模型，定义loss函数以及单步梯度下降过程函数
填空一： 在__init__构造函数中建立模型所需的参数
填空二： 实现softmax的交叉熵损失函数( 不-要-直-接-用-pytorch-内-置-的-loss-函-数)
"""
epsilon = 1e-12


class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.init.xavier_uniform_
        """
        #todo 填空一，构建模型所需的参数 self.W, self.b
        可以参考logistic_regression_exercise
        """
        # -----------------------------------------
        self.W = nn.Parameter(torch.randn(
            2, 3).to(torch.float64))
        self.b = nn.Parameter(torch.zeros(1).to(torch.float64))
        # -----------------------------------------

    def forward(self, inp):
        logit = torch.matmul(inp, self.W) + self.b  # shape (N, 3)
        pred = F.softmax(logit, dim=1)  # shape (N, 3)  #<-注意这里已经算了softmax
        return pred


def computer_loss(pred, label):
    one_hot_label = torch.zeros(label.shape[0], 3)
    one_hot_label.scatter_(1, label.unsqueeze(
        1), 1.0)  # onehot label shape (N,3)
    """
    填空二
    输入 label shape(N, 3) pred shape(N, 3)
    输出 losses (注意是losses)，shape(N,)，每一个样本一个loss
    #todo 实现交叉熵损失函数(不使用pytorch内置的softmax交叉熵损失函数)
    """
    # -----------------------------------------
    # print(one_hot_label.shape, pred.shape)
    losses = -torch.sum(torch.mul(one_hot_label, torch.log(pred)), 1)
    # -----------------------------------------
    loss = torch.mean(losses)
    accuracy = torch.mean(torch.eq(torch.argmax(pred, dim=-1), label).float())
    return loss, accuracy


def train_one_step(model, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss, accuracy = computer_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach(), accuracy.detach(), model.W.detach(), model.b.detach()


if __name__ == '__main__':
    # 定义模型与优化器
    model = SoftmaxRegression()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 生成数据集
    x1, x2, y = list(zip(*dataset))
    x = torch.tensor(list(zip(x1, x2))).to(torch.float64)
    y = torch.tensor(y).long()

    # 训练
    for i in range(1000):
        loss, accuracy, W_opt, b_opt = train_one_step(model, optimizer, x, y)
        if i % 50 == 49:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
            # print(W_opt, b_opt)

    # 展示结果
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')
    x = np.arange(0., 10., 0.1)
    y = np.arange(0., 10., 0.1)
    X, Y = np.meshgrid(x, y)
    inp = torch.tensor(list(zip(X.reshape(-1), Y.reshape(-1))))
    Z = model(inp)
    Z = Z.detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z)
    plt.show()
