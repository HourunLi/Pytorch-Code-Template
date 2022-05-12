import matplotlib.pyplot as plt
# from matplotlib import animation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
生成数据集 (无需填写代码)
其中C1 代表的是从高斯分布采样 (X,Y) ~ N(3,6,1,1,0)
其中C2 代表的是从高斯分布采样 (X,Y) ~ N(6,3,1,1,0)
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

dataset = np.concatenate((C1, C2), axis=0)

# print(C1)
# print("-----------------------------")
# print(C2)
# print("-----------------------------")
# print(*zip(*dataset))
# np.random.shuffle(dataset)

# 用于可视化数据集分布
# plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
# plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
# plt.show()


"""
建立模型 (需要填空哦！)
在这一部分，建立逻辑回归模型，定义loss函数以及单步梯度下降过程函数
填空一： 实现sigmoid的交叉熵损失函数( 不-要-直-接-用-pytorch-内-置-的-loss-函-数)
"""
epsilon = 1e-12


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        # nn.init.xavier_uniform_
        self.W = nn.Parameter(torch.randn(
            2, 1).uniform_(-0.1, 0.1).to(torch.float))
        self.b = nn.Parameter(torch.zeros(1).to(torch.float))

    def forward(self, inp):
        logit = torch.matmul(inp, self.W) + self.b  # shape (N,1)
        pred = torch.sigmoid(logit)  # <- 注意这里已经算了sigmoid
        return pred


def computer_loss(pred, label):
    pred = torch.squeeze(pred, dim=1)
    """
    填空一
    输入 label shape(N,) pred shape(N,)
    输出 losses (注意是losses)，shape(N,)，每一个样本一个loss
    #todo 实现sigmoid的交叉熵损失函数(不使用pytorch内置的交叉熵loss函数)
    """
    # -----------------------------------------
    losses = -(torch.mul(label, torch.log(pred)) +
               torch.mul(torch.ones(list(pred.size()))-label, torch.log(torch.ones(list(pred.size()))-pred)))
    # print(losses)
    # -----------------------------------------
    loss = torch.mean(losses)

    pred = torch.where(pred > 0.5, torch.ones_like(pred),
                       torch.zeros_like(pred))
    accuracy = torch.mean(torch.eq(pred, label).float())
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
    model = LogisticRegression()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 生成数据集
    x1, x2, y = list(zip(*dataset))
    x = torch.tensor(list(zip(x1, x2))).to(torch.float)
    y = torch.tensor(y)

    # 训练
    animation_fram = []
    for i in range(400):
        loss, accuracy, W_opt, b_opt = train_one_step(model, optimizer, x, y)
        animation_fram.append((W_opt.numpy()[0, 0], W_opt.numpy()[
                              1, 0], b_opt.numpy(), loss.numpy()))
        if i % 20 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')

    # 展示结果 version1
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    W_1 = model.W[0].item()
    W_2 = model.W[1].item()
    b = model.b[0].item()
    xx = np.arange(10, step=0.1)
    yy = - W_1 / W_2 * xx - b / W_2
    print(W_1, W_2, b)
    plt.plot(xx, yy)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

# 展示结果 version2
"""
    有兴趣的同学可以运行下述被注释代码并查看结果，非强制，无需填空
    会生成一个名为‘logistic_regression.mp4’的文件，打开可以查看每次迭代模型的变化
    注意：在运行之前记得安装ffmpeg并加入，否则会报'No MovieWriters available'类似的错误
    安装可以参考：https://www.cnblogs.com/Neeo/articles/11677715.html或者其他的说明
    """
"""
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('Logistic Regression Example', fontsize=15)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    line_d, = ax.plot([], [], label='fit_line')
    C1_dots, = ax.plot([], [], '+', c='b', label='actual_dots')
    C2_dots, = ax.plot([], [], 'o', c='g', label='actual_dots')
    frame_text = ax.text(0.02, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    def init():
        line_d.set_data([], [])
        C1_dots.set_data([], [])
        C2_dots.set_data([], [])
        return (line_d,) + (C1_dots,) + (C2_dots,)
    def animate(i):
        xx = np.arange(10, step=0.1)
        a = animation_fram[i][0]
        b = animation_fram[i][1]
        c = animation_fram[i][2]
        yy = a / -b * xx + c / -b
        line_d.set_data(xx, yy)
        C1_dots.set_data(C1[:, 0], C1[:, 1])
        C2_dots.set_data(C2[:, 0], C2[:, 1])
        frame_text.set_text('Timestep = %.1d/%.1d\nLoss = %.3f' % (i, len(animation_fram), animation_fram[i][3]))
        return (line_d,) + (C1_dots,) + (C2_dots,)


    anim = animation.FuncAnimation(f, animate, init_func=init,
                                   frames=len(animation_fram), interval=30, blit=True)
    anim.save('logistic_regression.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    """
