#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 参数
learning_rate = 1e-4
dropout_rate = 0.5
max_epoch = 3
BATCH_SIZE = 50

'''
1.调用torchvision.datasets.MNIST 读取MNIST数据集 将数据包装为Dataset类
2.通过DataLoader将dataset变量变为迭代器
3.同样的方法处理训练和测试数据，设置BACTH_SIZE，思考train和test的时候是否需要shuffle
'''
# 下载MNIST数据集
DOWNLOAD_MNIST = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
test_loader = Data.DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


'''
写一个卷积网络
    __init__：初始化模型的地方，在这里声明模型的结构
    forward：调用模型进行计算，输入为x（按照batch_size组织好的样本数据），输出为模型预测结果
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=7, padding=3)  # Conv_1
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5)  # Conv_2
        self.out1 = nn.Linear(in_features=1600, out_features=1024)  # fc_3
        self.relu = nn.ReLU()  # relu
        self.dropout = nn.Dropout(p=dropout_rate)  # dropout
        self.out2 = nn.Linear(in_features=1024, out_features=10)  # fc_4

    def forward(self, x):
        # print(x.shape)
        conv1 = self.conv1(x)
        # print(conv1.shape)
        conv1_pooling = self.maxpool(conv1)
        # print(conv1_pooling.shape)
        conv2 = self.conv2(conv1_pooling)
        # print(conv2.shape)
        conv2_pooling = self.maxpool(conv2)
        # print(conv2_pooling.shape)
        flatten = torch.flatten(conv2_pooling, start_dim=1)
        # print(flatten.shape)
        out1 = self.out1(flatten)
        # print(out1.shape)
        out1_active = self.relu(out1)
        out1__drop = self.dropout(out1_active)
        out = self.out2(out1__drop)
        # out为x经过一系列计算后最后一层fc_4输出的logit
        output = F.softmax(out, dim=1)
        # print(output.shape)
        return output

# 测试 输入模型，遍历test_loader 输出模型的准确率
# 注意 在测试时需要禁止梯度回传 可以考虑eval()模式和torch.no_grad() 他们有区别吗？


def test(cnn):
    cnn.eval()
    test_correct = 0
    # 计算模型正确率
    for _, (test_x, test_y) in enumerate(test_loader):
        test_y_predict = cnn(test_x)
        test_y_predict = torch.max(test_y_predict, 1)[1]
        test_correct += int(torch.sum(test_y_predict == test_y))
        # print(test_correct)
    return float(100 * test_correct / len(test_data))


# 训练
def train(cnn):
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)  # 声明一个Adam的优化器
    loss_func = nn.CrossEntropyLoss()  # 实用CrossEntropyLoss
    for epoch in range(max_epoch):
        for step, (x, y) in enumerate(train_loader):
            cnn.train()
            y_predict = cnn(x)
            y_onehot = torch.zeros(y.shape[0], 10)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            loss = loss_func(y_predict, y_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 将模型调节为训练模式
            # 得到模型针对训练数据x的logit
            # 计算loss
            # 梯度回传
            # 更新优化器
            if step != 0 and step % 20 == 0:
                print("=" * 10, step, "=" * 5, "=" * 5,
                      "test accuracy is ", test(cnn), "=" * 10)


if __name__ == '__main__':
    cnn = CNN()
    train(cnn)
